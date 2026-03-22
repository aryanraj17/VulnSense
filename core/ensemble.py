"""
ensemble.py
-----------
Combines scores from CodeBERT, GNN, YARA, and AST
into a single final vulnerability verdict.
"""

import os
import sys

# ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


# ── Weights for each scanner ─────────────────────────────────────────────────
WEIGHTS = {
    'codebert': 0.50,
    'gnn'     : 0.25,
    'yara'    : 0.15,
    'ast'     : 0.10,
}

# ── Thresholds for final verdict ─────────────────────────────────────────────
VULN_THRESHOLD = 0.50

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ────────────────────────────────────────────────────────────────────────────
class CodeBERTScanner:

    def __init__(self):
        self.binary_model     = None
        self.multiclass_model = None
        self.tokenizer        = None
        self.label_map        = None
        self.idx_to_cwe       = None
        self._load_models()

    def _load_models(self):
        binary_path     = 'models/codebert/binary'
        multiclass_path = 'models/codebert/multiclass'
        map_path        = 'data/processed/cwe_label_map.json'

        if os.path.exists(binary_path):
            try:
                self.tokenizer    = RobertaTokenizer.from_pretrained(binary_path)
                self.binary_model = RobertaForSequenceClassification.from_pretrained(
                    binary_path
                )
                self.binary_model.to(DEVICE)
                self.binary_model.eval()
                print("  CodeBERT binary model loaded")
            except Exception as e:
                print(f"  CodeBERT binary load error: {e}")
        else:
            print("  CodeBERT binary model not found — run training first")

        if os.path.exists(multiclass_path):
            try:
                self.multiclass_model = RobertaForSequenceClassification.from_pretrained(
                    multiclass_path
                )
                self.multiclass_model.to(DEVICE)
                self.multiclass_model.eval()
                print("  CodeBERT multiclass model loaded")
            except Exception as e:
                print(f"  CodeBERT multiclass load error: {e}")

        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                self.label_map  = json.load(f)
                self.idx_to_cwe = {v: k for k, v in self.label_map.items()}

    def _calibrate_score(self, raw_prob: float) -> float:
        """
        Calibrates raw CodeBERT probability to reduce false positives.
        Applies a stricter curve — only very high probabilities stay high.
        """
        if raw_prob >= 0.99:
            return 0.95
        elif raw_prob >= 0.90:
            return 0.80
        elif raw_prob >= 0.70:
            return 0.60
        elif raw_prob >= 0.50:
            return 0.45
        else:
            return raw_prob * 0.5

    def predict(self, code: str) -> dict:
        if not self.binary_model or not self.tokenizer:
            return self._fallback_prediction(code)

        try:
            inputs = self.tokenizer(
                code,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                binary_outputs = self.binary_model(**inputs)
                binary_probs   = torch.softmax(binary_outputs.logits, dim=1)
                # ── calibrate score to reduce false positives ────────────────
                vuln_prob      = self._calibrate_score(float(binary_probs[0][1].cpu()))

            cwe_prediction = 'Unknown'
            cwe_confidence = 0.0

            if self.multiclass_model and vuln_prob > VULN_THRESHOLD:
                with torch.no_grad():
                    multi_outputs  = self.multiclass_model(**inputs)
                    multi_probs    = torch.softmax(multi_outputs.logits, dim=1)
                    cwe_idx        = int(torch.argmax(multi_probs[0]).cpu())
                    cwe_confidence = float(multi_probs[0][cwe_idx].cpu())

                if self.idx_to_cwe:
                    cwe_prediction = self.idx_to_cwe.get(cwe_idx, 'CWE-Other')

            return {
                'probability'   : vuln_prob,
                'is_vulnerable' : vuln_prob > VULN_THRESHOLD,
                'cwe_prediction': cwe_prediction,
                'cwe_confidence': cwe_confidence,
                'source'        : 'codebert'
            }

        except Exception as e:
            print(f"  CodeBERT prediction error: {e}")
            return self._fallback_prediction(code)

    def _fallback_prediction(self, code: str) -> dict:
        RISKY = [
            'strcpy', 'strcat', 'gets', 'sprintf',
            'system', 'exec', 'malloc', 'free',
        ]
        score = sum(0.15 for r in RISKY if r in code)
        score = min(score, 0.95)

        return {
            'probability'   : score,
            'is_vulnerable' : score > VULN_THRESHOLD,
            'cwe_prediction': 'CWE-119' if score > 0.5 else 'Safe',
            'cwe_confidence': score,
            'source'        : 'fallback'
        }


# ────────────────────────────────────────────────────────────────────────────
class EnsembleScanner:

    def __init__(self):
        print("Initializing Ensemble Scanner...")

        self.codebert = CodeBERTScanner()

        try:
            from core.yara_scanner import compile_rules, scan_code
            self.yara_rules = compile_rules('rules/custom')
            self.yara_scan  = scan_code
            print("  YARA scanner loaded")
        except Exception as e:
            self.yara_rules = None
            self.yara_scan  = None
            print(f"  YARA load error: {e}")

        try:
            from core.ast_parser import ASTParser
            self.ast_parser = ASTParser()
            print("  AST parser loaded")
        except Exception as e:
            self.ast_parser = None
            print(f"  AST load error: {e}")

        self.gnn_scanner = None
        self._load_gnn()

        print("Ensemble Scanner ready.\n")

    def _load_gnn(self):
        try:
            from core.gnn_scanner import GNNScanner
            gnn_path = 'models/gnn/gnn_model.pt'
            if os.path.exists(gnn_path):
                self.gnn_scanner = GNNScanner()
                print("  GNN scanner loaded")
            else:
                print("  GNN model not trained yet — will skip GNN score")
        except Exception as e:
            print(f"  GNN load error: {e}")

    def scan(self, code: str, lang: str = 'c') -> dict:
        print(f"  Running full scan ({lang})...")
        scores  = {}
        details = {}

        # ── 1. CodeBERT ──────────────────────────────────────────────────────
        cb_result          = self.codebert.predict(code)
        scores['codebert'] = cb_result['probability']
        details['codebert']= cb_result
        print(f"  CodeBERT score   : {scores['codebert']:.3f}")

        # ── 2. YARA ──────────────────────────────────────────────────────────
        yara_matches = []
        if self.yara_rules and self.yara_scan:
            yara_matches    = self.yara_scan(code, self.yara_rules)
            yara_score      = self._yara_to_score(yara_matches)
            scores['yara']  = yara_score
            details['yara'] = yara_matches
            print(f"  YARA score       : {scores['yara']:.3f} ({len(yara_matches)} matches)")
        else:
            scores['yara']  = 0.0
            details['yara'] = []

        # ── 3. AST ───────────────────────────────────────────────────────────
        if self.ast_parser:
            ast_features    = self.ast_parser.extract_features(code, lang)
            ast_score       = self._ast_to_score(ast_features)
            scores['ast']   = ast_score
            details['ast']  = ast_features
            dangerous_lines = self.ast_parser.get_dangerous_lines(code, lang)
            print(f"  AST score        : {scores['ast']:.3f}")
        else:
            scores['ast']   = 0.0
            details['ast']  = {}
            dangerous_lines = []

        # ── 4. GNN ───────────────────────────────────────────────────────────
        if self.gnn_scanner:
            gnn_result      = self.gnn_scanner.predict(code, lang)
            scores['gnn']   = gnn_result['probability']
            details['gnn']  = gnn_result
            print(f"  GNN score        : {scores['gnn']:.3f}")
        else:
            scores['gnn']   = scores['codebert']
            details['gnn']  = {'source': 'codebert_fallback'}

        # ── 5. Weighted ensemble ─────────────────────────────────────────────
        final_score = (
            scores['codebert'] * WEIGHTS['codebert'] +
            scores['gnn']      * WEIGHTS['gnn']      +
            scores['yara']     * WEIGHTS['yara']      +
            scores['ast']      * WEIGHTS['ast']
        )

        # YARA override — if 2+ HIGH/CRITICAL rules match, force vulnerable
        yara_critical = [
            m for m in yara_matches
            if m['severity'] in ('CRITICAL', 'HIGH')
        ]
        yara_override = len(yara_critical) >= 2

        is_vulnerable = final_score > VULN_THRESHOLD or yara_override

        cwe_prediction = details['codebert'].get('cwe_prediction', 'Unknown')
        if not is_vulnerable:
            cwe_prediction = 'Safe'

        print(f"  Final score      : {final_score:.3f}")
        print(f"  Verdict          : {'VULNERABLE' if is_vulnerable else 'SAFE'}")

        return {
            'final_score'      : round(final_score, 4),
            'is_vulnerable'    : is_vulnerable,
            'cwe_prediction'   : cwe_prediction,
            'individual_scores': scores,
            'yara_matches'     : yara_matches,
            'dangerous_lines'  : dangerous_lines,
            'details'          : details,
            'language'         : lang,
        }

    def _yara_to_score(self, matches: list) -> float:
        if not matches:
            return 0.0

        SEVERITY_SCORES = {
            'CRITICAL': 1.0,
            'HIGH'    : 0.8,
            'MEDIUM'  : 0.5,
            'LOW'     : 0.3,
        }

        scores = [SEVERITY_SCORES.get(m['severity'], 0.3) for m in matches]
        base   = max(scores)
        bonus  = min(len(matches) * 0.05, 0.20)
        return min(base + bonus, 1.0)

    def _ast_to_score(self, features: dict) -> float:
        if not features:
            return 0.0

        score      = 0.0
        n_dangerous = len(features.get('dangerous_calls', []))
        score      += min(n_dangerous * 0.15, 0.60)

        cyclomatic = features.get('cyclomatic', 1)
        if cyclomatic > 10:
            score += 0.20
        elif cyclomatic > 5:
            score += 0.10

        if features.get('has_pointer') and features.get('has_malloc'):
            score += 0.10

        if features.get('has_malloc') and features.get('has_free'):
            score += 0.10

        return min(score, 1.0)


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Testing Ensemble Scanner")
    print("=" * 55)

    scanner = EnsembleScanner()

    vulnerable_code = """
    void vulnerable_function(char *input, int size) {
        char buffer[64];
        char *ptr = malloc(size);
        strcpy(buffer, input);
        if (ptr != NULL) {
            memcpy(ptr, input, strlen(input));
        }
        free(ptr);
        system(input);
    }
    """

    safe_code = """
    int add_numbers(int a, int b) {
        if (a > 1000 || b > 1000) {
            return -1;
        }
        int result = a + b;
        return result;
    }
    """

    print("\n--- Vulnerable Code ---")
    result1 = scanner.scan(vulnerable_code, lang='c')
    print(f"\n  Final Score    : {result1['final_score']}")
    print(f"  Verdict        : {'🔴 VULNERABLE' if result1['is_vulnerable'] else '🟢 SAFE'}")
    print(f"  CWE            : {result1['cwe_prediction']}")
    print(f"  Dangerous lines: {result1['dangerous_lines']}")

    print("\n--- Safe Code ---")
    result2 = scanner.scan(safe_code, lang='c')
    print(f"\n  Final Score    : {result2['final_score']}")
    print(f"  Verdict        : {'🔴 VULNERABLE' if result2['is_vulnerable'] else '🟢 SAFE'}")
    print(f"  CWE            : {result2['cwe_prediction']}")
"""
explainer.py
------------
Provides explainability for vulnerability predictions using SHAP.
Highlights which lines and tokens contributed most to the prediction.
Works with both CodeBERT and fallback sklearn models.
"""

import numpy as np
import json
import os
from typing import Union


# ── SHAP import (graceful fallback if not installed) ─────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("  Explainer WARNING: shap not installed. Run: pip install shap")


# ── Transformers import ───────────────────────────────────────────────────────
try:
    import torch
    from transformers import RobertaTokenizer, RobertaForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ────────────────────────────────────────────────────────────────────────────
class VulnerabilityExplainer:
    """
    Explains why a piece of code was flagged as vulnerable.

    Two explanation methods:
    1. SHAP token importance  — which tokens drove the prediction
    2. Line risk scoring      — maps token importance back to line numbers
    """

    def __init__(self, model_dir: str = 'models/codebert/binary'):
        self.model      = None
        self.tokenizer  = None
        self.model_dir  = model_dir
        self.loaded     = False

        self._load_model()

    def _load_model(self):
        """Loads CodeBERT model and tokenizer if available."""
        if not TORCH_AVAILABLE:
            print("  Explainer: PyTorch not available")
            return

        if not os.path.exists(self.model_dir):
            print(f"  Explainer: model not found at {self.model_dir}")
            print("  Run CodeBERT training first.")
            return

        try:
            print(f"  Loading model from {self.model_dir}...")
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_dir)
            self.model     = RobertaForSequenceClassification.from_pretrained(
                self.model_dir
            )
            self.model.eval()
            self.loaded = True
            print("  Explainer: model loaded successfully")
        except Exception as e:
            print(f"  Explainer: could not load model — {e}")

    def get_token_importance(self, code: str) -> dict:
        """
        Uses attention weights from CodeBERT to score each token.

        Returns:
        {
            'tokens'     : ['void', 'func', '(', ...],
            'scores'     : [0.1, 0.8, 0.05, ...],
            'top_tokens' : [('strcpy', 0.92), ('buffer', 0.87), ...]
        }
        """
        if not self.loaded:
            return self._fallback_token_importance(code)

        try:
            import torch

            # tokenize the code
            inputs = self.tokenizer(
                code,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )

            # forward pass with attention output
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True
                )

            # get attention weights from last layer
            # shape: (batch, heads, seq_len, seq_len)
            attention = outputs.attentions[-1]

            # average across heads, take CLS token attention
            # CLS token (index 0) attends to all other tokens
            cls_attention = attention[0].mean(dim=0)[0]
            scores        = cls_attention.numpy()

            # get tokens
            token_ids = inputs['input_ids'][0].numpy()
            tokens    = self.tokenizer.convert_ids_to_tokens(token_ids)

            # normalize scores to 0-1
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

            # get top tokens sorted by importance
            token_score_pairs = list(zip(tokens, scores))
            token_score_pairs = [
                (t, float(s)) for t, s in token_score_pairs
                if t not in ['<s>', '</s>', '<pad>', 'Ċ', 'Ġ']
            ]
            top_tokens = sorted(token_score_pairs, key=lambda x: x[1], reverse=True)[:15]

            return {
                'tokens'    : tokens,
                'scores'    : scores.tolist(),
                'top_tokens': top_tokens,
                'method'    : 'attention'
            }

        except Exception as e:
            print(f"  Token importance error: {e}")
            return self._fallback_token_importance(code)

    def get_line_risk_scores(self, code: str) -> dict:
        """
        Maps token importance back to line numbers.
        Returns a risk score (0-1) for each line of code.

        Returns:
        {
            'line_scores'   : {1: 0.2, 2: 0.9, 3: 0.1, ...},
            'high_risk_lines': [2, 5, 7],
            'annotated_code' : [(line_num, code_line, risk_score), ...]
        }
        """
        if not self.loaded:
            return self._fallback_line_scores(code)

        try:
            import torch

            inputs = self.tokenizer(
                code,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_attentions=True
                )

            attention  = outputs.attentions[-1]
            cls_attn   = attention[0].mean(dim=0)[0]
            scores     = cls_attn.numpy()

            token_ids  = inputs['input_ids'][0].numpy()
            tokens     = self.tokenizer.convert_ids_to_tokens(token_ids)

            # map tokens back to lines using offsets
            # RobertaTokenizer encodes newlines as 'Ċ'
            line_scores  = {}
            current_line = 1
            line_score_accumulator = {}

            for token, score in zip(tokens, scores):
                if token in ['<s>', '</s>', '<pad>']:
                    continue

                # Ċ is newline in RoBERTa tokenizer
                if 'Ċ' in token:
                    current_line += 1
                    continue

                if current_line not in line_score_accumulator:
                    line_score_accumulator[current_line] = []

                line_score_accumulator[current_line].append(float(score))

            # average scores per line
            for line_num, score_list in line_score_accumulator.items():
                line_scores[line_num] = float(np.mean(score_list))

            # normalize to 0-1
            if line_scores:
                max_score = max(line_scores.values())
                min_score = min(line_scores.values())
                for k in line_scores:
                    line_scores[k] = (line_scores[k] - min_score) / (max_score - min_score + 1e-8)

            # identify high risk lines (top 25% by score)
            if line_scores:
                threshold       = np.percentile(list(line_scores.values()), 75)
                high_risk_lines = [l for l, s in line_scores.items() if s >= threshold]
            else:
                high_risk_lines = []

            # build annotated code list
            code_lines     = code.split('\n')
            annotated_code = []
            for i, line in enumerate(code_lines, 1):
                score = line_scores.get(i, 0.0)
                annotated_code.append((i, line, score))

            return {
                'line_scores'    : line_scores,
                'high_risk_lines': sorted(high_risk_lines),
                'annotated_code' : annotated_code,
                'method'         : 'attention'
            }

        except Exception as e:
            print(f"  Line risk error: {e}")
            return self._fallback_line_scores(code)

    def explain(self, code: str, prediction: dict) -> dict:
        """
        Full explanation combining token importance and line scores.

        Args:
            code       : source code string
            prediction : output from codebert_scanner or ensemble

        Returns complete explanation dict for Streamlit display.
        """
        token_info = self.get_token_importance(code)
        line_info  = self.get_line_risk_scores(code)

        # combine with AST dangerous lines if available
        try:
            from core.ast_parser import ASTParser
            ast_parser      = ASTParser()
            dangerous_lines = ast_parser.get_dangerous_lines(code)
        except Exception:
            dangerous_lines = []

        # merge AST dangerous lines into high risk lines
        all_risk_lines = list(set(
            line_info.get('high_risk_lines', []) + dangerous_lines
        ))

        return {
            'prediction'     : prediction,
            'top_tokens'     : token_info.get('top_tokens', []),
            'line_scores'    : line_info.get('line_scores', {}),
            'high_risk_lines': sorted(all_risk_lines),
            'annotated_code' : line_info.get('annotated_code', []),
            'dangerous_lines': dangerous_lines,
            'method'         : token_info.get('method', 'fallback'),
        }

    # ── Fallback methods (used before model is trained) ──────────────────────
    def _fallback_token_importance(self, code: str) -> dict:
        """
        Rule-based token importance when model is not available.
        Uses known dangerous keywords as importance signal.
        """
        RISKY_TOKENS = {
            'strcpy': 0.95, 'strcat': 0.90, 'gets': 0.95,
            'sprintf': 0.85, 'scanf': 0.80, 'memcpy': 0.80,
            'malloc': 0.70, 'free': 0.70, 'realloc': 0.75,
            'system': 0.90, 'exec': 0.90, 'popen': 0.85,
            'printf': 0.60, 'fprintf': 0.60,
            'NULL': 0.50, 'argv': 0.65, 'stdin': 0.65,
        }

        tokens     = code.split()
        scores     = []
        top_tokens = []

        for token in tokens:
            clean = token.strip('();,{}[]')
            score = RISKY_TOKENS.get(clean, 0.1)
            scores.append(score)
            if score > 0.5:
                top_tokens.append((clean, score))

        top_tokens = sorted(top_tokens, key=lambda x: x[1], reverse=True)[:15]

        return {
            'tokens'    : tokens,
            'scores'    : scores,
            'top_tokens': top_tokens,
            'method'    : 'fallback_rules'
        }

    def _fallback_line_scores(self, code: str) -> dict:
        """
        Rule-based line scoring when model is not available.
        """
        RISKY_PATTERNS = [
            'strcpy', 'strcat', 'gets', 'sprintf',
            'memcpy', 'malloc', 'free', 'system',
            'exec', 'popen', 'scanf', 'printf',
        ]

        code_lines     = code.split('\n')
        line_scores    = {}
        annotated_code = []

        for i, line in enumerate(code_lines, 1):
            score = 0.1
            for pattern in RISKY_PATTERNS:
                if pattern in line:
                    score = max(score, 0.8)
                    break
            line_scores[i]    = score
            annotated_code.append((i, line, score))

        high_risk_lines = [l for l, s in line_scores.items() if s >= 0.7]

        return {
            'line_scores'    : line_scores,
            'high_risk_lines': high_risk_lines,
            'annotated_code' : annotated_code,
            'method'         : 'fallback_rules'
        }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("Testing Explainer...")
    print("=" * 55)

    explainer = VulnerabilityExplainer()

    test_code = """
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

    test_prediction = {
        'is_vulnerable'  : True,
        'probability'    : 0.91,
        'cwe_prediction' : 'CWE-119',
        'severity'       : 'HIGH'
    }

    print("\nGetting token importance...")
    token_info = explainer.get_token_importance(test_code)
    print(f"  Method      : {token_info['method']}")
    print(f"  Top tokens  :")
    for token, score in token_info['top_tokens'][:8]:
        bar = '█' * int(score * 20)
        print(f"    {token:<15} {bar} {score:.3f}")

    print("\nGetting line risk scores...")
    line_info = explainer.get_line_risk_scores(test_code)
    print(f"  High risk lines: {line_info['high_risk_lines']}")
    print(f"\n  Line by line:")
    for line_num, line_text, score in line_info['annotated_code']:
        if line_text.strip():
            bar   = '█' * int(score * 10)
            print(f"    Line {line_num:2d} [{bar:<10}] {score:.2f}  {line_text.strip()[:50]}")
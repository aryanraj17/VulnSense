# VulnSense Project Status (Codebase Snapshot)

## 1) Project Purpose
VulnSense is an AI-assisted vulnerability detection system focused primarily on **C/C++** (with limited Python support in AST parsing and heuristic language detection). It combines multiple detection signals:
- Transformer classifier (GraphCodeBERT-based pipeline)
- GNN over AST-derived graphs
- Custom YARA signature rules
- AST structural heuristics
- Severity/risk scoring and remediation guidance
- Feedback collection for active-learning-driven retraining

---

## 2) High-Level Architecture

### Core pipeline flow
1. **Input code** is accepted as text/file and language is detected (`core/utils.py`).
2. **Ensemble scanner** (`core/ensemble.py`) runs:
   - CodeBERT/GraphCodeBERT classifier wrapper
   - YARA scanner
   - AST parser + heuristic AST risk score
   - GNN scanner (if model + deps available)
3. Scores are **weighted and combined** into final verdict (`VULN_THRESHOLD=0.50`).
4. **YARA override logic** can force vulnerable verdict for high-risk signature combinations.
5. Output can be enriched with:
   - **Severity + CVSS-style scoring** (`core/severity_scorer.py`)
   - **Explainability** (`core/explainer.py`)
   - **Auto-fix suggestion** (`core/autofix.py`)
6. User corrections are saved via **active learning feedback store** (`core/active_learning.py`) and can be prepared for retraining.

### Ensemble weighting (current)
- CodeBERT: **0.50**
- GNN: **0.25**
- YARA: **0.15**
- AST: **0.10**

If GNN is unavailable, GNN score falls back to CodeBERT score.

---

## 3) Completed Modules and Purpose

## `core/preprocessor.py`
- BigVul preprocessing pipeline.
- Cleans code, normalizes CWE labels, balances vulnerable/safe ratio, splits train/val/test, emits `cwe_label_map.json`.
- Target split approx 70/15/15.

## `core/codebert_trainer.py`
- Fine-tuning pipeline using `microsoft/graphcodebert-base`.
- Supports:
  - Binary vuln classification
  - Multi-class CWE classification
- Includes short vulnerable/safe snippet augmentation to improve small-snippet behavior.
- Includes temperature scaling helper.

## `core/gnn_scanner.py`
- GraphSAGE model for graph-based vulnerability detection.
- Builds AST graphs with 18-dim node features and bidirectional parent-child edges.
- Uses PyTorch Geometric if available.

## `core/ast_parser.py`
- tree-sitter based parsing for C and Python.
- Feature extraction for structural vulnerability signals.
- Dangerous function line extraction and lightweight AST visualization.

## `core/yara_scanner.py`
- Compiles custom `.yar` files and scans code via temporary file.
- Returns rule, CWE, severity, matched strings.

## `core/ensemble.py`
- Primary orchestrator combining detector outputs.
- Final weighted score + YARA override policy.
- Returns detailed breakdown and dangerous lines.

## `core/severity_scorer.py`
- Maps scan output to severity tiers + CVSS-style score.
- CWE-specific base severity and remediation advice.
- Produces summary + risk label.

## `core/explainer.py`
- Model explanation utilities (attention-based token/line importance).
- Fallback heuristics when model/dependencies unavailable.

## `core/autofix.py`
- LLM-based fix generation (Groq Llama 3.3 70B path intended).
- Rule-based fallback remediations for common CWE families.

## `core/active_learning.py`
- Feedback store (`data/feedback.json`) with FP/FN/confirmed labeling.
- Uncertainty sampling for selecting retraining data.
- Retrain trigger threshold currently set to 50 feedback items.

## `core/utils.py`
- Language detection (extension + regex patterns), file IO helpers, truncation and display utilities.

---

## 4) Trained Models, Artifacts, and Datasets Present

## Model artifacts present
- `models/codebert/binary_metrics.json` exists and indicates trained binary model metrics (accuracy/f1 around **0.933**).
- `models/gnn/gnn_model.pt` and `models/gnn/gnn_metrics.json` exist; GNN metrics are lower (accuracy/f1 around **0.68**).
- **Note:** `ensemble.py` expects model directories:
  - `models/codebert/binary`
  - `models/codebert/multiclass`
  These directories are not visible in current file listing, so runtime may rely on fallback behavior unless those folders are restored separately.

## Dataset artifacts present
- Processed splits exist:
  - `data/processed/train.json` (17,159 samples)
  - `data/processed/val.json` (3,666 samples)
  - `data/processed/test.json` (3,675 samples)
- Label map exists with 12 classes (`CWE-119 ... CWE-94`, `CWE-Other`, `Safe`).
- Feedback file exists with 3 stored entries.

## Source dataset assumptions
- Preprocessor expects `data/raw/bigvul.csv` (not present in tracked files list).
- Training docs/comments indicate BigVul as the core training source.

---

## 5) Scanner/Rule Coverage and Integrations

## Custom YARA rules present (`rules/custom/`)
- CWE-119 Buffer Overflow
- CWE-120 Buffer Copy issues
- CWE-125 OOB Read
- CWE-787 OOB Write
- CWE-416 Use-after-free
- CWE-476 Null pointer
- CWE-190 Integer overflow
- CWE-20 Improper input validation
- CWE-89 SQL injection
- CWE-94 Code injection
- `cwe_other`

## Integrations/libraries referenced
- PyTorch / Transformers
- tree-sitter / tree-sitter-languages
- PyTorch Geometric
- YARA
- SHAP
- Groq API client
- dotenv

---

## 6) End-to-End Workflow (Developer Mental Model)

1. **Prepare data**
   - Place BigVul CSV at `data/raw/bigvul.csv`.
   - Run preprocessing to generate balanced JSON splits + label map.

2. **Train models**
   - Train GraphCodeBERT binary/multiclass models via `core/codebert_trainer.py`.
   - Train/evaluate GNN via `core/gnn_scanner.py` training path.
   - Save checkpoints/artifacts under `models/`.

3. **Run detection**
   - Instantiate `EnsembleScanner`.
   - Scan code snippet/file.
   - Get per-scanner scores + final verdict + predicted CWE.

4. **Post-processing**
   - Pass scan output to `SeverityScorer` for severity/CVSS/remediation.
   - Optionally run `VulnerabilityExplainer` for token/line risk explanation.
   - Optionally run `AutoFixer` for patch suggestion.

5. **Feedback loop**
   - Store FP/FN/confirmed outcomes in feedback store.
   - Once threshold reached, prepare retraining set (`data/processed/retrain.json`).

---

## 7) Pending / Incomplete / Fragile Areas Visible in Code

- **UI entrypoint mismatch:** `README.md` says `streamlit run app.py`, but `app.py` is not present in repository snapshot.
- **CodeBERT artifact mismatch risk:** ensemble loader looks for `models/codebert/binary` and `.../multiclass`, but only metrics file is visible in `models/codebert/`.
- **AutoFixer Groq client initialization appears incomplete:** `_init_client()` logs key status but does not obviously assign a constructed client instance to `self.client`; this likely forces fallback mode unless set elsewhere.
- **YARA scanner top-level cleanup bug:** file contains a top-level `os.unlink(tmp_path)` inside suppressed context before `tmp_path` definition; harmless due to suppression, but indicates leftover/debug code smell.
- **Dependency-conditional behavior:** major functionality degrades gracefully when libs are missing (PyG/SHAP/tree-sitter/Groq), which is useful but can hide capability loss in production.
- **Language support asymmetry:** utils detect Java/JS but labels them partial (YARA-only), while primary ML path is tuned to C/C++.
- **No explicit orchestration/tests in repo root:** there are quick-tests inside modules, but no unified test suite or CI config in visible files.

---

## 8) Known Limitations (from implementation)

- Primarily optimized for C/C++ vulnerability patterns and BigVul-style functions.
- Score calibration and ensemble thresholds are static constants (manual tuning, no learned calibration layer).
- Attention-based explanation is approximate and not guaranteed to reflect causal feature attribution.
- Fallback heuristics may produce false positives/negatives on uncommon patterns.
- Rule-based auto-fix is regex-driven and may not preserve semantics for complex code.
- YARA scanning uses temporary file materialization; large-scale batch workflows may need optimization.

---

## 9) Folder Structure (Quick Orientation)

- `core/` — Main logic: scanning, training, parsing, severity, explainability, autofix, active learning.
- `data/`
  - `processed/` — train/val/test JSON + label map (+ retrain set when generated)
  - `feedback.json` — user correction memory
- `models/`
  - `codebert/` — transformer artifacts/metrics
  - `gnn/` — graph model + metrics
  - `grammars/` — compiled language grammar binaries
- `rules/custom/` — custom YARA signatures by CWE family
- `notebooks/` — utility scripts for grammar build/status
- `requirements.txt` — Python dependencies
- `README.md` — high-level project description and setup

---

## 10) Practical Developer Notes (Return-to-Project Checklist)

- Validate runtime prerequisites first: `tree-sitter-languages`, `yara-python`, `torch`, `transformers`, and optionally `torch-geometric`, `shap`, `groq`.
- Confirm model artifact paths expected by `core/ensemble.py` exist before demoing.
- Reconcile/restore the missing Streamlit entrypoint (`app.py`) or update README to actual launch command.
- If you want active learning retrain flow, ensure feedback volume reaches threshold (50) or temporarily lower threshold for dev tests.
- Consider hardening before production:
  - Add integration tests for ensemble pipeline.
  - Add artifact/path validation script.
  - Fix minor code hygiene issues in `yara_scanner.py` and `autofix.py` initialization path.

---

## 11) Current Snapshot Conclusion
The repository contains a **substantial, mostly implemented multi-engine vulnerability detection stack** with training pipelines and post-processing modules in place. The main gaps are around **runtime packaging/orchestration completeness** (missing app entrypoint, artifact path consistency, and some integration polish) rather than missing core algorithms.

"""
project_status.py
-----------------
Run this to get a complete snapshot of your project status.
Share the output with Claude to get updated on your progress.
"""

import os
import json
import subprocess
from pathlib import Path


def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "N/A"


def check_file(path):
    return "✅" if os.path.exists(path) else "❌"


def file_size(path):
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size > 1e6:
            return f"{size/1e6:.1f} MB"
        elif size > 1e3:
            return f"{size/1e3:.1f} KB"
        return f"{size} B"
    return "missing"


print("=" * 60)
print("  VulnSense — Project Status Report")
print("=" * 60)

# ── Git Info ──────────────────────────────────────────────────
print("\n📌 GIT STATUS")
print(f"  Branch       : {run_cmd('git branch --show-current')}")
print(f"  Last commit  : {run_cmd('git log -1 --format=%s')}")
print(f"  Commit date  : {run_cmd('git log -1 --format=%cd --date=short')}")
print(f"  Total commits: {run_cmd('git rev-list --count HEAD')}")
print(f"\n  Recent commits:")
commits = run_cmd('git log --oneline -10')
for line in commits.split('\n'):
    print(f"    {line}")

# ── Core Files ────────────────────────────────────────────────
print("\n📁 CORE FILES")
core_files = [
    ('preprocessor.py',      'core/preprocessor.py'),
    ('codebert_trainer.py',  'core/codebert_trainer.py'),
    ('ast_parser.py',        'core/ast_parser.py'),
    ('yara_scanner.py',      'core/yara_scanner.py'),
    ('explainer.py',         'core/explainer.py'),
    ('ensemble.py',          'core/ensemble.py'),
    ('severity_scorer.py',   'core/severity_scorer.py'),
    ('autofix.py',           'core/autofix.py'),
    ('gnn_scanner.py',       'core/gnn_scanner.py'),
    ('active_learning.py',   'core/active_learning.py'),
    ('utils.py',             'core/utils.py'),
    ('app.py (frontend)',    'app.py'),
]

for name, path in core_files:
    print(f"  {check_file(path)} {name:<25} {file_size(path)}")

# ── Models ────────────────────────────────────────────────────
print("\n🤖 TRAINED MODELS")
models = [
    ('CodeBERT Binary',     'models/codebert/binary/model.safetensors'),
    ('CodeBERT Multiclass', 'models/codebert/multiclass/model.safetensors'),
    ('GNN Model',           'models/gnn/gnn_model.pt'),
]

for name, path in models:
    print(f"  {check_file(path)} {name:<25} {file_size(path)}")

# ── Model Metrics ─────────────────────────────────────────────
print("\n📊 MODEL METRICS")
metrics_files = [
    ('Binary metrics',     'models/codebert/binary_metrics.json'),
    ('Multiclass metrics', 'models/codebert/multiclass_metrics.json'),
    ('GNN metrics',        'models/gnn/gnn_metrics.json'),
]

for name, path in metrics_files:
    if os.path.exists(path):
        with open(path) as f:
            m = json.load(f)
        print(f"  ✅ {name:<25} F1={m.get('f1',0):.4f}  Acc={m.get('accuracy',0):.4f}")
    else:
        print(f"  ❌ {name:<25} not found")

# ── Dataset ───────────────────────────────────────────────────
print("\n📂 DATASET")
datasets = [
    ('BigVul CSV',     'data/raw/bigvul.csv'),
    ('Devign CSV',     'data/raw/devign.csv'),
    ('CVEfixes CSV',   'data/raw/cvefixes.csv'),
    ('train.json',     'data/processed/train.json'),
    ('val.json',       'data/processed/val.json'),
    ('test.json',      'data/processed/test.json'),
    ('CWE label map',  'data/processed/cwe_label_map.json'),
]

for name, path in datasets:
    print(f"  {check_file(path)} {name:<25} {file_size(path)}")

# ── Processed Data Stats ──────────────────────────────────────
if os.path.exists('data/processed/train.json'):
    with open('data/processed/train.json') as f:
        train = json.load(f)
    with open('data/processed/val.json') as f:
        val = json.load(f)
    with open('data/processed/test.json') as f:
        test = json.load(f)

    print(f"\n  Processed split sizes:")
    print(f"    Train : {len(train)}")
    print(f"    Val   : {len(val)}")
    print(f"    Test  : {len(test)}")

    vuln  = sum(1 for x in train if x['label'] == 1)
    safe  = sum(1 for x in train if x['label'] == 0)
    print(f"    Train vuln/safe: {vuln}/{safe}")

# ── YARA Rules ────────────────────────────────────────────────
print("\n🔍 YARA RULES")
yar_files = list(Path('rules/custom').glob('*.yar')) if Path('rules/custom').exists() else []
print(f"  Total rule files: {len(yar_files)}")
for f in sorted(yar_files):
    print(f"    ✅ {f.name}")

# ── Feedback ──────────────────────────────────────────────────
print("\n💬 ACTIVE LEARNING FEEDBACK")
if os.path.exists('data/feedback.json'):
    with open('data/feedback.json') as f:
        feedback = json.load(f)
    fp = sum(1 for x in feedback if x['feedback_type'] == 'false_positive')
    fn = sum(1 for x in feedback if x['feedback_type'] == 'false_negative')
    co = sum(1 for x in feedback if x['feedback_type'] == 'confirmed')
    print(f"  Total feedback   : {len(feedback)}")
    print(f"  False positives  : {fp}")
    print(f"  False negatives  : {fn}")
    print(f"  Confirmed        : {co}")
else:
    print("  No feedback collected yet")

# ── Requirements ──────────────────────────────────────────────
print("\n📦 KEY PACKAGES")
packages = [
    'torch', 'transformers', 'torch_geometric',
    'tree_sitter', 'yara', 'groq', 'streamlit',
    'sklearn', 'shap', 'pandas', 'numpy'
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'installed')
        print(f"  ✅ {pkg:<20} {ver}")
    except ImportError:
        print(f"  ❌ {pkg:<20} not installed")

# ── Progress Summary ──────────────────────────────────────────
print("\n" + "=" * 60)
print("  PROGRESS SUMMARY")
print("=" * 60)

components = [
    ('Preprocessor',          os.path.exists('core/preprocessor.py')),
    ('CodeBERT trainer',      os.path.exists('core/codebert_trainer.py')),
    ('CodeBERT binary model', os.path.exists('models/codebert/binary/model.safetensors')),
    ('CodeBERT multiclass',   os.path.exists('models/codebert/multiclass/model.safetensors')),
    ('AST parser',            os.path.exists('core/ast_parser.py')),
    ('GNN scanner',           os.path.exists('core/gnn_scanner.py')),
    ('GNN model trained',     os.path.exists('models/gnn/gnn_model.pt')),
    ('YARA scanner',          os.path.exists('core/yara_scanner.py')),
    ('YARA rules (11)',       len(yar_files) >= 11),
    ('Explainability',        os.path.exists('core/explainer.py')),
    ('Ensemble scanner',      os.path.exists('core/ensemble.py')),
    ('Severity scorer',       os.path.exists('core/severity_scorer.py')),
    ('Auto-fix (Groq)',       os.path.exists('core/autofix.py')),
    ('Active learning',       os.path.exists('core/active_learning.py')),
    ('Language detector',     os.path.exists('core/utils.py')),
    ('Streamlit frontend',    os.path.exists('app.py')),
    ('Dataset processed',     os.path.exists('data/processed/train.json')),
]

done  = sum(1 for _, v in components if v)
total = len(components)
pct   = int(done / total * 100)

for name, status in components:
    icon = "✅" if status else "❌"
    print(f"  {icon} {name}")

bar   = "█" * (pct // 5) + "░" * (20 - pct // 5)
print(f"\n  Overall: [{bar}] {pct}% ({done}/{total})")
print("=" * 60)
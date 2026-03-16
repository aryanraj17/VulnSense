"""
preprocessor.py
---------------
Loads BigVul dataset, cleans it, balances it,
and saves train/val/test splits ready for CodeBERT training.
"""

import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split

# ── Column names matching bstee615/bigvul ────────────────────────────────────
CODE_COL  = 'func_before'   # actual C/C++ function code
LABEL_COL = 'vul'           # 1 = vulnerable, 0 = safe
CWE_COL   = 'CWE ID'        # e.g. "CWE-119"

# ── Top 10 CWE categories we classify into ──────────────────────────────────
TOP_CWES = [
    'CWE-119',   # Buffer overflow
    'CWE-120',   # Buffer copy without checking size
    'CWE-125',   # Out-of-bounds read
    'CWE-787',   # Out-of-bounds write
    'CWE-476',   # NULL pointer dereference
    'CWE-416',   # Use after free
    'CWE-190',   # Integer overflow
    'CWE-20',    # Improper input validation
    'CWE-89',    # SQL injection
    'CWE-94',    # Code injection
]

RANDOM_SEED     = 42
MIN_CODE_LENGTH = 50    # ignore functions shorter than this
MAX_CODE_LENGTH = 5000  # ignore functions longer than this (outliers)


# ────────────────────────────────────────────────────────────────────────────
def clean_code(code: str) -> str:
    """
    Light cleaning of raw C/C++ code.
    - Removes blank lines
    - Removes comment-only lines
    - Preserves actual code structure for CodeBERT
    """
    if not isinstance(code, str):
        return ""

    cleaned = []
    for line in code.split('\n'):
        stripped = line.strip()
        # skip blank lines
        if not stripped:
            continue
        # skip pure comment lines
        if stripped.startswith('//') or \
           stripped.startswith('/*') or \
           stripped.startswith('*'):
            continue
        cleaned.append(line)

    return '\n'.join(cleaned)


# ────────────────────────────────────────────────────────────────────────────
def normalize_cwe(row) -> str:
    """
    Maps a CWE ID to one of our TOP_CWES.
    - Safe samples (label=0) get 'Safe'
    - Vulnerable samples with known CWE keep their CWE
    - Everything else gets 'CWE-Other'
    """
    # safe code has no CWE
    if row[LABEL_COL] == 0:
        return 'Safe'

    cwe = row[CWE_COL]

    if not isinstance(cwe, str):
        return 'CWE-Other'

    # BigVul sometimes has multiple CWEs like "CWE-119 CWE-120"
    # we take the first one only
    first_cwe = cwe.strip().split()[0]

    return first_cwe if first_cwe in TOP_CWES else 'CWE-Other'


# ────────────────────────────────────────────────────────────────────────────
def load_and_clean(csv_path: str) -> pd.DataFrame:
    """
    Loads the BigVul CSV and performs cleaning.
    """
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw shape           : {df.shape}")

    # keep only what we need
    df = df[[CODE_COL, LABEL_COL, CWE_COL]].copy()

    # drop rows with missing code or label
    df = df.dropna(subset=[CODE_COL, LABEL_COL])
    print(f"  After null drop     : {df.shape}")

    # ensure label is integer
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    # clean the code
    print("  Cleaning code samples (this takes ~1 min)...")
    df[CODE_COL] = df[CODE_COL].apply(clean_code)

    # drop rows where cleaned code is too short or too long
    lengths = df[CODE_COL].str.len()
    df = df[(lengths >= MIN_CODE_LENGTH) & (lengths <= MAX_CODE_LENGTH)]
    print(f"  After length filter : {df.shape}")

    # normalize CWE labels
    print("  Normalizing CWE labels...")
    df['cwe_normalized'] = df.apply(normalize_cwe, axis=1)

    # rename to clean names
    df = df.rename(columns={
        CODE_COL:  'code',
        LABEL_COL: 'label',
    })

    # drop original CWE column (we have cwe_normalized now)
    df = df.drop(columns=[CWE_COL])

    print(f"\n  Label distribution:")
    print(df['label'].value_counts().to_string())
    print(f"\n  CWE distribution:")
    print(df['cwe_normalized'].value_counts().to_string())

    return df


# ────────────────────────────────────────────────────────────────────────────
def balance_dataset(df: pd.DataFrame, safe_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Balances the dataset so the model doesn't just learn to always say 'safe'.

    Strategy:
    - Keep ALL vulnerable samples
    - Keep safe samples = vulnerable count x safe_multiplier (default 1.5x)

    This gives a ~40/60 vulnerable/safe ratio which trains well.
    """
    vulnerable   = df[df['label'] == 1]
    safe         = df[df['label'] == 0]
    n_vuln       = len(vulnerable)
    n_safe_keep  = min(int(n_vuln * safe_multiplier), len(safe))

    print(f"\nBalancing dataset:")
    print(f"  Vulnerable         : {n_vuln}")
    print(f"  Safe (before)      : {len(safe)}")

    safe_sampled = safe.sample(n=n_safe_keep, random_state=RANDOM_SEED)
    balanced = pd.concat([vulnerable, safe_sampled])
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"  Safe (after)       : {n_safe_keep}")
    print(f"  Final dataset size : {len(balanced)}")

    return balanced


# ────────────────────────────────────────────────────────────────────────────
def split_and_save(df: pd.DataFrame, output_dir: str):
    """
    Splits into train / val / test  (70 / 15 / 15)
    Saves each split as a JSON file.
    Saves a CWE label map for decoding predictions later.
    """
    os.makedirs(output_dir, exist_ok=True)

    # first carve out test set
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=df['label']
    )

    # then split train and val from the remainder
    # 0.176 of 0.85 ≈ 0.15 of total
    train, val = train_test_split(
        train_val,
        test_size=0.176,
        random_state=RANDOM_SEED,
        stratify=train_val['label']
    )

    print(f"\nSplit sizes:")
    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")
    print(f"  Test  : {len(test)}")

    # save each split
    for name, split in [('train', train), ('val', val), ('test', test)]:
        path = os.path.join(output_dir, f'{name}.json')
        split.to_json(path, orient='records', indent=2)
        print(f"  Saved {path}")

    # ── Build and save CWE label map ────────────────────────────────────────
    # Maps category name → integer index
    # e.g. {"CWE-119": 0, "CWE-120": 1, ..., "Safe": 10, "CWE-Other": 11}
    all_cwes  = sorted(df['cwe_normalized'].unique().tolist())
    label_map = {cwe: idx for idx, cwe in enumerate(all_cwes)}

    map_path = os.path.join(output_dir, 'cwe_label_map.json')
    with open(map_path, 'w') as f:
        json.dump(label_map, f, indent=2)

    print(f"\n  CWE label map ({len(label_map)} classes):")
    for k, v in label_map.items():
        print(f"    {v} → {k}")
    print(f"  Saved {map_path}")

    return label_map


# ────────────────────────────────────────────────────────────────────────────
def run_preprocessing():
    print("=" * 55)
    print("  BigVul Preprocessing Pipeline")
    print("=" * 55)

    csv_path   = 'data/raw/bigvul.csv'
    output_dir = 'data/processed'

    if not os.path.exists(csv_path):
        print(f"\nERROR: {csv_path} not found.")
        print("Please save BigVul CSV to data/raw/bigvul.csv")
        return

    df        = load_and_clean(csv_path)
    df        = balance_dataset(df)
    label_map = split_and_save(df, output_dir)

    print("\n" + "=" * 55)
    print("  Preprocessing complete!")
    print(f"  Files in: {output_dir}/")
    print("    train.json")
    print("    val.json")
    print("    test.json")
    print("    cwe_label_map.json")
    print("  Next step: CodeBERT fine-tuning")
    print("=" * 55)


if __name__ == '__main__':
    run_preprocessing()
"""
codebert_trainer.py
-------------------
Fine-tunes Microsoft's GraphCodeBERT model on BigVul dataset.
GraphCodeBERT improves over CodeBERT by incorporating data flow
graphs during pre-training, giving better performance on both
short snippets and complex production functions.

Trains for both:
  1. Binary classification  — vulnerable vs safe
  2. Multi-class CWE        — which type of vulnerability

Saves the trained model to models/codebert/
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from tqdm import tqdm


# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = 'data/processed'
MODEL_SAVE_DIR = 'models/codebert'

# switched from codebert-base to graphcodebert-base
# graphcodebert uses data flow graphs during pre-training
# giving much better performance on short code snippets
GRAPHCODEBERT_MODEL = 'microsoft/graphcodebert-base'

MAX_LENGTH    = 512
BATCH_SIZE    = 64       # good balance for 4GB VRAM
EPOCHS        = 5
LEARNING_RATE = 2e-5
WARMUP_RATIO  = 0.1      # 10% of steps for warmup
WEIGHT_DECAY  = 0.01
RANDOM_SEED   = 42

# temperature scaling — reduces overconfidence
# values > 1 make predictions less extreme
TEMPERATURE = 1.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Short vulnerable snippets to augment training data ────────────────────────
# These teach the model to recognize short snippets
# which BigVul doesn't have enough of
SHORT_SNIPPETS = [
    # CWE-119 Buffer Overflow
    {'code': 'void f(char *s){char b[64];strcpy(b,s);}',
     'label': 1, 'cwe_normalized': 'CWE-119'},
    {'code': 'void f(char *s){char b[32];strcat(b,s);}',
     'label': 1, 'cwe_normalized': 'CWE-119'},
    {'code': 'void f(){char b[10];gets(b);}',
     'label': 1, 'cwe_normalized': 'CWE-119'},
    {'code': 'void f(char *s){char b[64];sprintf(b,s);}',
     'label': 1, 'cwe_normalized': 'CWE-20'},

    # CWE-476 NULL Pointer
    {'code': 'void f(int n){char *p=malloc(n);p[0]=1;}',
     'label': 1, 'cwe_normalized': 'CWE-476'},
    {'code': 'void f(){FILE *fp=fopen("x","r");fread(fp,1,1,fp);}',
     'label': 1, 'cwe_normalized': 'CWE-476'},

    # CWE-416 Use After Free
    {'code': 'void f(){char *p=malloc(10);free(p);p[0]=1;}',
     'label': 1, 'cwe_normalized': 'CWE-416'},
    {'code': 'void f(){int *p=malloc(4);free(p);*p=1;}',
     'label': 1, 'cwe_normalized': 'CWE-416'},

    # CWE-20 Format String
    {'code': 'void f(char *s){printf(s);}',
     'label': 1, 'cwe_normalized': 'CWE-20'},
    {'code': 'void f(char *s){fprintf(stderr,s);}',
     'label': 1, 'cwe_normalized': 'CWE-20'},

    # CWE-89 SQL Injection
    {'code': 'void f(char *s){char q[256];sprintf(q,"SELECT * FROM users WHERE name=\'%s\'",s);exec(q);}',
     'label': 1, 'cwe_normalized': 'CWE-89'},

    # CWE-94 Code Injection
    {'code': 'void f(char *s){system(s);}',
     'label': 1, 'cwe_normalized': 'CWE-94'},
    {'code': 'void f(char *s){char cmd[256];sprintf(cmd,"ls %s",s);system(cmd);}',
     'label': 1, 'cwe_normalized': 'CWE-94'},

    # Safe code snippets
    {'code': 'int add(int a,int b){if(a>INT_MAX-b)return -1;return a+b;}',
     'label': 0, 'cwe_normalized': 'Safe'},
    {'code': 'char* safe(const char *s){if(!s)return NULL;size_t l=strlen(s);if(l>1024)return NULL;char *d=malloc(l+1);if(!d)return NULL;strncpy(d,s,l);d[l]=0;return d;}',
     'label': 0, 'cwe_normalized': 'Safe'},
    {'code': 'int max(int a,int b){return a>b?a:b;}',
     'label': 0, 'cwe_normalized': 'Safe'},
    {'code': 'void swap(int *a,int *b){int t=*a;*a=*b;*b=t;}',
     'label': 0, 'cwe_normalized': 'Safe'},
    {'code': 'int clamp(int v,int lo,int hi){if(v<lo)return lo;if(v>hi)return hi;return v;}',
     'label': 0, 'cwe_normalized': 'Safe'},
]


# ── Dataset Class ─────────────────────────────────────────────────────────────
class VulnerabilityDataset(Dataset):
    """
    PyTorch Dataset that loads our processed JSON files.
    Supports data augmentation with short snippets.
    """

    def __init__(
        self,
        json_path  : str,
        tokenizer,
        label_map  : dict,
        mode       : str = 'binary',
        augment    : bool = False
    ):
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.mode      = mode

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # augment training data with short snippets
        if augment:
            # repeat short snippets multiple times to give them weight
            self.data = self.data + SHORT_SNIPPETS * 10
            print(f"  Augmented with {len(SHORT_SNIPPETS) * 10} short snippets")

        print(f"  Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item['code']

        encoding = self.tokenizer(
            code,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids      = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        if self.mode == 'binary':
            label = int(item['label'])
        else:
            cwe   = item.get('cwe_normalized', 'Safe')
            label = self.label_map.get(cwe, self.label_map.get('CWE-Other', 0))

        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'label'         : torch.tensor(label, dtype=torch.long)
        }


# ── Temperature Scaling ───────────────────────────────────────────────────────
class TemperatureScaler:
    """
    Applies temperature scaling to calibrate model probabilities.
    Dividing logits by T > 1 makes predictions less extreme.
    This fixes the overconfidence problem we saw with CodeBERT.
    """

    def __init__(self, temperature: float = TEMPERATURE):
        self.temperature = temperature

    def scale(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


# ── Metrics Helper ────────────────────────────────────────────────────────────
def compute_metrics(
    preds    : list,
    labels   : list,
    mode     : str,
    label_map: dict = None
) -> dict:
    acc       = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall    = recall_score(labels, preds, average='weighted', zero_division=0)

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")

    if mode == 'multiclass' and label_map:
        idx_to_cwe   = {v: k for k, v in label_map.items()}
        target_names = [idx_to_cwe[i] for i in range(len(label_map))]
        print("\n  Full classification report:")
        print(classification_report(
            labels, preds,
            target_names=target_names,
            zero_division=0
        ))

    return {
        'accuracy' : acc,
        'f1'       : f1,
        'precision': precision,
        'recall'   : recall
    }


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, scaler_amp):
    """
    One training epoch with mixed precision for speed.
    """
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="  Training", leave=False):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

        total_loss += loss.item()
        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler_amp.step(optimizer)
        scaler_amp.update()
        scheduler.step()

    return total_loss / len(loader)


# ── Evaluation Loop ───────────────────────────────────────────────────────────
def evaluate(model, loader, temp_scaler):
    """
    Evaluation with temperature scaling applied.
    """
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", leave=False):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            # apply temperature scaling before argmax
            scaled_logits = temp_scaler.scale(outputs.logits)
            preds         = torch.argmax(scaled_logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels, total_loss / len(loader)


# ── Main Training Function ────────────────────────────────────────────────────
def train_model(mode: str = 'binary'):
    """
    Full training pipeline for one mode.
    """
    print(f"\n{'='*55}")
    print(f"  Training GraphCodeBERT — mode: {mode.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"{'='*55}")

    torch.manual_seed(RANDOM_SEED)

    # load label map
    map_path = os.path.join(PROCESSED_DIR, 'cwe_label_map.json')
    with open(map_path, 'r') as f:
        label_map = json.load(f)

    num_labels = 2 if mode == 'binary' else len(label_map)
    print(f"  Number of classes: {num_labels}")

    # load tokenizer and model
    print(f"\n  Loading GraphCodeBERT tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(GRAPHCODEBERT_MODEL)
    model     = RobertaForSequenceClassification.from_pretrained(
        GRAPHCODEBERT_MODEL,
        num_labels=num_labels
    )
    model.to(DEVICE)

    # temperature scaler
    temp_scaler = TemperatureScaler(TEMPERATURE)

    # load datasets
    # augment only training data
    print(f"\n  Loading datasets...")
    train_dataset = VulnerabilityDataset(
        os.path.join(PROCESSED_DIR, 'train.json'),
        tokenizer, label_map, mode,
        augment=True    # add short snippets to training
    )
    val_dataset = VulnerabilityDataset(
        os.path.join(PROCESSED_DIR, 'val.json'),
        tokenizer, label_map, mode,
        augment=False
    )
    test_dataset = VulnerabilityDataset(
        os.path.join(PROCESSED_DIR, 'test.json'),
        tokenizer, label_map, mode,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    total_steps   = len(train_loader) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # mixed precision scaler
    scaler_amp = torch.amp.GradScaler()

    # training loop
    best_f1         = 0.0
    best_model_path = os.path.join(MODEL_SAVE_DIR, mode)
    os.makedirs(best_model_path, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n  Epoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler_amp
        )
        val_preds, val_labels, val_loss = evaluate(
            model, val_loader, temp_scaler
        )

        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Val metrics:")
        metrics = compute_metrics(val_preds, val_labels, mode, label_map)

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ Best model saved (F1: {best_f1:.4f})")

    # final test evaluation
    print(f"\n{'='*55}")
    print(f"  Final Test Evaluation — {mode.upper()}")
    print(f"{'='*55}")

    best_model = RobertaForSequenceClassification.from_pretrained(best_model_path)
    best_model.to(DEVICE)

    test_preds, test_labels, test_loss = evaluate(
        best_model, test_loader, temp_scaler
    )

    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test metrics:")
    test_metrics = compute_metrics(
        test_preds, test_labels, mode, label_map
    )

    # save metrics
    metrics_path = os.path.join(MODEL_SAVE_DIR, f'{mode}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    return test_metrics


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("GraphCodeBERT Fine-Tuning Pipeline")
    print(f"Model  : {GRAPHCODEBERT_MODEL}")
    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}")
    print(f"Batch  : {BATCH_SIZE}")
    print(f"Temp   : {TEMPERATURE}")

    # check processed data exists
    for fname in ['train.json', 'val.json', 'test.json', 'cwe_label_map.json']:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run preprocessor.py first.")
            exit(1)

    # train binary classifier
    binary_metrics = train_model(mode='binary')

    # train multiclass CWE classifier
    multiclass_metrics = train_model(mode='multiclass')

    print("\n" + "=" * 55)
    print("  All training complete!")
    print(f"  Binary    F1 : {binary_metrics['f1']:.4f}")
    print(f"  Multiclass F1: {multiclass_metrics['f1']:.4f}")
    print("  Models saved in models/codebert/")
    print("=" * 55)
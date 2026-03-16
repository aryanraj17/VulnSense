"""
codebert_trainer.py
-------------------
Fine-tunes Microsoft's CodeBERT model on BigVul dataset.
Trains for both:
  1. Binary classification  — vulnerable vs safe
  2. Multi-class CWE        — which type of vulnerability

Saves the trained model to models/codebert/
"""

import os
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


# ── Config ───────────────────────────────────────────────────────────────────
PROCESSED_DIR   = 'data/processed'
MODEL_SAVE_DIR  = 'models/codebert'
CODEBERT_MODEL  = 'microsoft/codebert-base'

MAX_LENGTH      = 512       # max tokens CodeBERT can handle
BATCH_SIZE      = 16         # lower if you get out-of-memory errors
EPOCHS          = 5
LEARNING_RATE   = 2e-5
WARMUP_STEPS    = 100
RANDOM_SEED     = 42

# Use GPU if available, otherwise CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Dataset Class ────────────────────────────────────────────────────────────
class VulnerabilityDataset(Dataset):
    """
    PyTorch Dataset that loads our processed JSON files.
    Each item returns tokenized code + its label.
    """

    def __init__(self, json_path: str, tokenizer, label_map: dict, mode: str = 'binary'):
        """
        Args:
            json_path  : path to train.json / val.json / test.json
            tokenizer  : CodeBERT tokenizer
            label_map  : CWE name → integer index
            mode       : 'binary' (vuln/safe) or 'multiclass' (CWE type)
        """
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.mode      = mode

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        print(f"  Loaded {len(self.data)} samples from {json_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item['code']

        # ── Tokenize the code ────────────────────────────────────────────────
        encoding = self.tokenizer(
            code,
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids      = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # ── Get label ────────────────────────────────────────────────────────
        if self.mode == 'binary':
            label = int(item['label'])
        else:
            # multiclass — use CWE category index
            cwe   = item.get('cwe_normalized', 'Safe')
            label = self.label_map.get(cwe, self.label_map.get('CWE-Other', 0))

        return {
            'input_ids'     : input_ids,
            'attention_mask': attention_mask,
            'label'         : torch.tensor(label, dtype=torch.long)
        }


# ── Metrics Helper ───────────────────────────────────────────────────────────
def compute_metrics(preds: list, labels: list, mode: str, label_map: dict = None):
    """
    Computes accuracy, F1, precision, recall.
    Prints a full classification report for multiclass mode.
    """
    acc       = accuracy_score(labels, preds)
    f1        = f1_score(labels, preds, average='weighted', zero_division=0)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall    = recall_score(labels, preds, average='weighted', zero_division=0)

    print(f"    Accuracy  : {acc:.4f}")
    print(f"    F1        : {f1:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")

    if mode == 'multiclass' and label_map:
        # reverse the label map for readable report
        idx_to_cwe = {v: k for k, v in label_map.items()}
        target_names = [idx_to_cwe[i] for i in range(len(label_map))]
        print("\n  Full classification report:")
        print(classification_report(
            labels, preds,
            target_names=target_names,
            zero_division=0
        ))

    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler()

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
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

    return total_loss / len(loader)


# ── Evaluation Loop ───────────────────────────────────────────────────────────
def evaluate(model, loader):
    """
    Runs model on validation/test set without updating weights.
    Returns all predictions and true labels.
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

            # get predicted class (highest logit)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    return all_preds, all_labels, avg_loss


# ── Main Training Function ────────────────────────────────────────────────────
def train_model(mode: str = 'binary'):
    """
    Full training pipeline for one mode.

    Args:
        mode: 'binary' or 'multiclass'
    """
    print(f"\n{'='*55}")
    print(f"  Training CodeBERT — mode: {mode.upper()}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*55}")

    torch.manual_seed(RANDOM_SEED)

    # ── Load label map ───────────────────────────────────────────────────────
    map_path  = os.path.join(PROCESSED_DIR, 'cwe_label_map.json')
    with open(map_path, 'r') as f:
        label_map = json.load(f)

    num_labels = 2 if mode == 'binary' else len(label_map)
    print(f"  Number of classes: {num_labels}")

    # ── Load tokenizer and model ─────────────────────────────────────────────
    print(f"\n  Loading CodeBERT tokenizer and model...")
    tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_MODEL)
    model     = RobertaForSequenceClassification.from_pretrained(
        CODEBERT_MODEL,
        num_labels=num_labels
    )
    model.to(DEVICE)

    # ── Load datasets ────────────────────────────────────────────────────────
    print(f"\n  Loading datasets...")
    train_dataset = VulnerabilityDataset(
        os.path.join(PROCESSED_DIR, 'train.json'),
        tokenizer, label_map, mode
    )
    val_dataset = VulnerabilityDataset(
        os.path.join(PROCESSED_DIR, 'val.json'),
        tokenizer, label_map, mode
    )
    test_dataset = VulnerabilityDataset(
        os.path.join(PROCESSED_DIR, 'test.json'),
        tokenizer, label_map, mode
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Optimizer and scheduler ──────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_steps   = len(train_loader) * EPOCHS
    scheduler     = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # ── Training loop ────────────────────────────────────────────────────────
    best_f1        = 0.0
    best_model_path = os.path.join(MODEL_SAVE_DIR, f'{mode}')
    os.makedirs(best_model_path, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n  Epoch {epoch}/{EPOCHS}")

        train_loss              = train_epoch(model, train_loader, optimizer, scheduler)
        val_preds, val_labels, val_loss = evaluate(model, val_loader)

        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        print(f"  Val metrics:")
        metrics = compute_metrics(val_preds, val_labels, mode, label_map)

        # save best model based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            print(f"  ✓ Best model saved (F1: {best_f1:.4f})")

    # ── Final test evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Final Test Evaluation — {mode.upper()}")
    print(f"{'='*55}")

    # load best saved model for test evaluation
    best_model = RobertaForSequenceClassification.from_pretrained(best_model_path)
    best_model.to(DEVICE)

    test_preds, test_labels, test_loss = evaluate(best_model, test_loader)
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test metrics:")
    test_metrics = compute_metrics(test_preds, test_labels, mode, label_map)

    # save metrics to JSON for later reference
    metrics_path = os.path.join(MODEL_SAVE_DIR, f'{mode}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\n  Metrics saved to {metrics_path}")

    return test_metrics


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("CodeBERT Fine-Tuning Pipeline")
    print(f"Device: {DEVICE}")

    # check processed data exists
    for fname in ['train.json', 'val.json', 'test.json', 'cwe_label_map.json']:
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run preprocessor.py first.")
            exit(1)

    # train binary classifier first
    binary_metrics = train_model(mode='binary')

    # then train multiclass CWE classifier
    multiclass_metrics = train_model(mode='multiclass')

    print("\n" + "="*55)
    print("  All training complete!")
    print(f"  Binary    F1 : {binary_metrics['f1']:.4f}")
    print(f"  Multiclass F1: {multiclass_metrics['f1']:.4f}")
    print("  Models saved in models/codebert/")
    print("="*55)
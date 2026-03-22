"""
gnn_scanner.py
--------------
Graph Neural Network based vulnerability scanner.
Converts code AST into a graph and uses GNN to detect
structural vulnerability patterns.

Architecture: GraphSAGE (Graph Sample and Aggregate)
Training data: BigVul dataset (same as CodeBERT)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


# ── Check PyTorch Geometric ───────────────────────────────────────────────────
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, global_mean_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("  GNN: torch-geometric not available")
    print("  Run: pip install torch-geometric")


# ── Config ────────────────────────────────────────────────────────────────────
GNN_MODEL_PATH = 'models/gnn/gnn_model.pt'
FEATURE_DIM    = 18
HIDDEN_DIM     = 128
OUTPUT_DIM     = 2
EPOCHS         = 100
LEARNING_RATE  = 0.001
BATCH_SIZE     = 64
RANDOM_SEED    = 42

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ────────────────────────────────────────────────────────────────────────────
class GraphSAGEModel(nn.Module):
    """
    GraphSAGE model for vulnerability detection.

    Architecture:
    Input features → SAGEConv → SAGEConv → SAGEConv →
    Global pooling → Linear → Output (vuln probability)

    GraphSAGE works by aggregating features from neighboring
    nodes — perfect for AST where parent nodes summarize children.
    """

    def __init__(
        self,
        input_dim : int = FEATURE_DIM,
        hidden_dim: int = HIDDEN_DIM,
        output_dim: int = OUTPUT_DIM
    ):
        super(GraphSAGEModel, self).__init__()

        self.conv1 = SAGEConv(input_dim,  hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim // 2)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 2)

        self.dropout = nn.Dropout(p=0.3)

        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, output_dim)

    def forward(self, x, edge_index, batch):
        """
        Forward pass through the GNN.

        Args:
            x          : node feature matrix [num_nodes, feature_dim]
            edge_index : graph connectivity [2, num_edges]
            batch      : batch assignment vector [num_nodes]
        """
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# ────────────────────────────────────────────────────────────────────────────
class GNNDataBuilder:
    """
    Converts code into PyTorch Geometric graph data objects.
    Uses AST parser to extract features and build graph structure.
    """

    def __init__(self):
        from core.ast_parser import ASTParser
        self.ast_parser = ASTParser()

    def code_to_graph(self, code: str, label: int = 0, lang: str = 'c'):
        """
        Converts a code string into a PyTorch Geometric Data object.

        Graph structure:
        - Nodes : AST node types with rich feature vectors
        - Edges : parent-child relationships in AST (bidirectional)
        """
        if not PYG_AVAILABLE:
            return None

        try:
            tree = self.ast_parser.parse(code, lang)
            if not tree:
                return self._empty_graph(label)

            nodes, edges = self._extract_graph(tree, code)

            if len(nodes) == 0:
                return self._empty_graph(label)

            x          = torch.tensor(nodes, dtype=torch.float)
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() \
                         if edges else torch.zeros((2, 0), dtype=torch.long)
            y          = torch.tensor([label], dtype=torch.long)

            return Data(x=x, edge_index=edge_index, y=y)

        except Exception:
            return self._empty_graph(label)

    def _extract_graph(self, tree, code: str):
        """
        Extracts nodes and edges from AST tree.
        Each node gets a rich feature vector based on its type
        and structural properties.
        """
        NODE_TYPE_MAP = {
            'function_definition'  : 0,
            'call_expression'      : 1,
            'pointer_declarator'   : 2,
            'subscript_expression' : 3,
            'assignment_expression': 4,
            'binary_expression'    : 5,
            'if_statement'         : 6,
            'return_statement'     : 7,
            'declaration'          : 8,
            'identifier'           : 9,
            'string_literal'       : 10,
            'number_literal'       : 11,
            'parameter_list'       : 12,
            'compound_statement'   : 13,
            'while_statement'      : 14,
            'for_statement'        : 15,
            'unary_expression'     : 16,
            'other'                : 17,
        }

        DANGEROUS_IDENTIFIERS = {
            'strcpy', 'strcat', 'gets', 'sprintf',
            'memcpy', 'malloc', 'free', 'system',
            'exec', 'popen', 'scanf',
        }

        nodes      = []
        edges      = []
        node_queue = [(tree.root_node, -1, 0)]
        node_idx   = 0

        while node_queue:
            node, parent_idx, depth = node_queue.pop(0)

            node_type    = node.type
            type_idx     = NODE_TYPE_MAP.get(node_type, 17)
            node_text    = node.text.decode('utf-8', errors='ignore').strip()
            is_dangerous = int(node_text in DANGEROUS_IDENTIFIERS)
            is_leaf      = int(node.child_count == 0)

            # ── Rich 18-dimensional feature vector ───────────────────────────
            feature     = [0.0] * 18

            # node type one-hot (index 0)
            feature[0]  = float(type_idx) / 17.0

            # dangerous function flag (index 1)
            feature[1]  = float(is_dangerous)

            # leaf node flag (index 2)
            feature[2]  = float(is_leaf)

            # structural features
            feature[3]  = float(node.child_count) / 10.0       # normalized child count
            feature[4]  = float(depth) / 20.0                  # normalized depth
            feature[5]  = float(len(node_text)) / 100.0        # normalized text length
            feature[6]  = float(node_text.count('(')) / 5.0    # function call indicator
            feature[7]  = float('*' in node_text)              # pointer indicator
            feature[8]  = float('[' in node_text)              # array access indicator
            feature[9]  = float(node_text.isdigit())           # numeric literal
            feature[10] = float('=' in node_text)              # assignment indicator
            feature[11] = float('+' in node_text or
                                 '-' in node_text)             # arithmetic indicator
            feature[12] = float(node_type == 'if_statement')   # conditional
            feature[13] = float(node_type == 'while_statement'
                                or node_type == 'for_statement') # loop
            feature[14] = float(node_type == 'return_statement') # return
            feature[15] = float('NULL' in node_text or
                                'null' in node_text)            # null check
            feature[16] = float(node_type == 'call_expression') # function call
            feature[17] = float(node_type == 'pointer_declarator') # pointer decl

            nodes.append(feature)

            # add bidirectional edges
            if parent_idx >= 0:
                edges.append([parent_idx, node_idx])
                edges.append([node_idx, parent_idx])

            current_idx = node_idx
            node_idx   += 1

            for child in node.children:
                node_queue.append((child, current_idx, depth + 1))

        return nodes, edges

    def _empty_graph(self, label: int):
        """Returns minimal graph when parsing fails."""
        x          = torch.zeros((1, FEATURE_DIM), dtype=torch.float)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        y          = torch.tensor([label], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, y=y)

    def build_dataset(
        self,
        json_path  : str,
        lang       : str = 'c',
        max_samples: int = 15000
    ):
        """
        Builds list of PyG Data objects from a JSON split file.
        Balances vulnerable and safe samples automatically.
        """
        print(f"  Building graph dataset from {json_path}...")

        with open(json_path, 'r') as f:
            data = json.load(f)

        vulnerable = [d for d in data if d['label'] == 1]
        safe       = [d for d in data if d['label'] == 0]
        n          = min(max_samples // 2, len(vulnerable), len(safe))
        samples    = vulnerable[:n] + safe[:n]

        graphs = []
        for sample in tqdm(samples, desc="  Converting to graphs"):
            graph = self.code_to_graph(
                sample['code'],
                sample['label'],
                lang
            )
            if graph is not None:
                graphs.append(graph)

        print(f"  Built {len(graphs)} graphs")
        return graphs


# ────────────────────────────────────────────────────────────────────────────
class GNNTrainer:
    """
    Handles GNN model training and evaluation.
    """

    def __init__(self):
        self.model   = GraphSAGEModel().to(DEVICE)
        self.builder = GNNDataBuilder()

    def train(self):
        """Full training pipeline."""
        print("=" * 55)
        print("  GNN Training Pipeline")
        print(f"  Device: {DEVICE}")
        print("=" * 55)

        torch.manual_seed(RANDOM_SEED)

        print("\nBuilding graph datasets...")
        train_graphs = self.builder.build_dataset(
            'data/processed/train.json', max_samples=15000
        )
        val_graphs   = self.builder.build_dataset(
            'data/processed/val.json',   max_samples=3000
        )
        test_graphs  = self.builder.build_dataset(
            'data/processed/test.json',  max_samples=3000
        )

        train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(val_graphs,   batch_size=BATCH_SIZE, shuffle=False)
        test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE, shuffle=False)

        optimizer = Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=20, gamma=0.5
        )

        best_f1 = 0.0
        os.makedirs('models/gnn', exist_ok=True)

        for epoch in range(1, EPOCHS + 1):
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                out   = self.model(batch.x, batch.edge_index, batch.batch)
                loss  = F.cross_entropy(out, batch.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            if epoch % 5 == 0:
                val_preds, val_labels = self._evaluate(val_loader)
                f1  = f1_score(
                    val_labels, val_preds,
                    average='weighted', zero_division=0
                )
                acc = accuracy_score(val_labels, val_preds)

                print(f"  Epoch {epoch:3d}/{EPOCHS} | "
                      f"Loss: {total_loss/len(train_loader):.4f} | "
                      f"Val F1: {f1:.4f} | "
                      f"Val Acc: {acc:.4f}")

                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(self.model.state_dict(), GNN_MODEL_PATH)
                    print(f"  ✓ Best model saved (F1: {best_f1:.4f})")

        # final test evaluation
        print(f"\n{'='*55}")
        print("  Final Test Evaluation")
        print(f"{'='*55}")

        best_model = GraphSAGEModel().to(DEVICE)
        best_model.load_state_dict(
            torch.load(GNN_MODEL_PATH, map_location=DEVICE)
        )

        test_preds, test_labels = self._evaluate(
            DataLoader(test_graphs, batch_size=BATCH_SIZE),
            model=best_model
        )

        test_f1  = f1_score(
            test_labels, test_preds,
            average='weighted', zero_division=0
        )
        test_acc = accuracy_score(test_labels, test_preds)

        print(f"  Test F1       : {test_f1:.4f}")
        print(f"  Test Accuracy : {test_acc:.4f}")

        metrics = {'f1': test_f1, 'accuracy': test_acc}
        with open('models/gnn/gnn_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n  Model saved to {GNN_MODEL_PATH}")
        return metrics

    def _evaluate(self, loader, model=None):
        """Runs evaluation on a data loader."""
        if model is None:
            model = self.model

        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch  = batch.to(DEVICE)
                out    = model(batch.x, batch.edge_index, batch.batch)
                preds  = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        return all_preds, all_labels


# ────────────────────────────────────────────────────────────────────────────
class GNNScanner:
    """
    Inference wrapper for trained GNN model.
    Used by ensemble.py for predictions.
    """

    def __init__(self):
        self.model   = None
        self.builder = GNNDataBuilder()
        self._load_model()

    def _load_model(self):
        if not PYG_AVAILABLE:
            return

        if not os.path.exists(GNN_MODEL_PATH):
            print(f"  GNN model not found at {GNN_MODEL_PATH}")
            print("  Run: python core/gnn_scanner.py to train")
            return

        try:
            self.model = GraphSAGEModel().to(DEVICE)
            self.model.load_state_dict(
                torch.load(GNN_MODEL_PATH, map_location=DEVICE)
            )
            self.model.eval()
            print("  GNN model loaded successfully")
        except Exception as e:
            print(f"  GNN load error: {e}")

    def predict(self, code: str, lang: str = 'c') -> dict:
        """
        Predicts vulnerability probability for code.

        Returns:
        {
            'probability'  : 0.73,
            'is_vulnerable': True,
            'source'       : 'gnn'
        }
        """
        if not self.model or not PYG_AVAILABLE:
            return self._fallback_predict(code)

        try:
            graph  = self.builder.code_to_graph(code, label=0, lang=lang)
            loader = DataLoader([graph], batch_size=1)
            batch  = next(iter(loader)).to(DEVICE)

            with torch.no_grad():
                out       = self.model(batch.x, batch.edge_index, batch.batch)
                probs     = F.softmax(out, dim=1)
                vuln_prob = float(probs[0][1].cpu())

            return {
                'probability'  : vuln_prob,
                'is_vulnerable': vuln_prob > 0.5,
                'source'       : 'gnn'
            }

        except Exception as e:
            print(f"  GNN predict error: {e}")
            return self._fallback_predict(code)

    def _fallback_predict(self, code: str) -> dict:
        """AST based fallback when model not available."""
        try:
            from core.ast_parser import ASTParser
            parser   = ASTParser()
            features = parser.extract_features(code)
            n_danger = len(features.get('dangerous_calls', []))
            score    = min(n_danger * 0.20, 0.90)
        except Exception:
            score = 0.5

        return {
            'probability'  : score,
            'is_vulnerable': score > 0.5,
            'source'       : 'fallback'
        }


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if not PYG_AVAILABLE:
        print("ERROR: torch-geometric not installed.")
        print("Run: pip install torch-geometric")
        exit(1)

    print("GNN Scanner — Training Mode")
    print(f"Device: {DEVICE}")

    trainer = GNNTrainer()
    metrics = trainer.train()

    print("\n" + "=" * 55)
    print("  GNN Training Complete!")
    print(f"  Final F1       : {metrics['f1']:.4f}")
    print(f"  Final Accuracy : {metrics['accuracy']:.4f}")
    print("=" * 55)
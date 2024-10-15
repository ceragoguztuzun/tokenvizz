import argparse
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import class_weight
from scipy import sparse
import random
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.metrics import matthews_corrcoef, roc_curve
from sklearn.metrics import matthews_corrcoef, make_scorer

def tune_threshold_for_mcc(y_true, y_probs):
    """
    Find the threshold that maximizes MCC by calculating MCC for a range of thresholds.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    best_mcc = -1
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold

    return best_threshold, best_mcc


# ----------------------------
# 1. Utility Functions
# ----------------------------

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def improved_one_hot_encode(seq, max_len=7):
    """
    Improved one-hot encoding for DNA sequences.
    """
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0.25,0.25,0.25,0.25]}
    seq = seq.upper()[:max_len]
    encoding = [mapping.get(char, [0,0,0,0]) for char in seq]
    encoding += [[0,0,0,0]] * (max_len - len(seq))  # Pad if necessary
    return [item for sublist in encoding for item in sublist]

# ----------------------------
# 2. Dataset Class
# ----------------------------

class ImprovedGraphDataset(Dataset):
    def __init__(self, base_dir, split, max_seq_len=7):
        self.graphs = []
        self.labels = []
        self.base_dir = base_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self._load_data()

    def __len__(self):
        """
        Returns the number of graphs in the dataset.
        """
        return len(self.graphs)

    def __getitem__(self, idx):
        """
        Returns the graph and corresponding label at index `idx`.
        """
        return self.graphs[idx], self.labels[idx]

    def _load_data(self):
        label_file = os.path.join(self.base_dir, f"data/NHEK/NHEK_{self.split}_label_mapping.txt")
        
        if not os.path.exists(label_file):
            print(f"[DEBUG] Label file {label_file} does not exist.")
            return

        with open(label_file, 'r') as f:
            label_lines = f.readlines()

        if not label_lines:
            print(f"[DEBUG] Label file {label_file} is empty.")
            return

        label_dict = {}
        for line in label_lines:
            if ':' not in line:
                print(f"[DEBUG] Skipping invalid line in label file: {line.strip()}")
                continue
            seq_label = line.strip().split(':')
            if len(seq_label) != 2:
                print(f"[DEBUG] Skipping improperly formatted line: {line.strip()}")
                continue
            seq_full_name, label = seq_label
            seq_full_name = seq_full_name.strip()
            label = label.strip()
            try:
                label = int(label)
                label_dict[seq_full_name] = label
            except ValueError:
                print(f"[DEBUG] Invalid label '{label}' for sequence '{seq_full_name}'. Skipping.")
                continue

        print(f"[DEBUG] Loaded {len(label_dict)} labels from {label_file}.")

        for seq_full_name, label in label_dict.items():
            try:
                seq_name, split_suffix = seq_full_name.rsplit('_NHEK_', 1)
            except ValueError:
                print(f"[DEBUG] Unable to parse split from seq_name '{seq_full_name}'. Skipping.")
                continue

            if split_suffix != self.split:
                print(f"[DEBUG] Split suffix '{split_suffix}' does not match current split '{self.split}'. Skipping.")
                continue

            attention_file = os.path.join(
                self.base_dir, 
                f"tokenviz/outputs/adjacency_matrices/NHEK_{self.split}/{seq_full_name}_attention_graph.npz"
            )
            
            node_file = os.path.join(
                self.base_dir,
                f"tokenviz/outputs/node_info/NHEK_{self.split}/{seq_full_name}_node_info.json"
            )
            
            if not os.path.exists(attention_file) or not os.path.exists(node_file):
                print(f"[DEBUG] Missing files for {seq_full_name}. Skipping.")
                continue

            try:
                attention_matrix = sparse.load_npz(attention_file).tocoo()
                if attention_matrix.nnz == 0:
                    print(f"[DEBUG] Attention matrix {attention_file} has no edges. Skipping {seq_full_name}.")
                    continue
            except Exception as e:
                print(f"[DEBUG] Failed to load attention matrix {attention_file}: {e}. Skipping {seq_full_name}.")
                continue

            edge_index = torch.tensor(np.vstack((attention_matrix.row, attention_matrix.col)), dtype=torch.long)
            edge_attr = torch.tensor(attention_matrix.data, dtype=torch.float).unsqueeze(1)

            try:
                with open(node_file, 'r') as f:
                    node_info = json.load(f)
            except Exception as e:
                print(f"[DEBUG] Failed to load node info from {node_file}: {e}. Skipping {seq_full_name}.")
                continue
            
            node_features_list = []
            promoter_indices = []
            enhancer_indices = []
            
            promoter_start, promoter_end, enhancer_start, enhancer_end = self.get_promoter_enhancer_positions(seq_full_name)

            for node_id, info in node_info.items():
                string_encoded = improved_one_hot_encode(info.get('string', 'N'), self.max_seq_len)
                position = info.get('position', '0-0')
                try:
                    start, end = map(int, position.split('-'))
                except ValueError:
                    start, end = 0, 0

                # Classify node as promoter or enhancer based on position
                node_pos = (start + end) // 2
                if promoter_start <= node_pos < promoter_end:
                    promoter_indices.append(node_id)
                elif enhancer_start <= node_pos < enhancer_end:
                    enhancer_indices.append(node_id)

                weighted_degree = float(info.get('weighted_degree', 0.0))
                gc_content = sum(1 for base in info.get('string', '') if base in ['G', 'C']) / len(info.get('string', ''))
                feature = string_encoded + [start, end, weighted_degree, gc_content]
                node_features_list.append(feature)
            
            if not node_features_list:
                continue

            # Filter out graphs that don't have both promoter and enhancer nodes
            if not promoter_indices or not enhancer_indices:
                #print(f"[DEBUG] Skipping {seq_full_name} as it does not have both promoter and enhancer nodes.")
                continue

            try:
                node_features = torch.tensor(node_features_list, dtype=torch.float)
            except ValueError as ve:
                print(f"[DEBUG] Failed to convert node features to tensor for {seq_full_name}: {ve}. Skipping.")
                continue

            expected_feature_size = 4 * self.max_seq_len + 4  # 4 bases * max_len + start + end + weighted_degree + gc_content
            if node_features.size(1) != expected_feature_size:
                print(f"[DEBUG] Unexpected feature size for {seq_full_name}. Expected {expected_feature_size}, got {node_features.size(1)}. Skipping.")
                continue

            self.graphs.append((node_features, edge_index, edge_attr))
            self.labels.append(label)
        
        print(f"[DEBUG] Total graphs loaded for split '{self.split}': {len(self.graphs)}")

    def get_promoter_enhancer_positions(self, seq_full_name):
        """
        Function to get the start and end positions of the promoter and enhancer regions.
        """
        # Dummy values; this function should be implemented based on your specific logic
        promoter_start, promoter_end = 0, 3000  # Placeholder
        enhancer_start, enhancer_end = 3010, 5010  # Placeholder
        return promoter_start, promoter_end, enhancer_start, enhancer_end


# ----------------------------
# 3. Enhanced GCN Model Definition
# ----------------------------
class ImprovedGCN(nn.Module):
    def __init__(self, num_node_features, hidden_channels, dropout=0.0):
        super(ImprovedGCN, self).__init__()
        self.conv1 = nn.Linear(num_node_features, hidden_channels)
        self.conv2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels * 2, 1)

    def forward(self, x, adj):
        if adj.dim() != 2:
            raise ValueError(f"adj should be 2D, got shape {adj.shape}")
        
        x = F.relu(self.conv1(torch.matmul(adj, x)))
        x = F.relu(self.conv2(torch.matmul(adj, x)))

        x_mean = torch.mean(x, dim=0)
        x_max, _ = torch.max(x, dim=0)
        x = torch.cat([x_mean, x_max])

        return self.classifier(x).squeeze()

# ----------------------------
# 4. Training and Evaluation Functions
# ----------------------------
def collate_fn(batch):
    graphs, labels = zip(*batch)
    max_nodes = max(graph[0].size(0) for graph in graphs)

    padded_features = []
    padded_adjs = []
    for node_features, edge_index, edge_attr in graphs:
        num_nodes = node_features.size(0)

        # Pad node features
        padded_feat = F.pad(node_features, (0, 0, 0, max_nodes - num_nodes))
        padded_features.append(padded_feat)

        # Create square adjacency matrix
        adj = torch.zeros(max_nodes, max_nodes)  # Square adjacency matrix

        # Fill adjacency matrix using edge_index and edge_attr
        # Ensure edge_attr is 1D and matches the number of edges in edge_index
        edge_attr = edge_attr.squeeze()

        if edge_attr.dim() != 1:
            raise ValueError(f"edge_attr should be 1D, got shape {edge_attr.shape}")

        adj[edge_index[0], edge_index[1]] = edge_attr  # Populate the adjacency matrix
        padded_adjs.append(adj)

    # Stack features and adjacency matrices for batching
    features = torch.stack(padded_features)
    adjs = torch.stack(padded_adjs)
    labels = torch.tensor(labels, dtype=torch.float)

    return features, adjs, labels

def train(model, loader, optimizer, scheduler, device, class_weights):
    model.train()
    total_loss = 0
    for features, adjs, labels in loader:
        features, adjs, labels = features.to(device), adjs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        batch_size = features.size(0)
        outputs = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            outputs[i] = model(features[i], adjs[i])
        
        loss = F.binary_cross_entropy_with_logits(outputs, labels, pos_weight=class_weights[1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for features, adjs, labels in loader:
            features, adjs, labels = features.to(device), adjs.to(device), labels.to(device)
            
            batch_size = features.size(0)
            batch_preds = torch.zeros(batch_size, device=device)
            for i in range(batch_size):
                batch_preds[i] = model(features[i], adjs[i])
            
            probs = torch.sigmoid(batch_preds)
            predictions.extend(probs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    fpr, tpr, thresholds = roc_curve(true_labels, predictions)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    pred_labels = (predictions >= optimal_threshold).astype(int)
    # Example of adjusting the threshold to 0.7

    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)
    auc = roc_auc_score(true_labels, predictions)
    mcc = matthews_corrcoef(true_labels, pred_labels)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'mcc': mcc
    }

class GCNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_channels=64, learning_rate=1e-3, weight_decay=1e-4, epochs=50, dropout=0.0, optimizer='AdamW'):
        self.hidden_channels = hidden_channels
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout = dropout
        self.optimizer = optimizer  # Fixing the attribute name here
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def fit(self, X, y, batch_size=128):
        self.classes_, y_indices = np.unique(y, return_inverse=True)
        # Update y to be indices (0, 1, ...) for consistency
        y = y_indices

        train_dataset = list(zip(X, y))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        num_node_features = X[0][0].shape[1]
        self.model = ImprovedGCN(num_node_features=num_node_features, hidden_channels=self.hidden_channels, dropout=self.dropout).to(self.device)

        optimizer = getattr(torch.optim, self.optimizer)(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learning_rate, epochs=self.epochs, steps_per_epoch=len(train_loader))

        class_weights = class_weight.compute_class_weight('balanced', classes=self.classes_, y=y)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(self.device)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for features, adjs, labels in train_loader:
                features, adjs, labels = features.to(self.device), adjs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                batch_size = features.size(0)
                outputs = torch.zeros(batch_size, device=self.device)
                for i in range(batch_size):
                    outputs[i] = self.model(features[i], adjs[i])

                # Use indices for labels
                loss = F.binary_cross_entropy_with_logits(outputs, labels.float(), pos_weight=class_weights_tensor[1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
        return self
    
    def predict(self, X):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for data in tqdm(X, desc="Predicting"):
                # Unpack the data correctly
                features, edge_index, edge_attr = data
                num_nodes = features.size(0)

                # Reconstruct the adjacency matrix
                adj = torch.zeros(num_nodes, num_nodes)
                adj[edge_index[0], edge_index[1]] = edge_attr.squeeze()

                features, adj = features.to(self.device), adj.to(self.device)
                output = self.model(features, adj)
                prob = torch.sigmoid(output).item()
                predicted_label = self.classes_[1] if prob >= 0.5 else self.classes_[0]
                predictions.append(predicted_label)
        return np.array(predictions)

    def score(self, X, y):
        y_pred = self.predict(X)
        return matthews_corrcoef(y, y_pred)
    
    def predict_proba(self, X):
        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for data in tqdm(X, desc="Predicting Probabilities"):
                features, edge_index, edge_attr = data
                num_nodes = features.size(0)

                # Reconstruct the adjacency matrix
                adj = torch.zeros(num_nodes, num_nodes)
                adj[edge_index[0], edge_index[1]] = edge_attr.squeeze()

                features, adj = features.to(self.device), adj.to(self.device)
                output = self.model(features, adj)
                prob = torch.sigmoid(output).item()
                probabilities.append([1 - prob, prob])
        return np.array(probabilities)




def main(args):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    print("Loading datasets...")
    train_dataset = ImprovedGraphDataset(base_dir=args.base_dir, split="train", max_seq_len=10)
    dev_dataset = ImprovedGraphDataset(base_dir=args.base_dir, split="dev", max_seq_len=10)
    test_dataset = ImprovedGraphDataset(base_dir=args.base_dir, split="test", max_seq_len=10)
    
    if len(train_dataset) == 0:
        print("No training data found. Please check your data paths and files.")
        return

    # Prepare data for grid search
    X_train = [graph for graph, _ in train_dataset]
    y_train = np.array([label for _, label in train_dataset])
    
    # Define parameter grid for GridSearchCV
    param_grid = {
        'hidden_channels': [64, 128, 256, 512],
        'learning_rate': [1e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        'weight_decay': [1e-6, 1e-5, 1e-4, 1e-3],
        'epochs': [50, 100, 150],
        'dropout': [0.2, 0.3, 0.5],
        'optimizer': ['AdamW', 'SGD', 'RMSprop']  # Corrected 'RMSProp' to 'RMSprop'
    }



    
    # Perform grid search
    print("Starting grid search...")
    gcn_classifier = GCNClassifier()
    #grid_search = GridSearchCV(gcn_classifier, param_grid, cv=3, scoring='matthews_corrcoef', n_jobs=1, verbose=2)

    mcc_scorer = make_scorer(matthews_corrcoef)
    grid_search = GridSearchCV(
        GCNClassifier(),
        param_grid,
        cv=3,
        scoring=mcc_scorer,
        n_jobs=1,
        verbose=2,
        error_score='raise'
    )

    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best MCC score: ", grid_search.best_score_)
    
    # Train final model with best parameters
    print("Training final model with best parameters...")
    best_model = GCNClassifier(**grid_search.best_params_)
    best_model.fit(X_train, y_train)
    
    # ---- Threshold tuning ----
    print("Tuning threshold based on MCC...")

    # Evaluate on dev set and get probabilities
    X_dev = [graph for graph, _ in dev_dataset]
    y_dev = np.array([label for _, label in dev_dataset])
    y_dev_probs = best_model.predict_proba(X_dev)  # Predict probabilities on the dev set
    
    # Find the best threshold for maximizing MCC
    best_threshold, best_mcc = tune_threshold_for_mcc(y_dev, y_dev_probs)
    print(f"Best threshold for MCC: {best_threshold}, Best MCC: {best_mcc}")

    # ---- Applying the best threshold to test set ----
    print("Evaluating on test set...")
    X_test = [graph for graph, _ in test_dataset]
    y_test = np.array([label for _, label in test_dataset])
    y_test_probs = best_model.predict_proba(X_test)  # Predict probabilities on the test set
    
    # Apply the best threshold found from the dev set
    y_test_pred = (y_test_probs >= best_threshold).astype(int)
    
    # Calculate test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_test_pred)
    }
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improved GCN for Small Graphs with Grid Search")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing data")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    
    args = parser.parse_args()
    main(args)
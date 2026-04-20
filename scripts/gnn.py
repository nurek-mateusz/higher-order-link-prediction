import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys
import random
import numpy as np
import pickle
import networkx as nx
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.under_sampling import RandomUnderSampler
import os


def create_node_mapping(G):
        unique_nodes = sorted(G.nodes())
        id_map = {node_id: i for i, node_id in enumerate(unique_nodes)}
        return id_map

def nx_to_edge_index(G, id_map):
        edge_index = []
        for u, v in G.edges():
                edge_index.append([id_map[u], id_map[v]])
                edge_index.append([id_map[v], id_map[u]])  # undirected
        return edge_index

def get_data_and_node_mapping(dataset, mode):
        # Read candidate triangles
        with open(f'../data/processed/{dataset}/trg_open_{mode}.pickle', 'rb') as file:
            candidates = pickle.load(file)

        # Labels
        with open(f'../data/processed/{dataset}/y_{mode}.pickle', 'rb') as file:
            labels = pickle.load(file)

        # Graph
        with open(f'../data/processed/{dataset}/G_{mode}.pickle', 'rb') as file:
            G = pickle.load(file)

        # Map node IDs to 0,1,...,N
        id_map = create_node_mapping(G)
        mapped_edge_indexes = nx_to_edge_index(G, id_map)
        mapped_candidates = [(id_map[u], id_map[v], id_map[w]) for (u, v, w) in candidates]

        return G, mapped_edge_indexes, mapped_candidates, labels

def under_sample(x_train, y_train, random_state, ratio=1):
    rus = RandomUnderSampler(sampling_strategy=ratio, random_state=random_state)
    x_resampled, y_resampled = rus.fit_resample(x_train, y_train)
    return x_resampled, y_resampled


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, node_features, edge_index):
        h = self.conv1(node_features, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h
    
class TrianglePooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, triangles):
        hu = h[triangles[:, 0]]
        hv = h[triangles[:, 1]]
        hw = h[triangles[:, 2]]
        z = torch.cat([
            hu, hv, hw,
            hu * hv, hv * hw, hu * hw
        ], dim=1)
        return z
    
class ClosureEventPredictor(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, dropout):
        super().__init__()
        self.gcn = GCN(in_dim=node_feat_dim, hidden_dim=hidden_dim, dropout=dropout)

        self.pool = TrianglePooling()

        self.mlp = nn.Sequential(
            nn.Linear(6*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, node_features, edge_index, triangles):
        h = self.gcn(node_features, edge_index)
        z = self.pool(h, triangles)
        logits = self.mlp(z).squeeze(-1)
        return logits



# --- MAIN ---
if __name__ == "__main__":
    RANDOM_STATE = 0
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    # dataset = 'coauth-MAG-History'
    # model_name = 'gcn'

    # --- PARSE ARGUMENTS ---
    if len(sys.argv) != 2:
        raise Exception('Wrong number of arguments')

    dataset = sys.argv[1]
    if dataset not in ['coauth-MAG-Geology', 'coauth-MAG-History', 'contact-high-school', 'contact-primary-school',
                        'email-Enron', 'email-Eu', 'NDC-classes', 'NDC-substances', 'tags-ask-ubuntu', 'threads-ask-ubuntu']:
        raise Exception('Wrong dataset name')

    model_name = 'gcn'

    print(f'PARAMS: {dataset} {model_name}')

    G_train, edge_index_train, candidates_train, labels_train = get_data_and_node_mapping(dataset, 'train')
    G_val, edge_index_val, candidates_val, labels_val = get_data_and_node_mapping(dataset, 'val')
    G_test, edge_index_test, candidates_test, labels_test = get_data_and_node_mapping(dataset, 'test')

    try:
        under_sampled_candidates_train, under_sampled_labels_train = under_sample(candidates_train, labels_train, random_state=RANDOM_STATE, ratio=0.33)
    except ValueError as e:
        # IF ValueError: The specified ratio required to generate new sample in the majority class while trying to remove samples. Please increase the ratio.
        # DO NOTHING - undersampling is not necessery because class are more or less equal
        print(e)
        # Use original data if undersampling fails
        under_sampled_candidates_train = candidates_train
        under_sampled_labels_train = labels_train
        

    # -- CREATE CONSTANT NODE FEATURES ---
    num_nodes_train = len(G_train.nodes())
    node_features_train = torch.ones(num_nodes_train, 1)

    num_nodes_val = len(G_val.nodes())
    node_features_val = torch.ones(num_nodes_val, 1)

    num_nodes_test = len(G_test.nodes())
    node_features_test = torch.ones(num_nodes_test, 1)

    node_feat_dim = 1

    # --- CONVERT TRAIN/VAL DATA TO TENSORS ---
    tensor_candidates_train = torch.tensor(under_sampled_candidates_train, dtype=torch.long)
    tensor_labels_train = torch.tensor(under_sampled_labels_train, dtype=torch.float32)
    tensor_edge_index_train = torch.tensor(edge_index_train, dtype=torch.long).t().contiguous()

    tensor_candidates_val = torch.tensor(candidates_val, dtype=torch.long)
    tensor_labels_val = torch.tensor(labels_val, dtype=torch.float32)
    tensor_edge_index_val = torch.tensor(edge_index_val, dtype=torch.long).t().contiguous()

    # -- HYPERPARAMETER GRID -- 
    param_grid = {
        "hidden_dim": [32, 64, 128],
        "lr": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "weight_decay": [1e-4, 5e-4]
    }

    epochs_train = 50
    epochs_test = 100

    best_score = -1
    best_params = None

    for hidden_dim in param_grid['hidden_dim']:
        for lr in param_grid['lr']:
            for dropout in param_grid['dropout']:
                for weight_decay in param_grid['weight_decay']:
                    model = ClosureEventPredictor(node_feat_dim, hidden_dim, dropout)

                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                    loss_fn = nn.BCEWithLogitsLoss()

                    # --- TRAINING LOOP ---
                    for epoch in range(epochs_train):
                        model.train()
                        optimizer.zero_grad()

                        logits = model(node_features_train, tensor_edge_index_train, tensor_candidates_train)
                        loss = loss_fn(logits, tensor_labels_train)

                        loss.backward()
                        optimizer.step()

                    # --- PARAMS EVALUATION ON VALIDATION SET ---
                    model.eval()
                    with torch.no_grad():
                        val_logits = model(node_features_val, tensor_edge_index_val, tensor_candidates_val)
                        val_probs = torch.sigmoid(val_logits).cpu().numpy()

                        avg_prec = average_precision_score(tensor_labels_val.cpu().numpy(), val_probs)
                        performance = avg_prec / (sum(tensor_labels_val.cpu().numpy()) / len(tensor_labels_val.cpu().numpy())) # avg_prec / random_baseline

                    if performance > best_score:
                        best_score = performance
                        best_params = {'hidden_dim': hidden_dim, 'lr': lr, 'dropout': dropout, 'weight_decay': weight_decay}

                    print(f"hd={hidden_dim}, lr={lr}, drop={dropout}, weight_decay={weight_decay} -> val Performance={performance:.4f}")

    print("Best params:", best_params)
    print("Best val performance:", best_score)

    # -- TEST ---
    model = ClosureEventPredictor(node_feat_dim, best_params['hidden_dim'], best_params['dropout'])
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()

    try:
        under_sampled_candidates_val, under_sampled_labels_val = under_sample(candidates_val, labels_val, random_state=RANDOM_STATE, ratio=0.33)
    except ValueError as e:
        # IF ValueError: The specified ratio required to generate new sample in the majority class while trying to remove samples. Please increase the ratio.
        # DO NOTHING - undersampling is not necessery because class are more or less equal
        print(e)
        # Use original data if undersampling fails
        under_sampled_candidates_val = candidates_val
        under_sampled_labels_val = labels_val

    # -- CONVERT VAL/TEST DATA TO TENSORS ---
    tensor_candidates_val = torch.tensor(under_sampled_candidates_val, dtype=torch.long)
    tensor_labels_val = torch.tensor(under_sampled_labels_val, dtype=torch.float32)
    tensor_edge_index_val = torch.tensor(edge_index_val, dtype=torch.long).t().contiguous()

    tensor_candidates_test = torch.tensor(candidates_test, dtype=torch.long)
    tensor_labels_test= torch.tensor(labels_test, dtype=torch.float32)
    tensor_edge_index_test = torch.tensor(edge_index_test, dtype=torch.long).t().contiguous()

    # --- TRAIN FINAL MODEL --
    for epoch in range(epochs_test):
        model.train()
        optimizer.zero_grad()

        logits = model(node_features_val, tensor_edge_index_val, tensor_candidates_val)
        loss = loss_fn(logits, tensor_labels_val)

        loss.backward()
        optimizer.step()

    # --- EVALUATE FINAL MODEL ---
    model.eval()
    with torch.no_grad():
        test_logits = model(node_features_test, tensor_edge_index_test, tensor_candidates_test)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        avg_prec = average_precision_score(tensor_labels_test.cpu().numpy(), test_probs)
        performance = avg_prec / (sum(tensor_labels_test.cpu().numpy()) / len(tensor_labels_test.cpu().numpy())) # avg_prec / random_baseline
        auc_score = roc_auc_score(tensor_labels_test.cpu().numpy(), test_probs)

    print(f"Test Performance: {performance:.4f}")

    # --- SAVE RESULTS ---

    # Prepare directories
    def ensure_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
    results_base = f"../results/gnn/{dataset}"
    model_dir = f"{results_base}/best_model"
    params_dir = f"{results_base}/best_params"
    metrics_dir = f"{results_base}/metrics"

    for d in [model_dir, params_dir, metrics_dir]:
        ensure_dir(d)

    # Save metrics
    with open(f"{metrics_dir}/test_{model_name}.csv", "w") as f:
        f.write("performance,avg_prec,auc\n")
        f.write(f"{performance},{avg_prec},{auc_score}\n")

    # Save best params
    with open(f"{params_dir}/best_params_{model_name}.txt", "w") as f:
        f.write(str(best_params))

    # Save best model
    with open(f"{model_dir}/best_model_{model_name}.pkl", "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)



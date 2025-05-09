from data import get_bace_gnn_datasets, BACE_GNN
from models import MLPClassifier
from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn.models import NeuralFingerprint

dataset = BACE_GNN(root='data/BACE_GNN')

train, test = get_bace_gnn_datasets(root='data/BACE_GNN')

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

ngf = NeuralFingerprint(
    in_channels     = dataset.num_features,  
    hidden_channels = 128,
    out_channels    = 128,
    num_layers      = 3
)
    
clf = MLPClassifier(input_dim=128, hidden_dim=128)

for p in ngf.parameters():
    p.requires_grad = True
    
loader = DataLoader(dataset, batch_size=64, shuffle=False)

all_embs = []
with torch.no_grad():
    for data in loader:
        emb = ngf(data.x.float(), data.edge_index, data.batch)  
        all_embs.append(emb.cpu())
all_embs = torch.cat(all_embs, dim=0) 
unique_embs = torch.unique(all_embs, dim=0)
print(f"Total graphs:             {all_embs.size(0)}")
print(f"Unique GNN embeddings:    {unique_embs.size(0)}")


optimizer = optim.Adam(list(clf.parameters())+list(ngf.parameters()), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(1, 201):
    clf.train()
    ngf.train()  
    total_loss = 0.0

    for data in train_loader:
        optimizer.zero_grad()
        emb = ngf(data.x.float(), data.edge_index, data.batch)

        logits = clf(emb)             
        labels = data.y.float() 

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        preds = (torch.sigmoid(logits) > 0.5).float()

    train_loss = total_loss / len(train)
    print(f"Epoch {epoch:>2}  Loss: {train_loss:.4f}")


ngf.eval(); clf.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for data in test_loader:
        emb    = ngf(data.x.float(), data.edge_index, data.batch)
        logits = clf(emb).view(-1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        labels = data.y.view(-1).cpu().numpy().astype(int)
        all_probs.extend(probs)
        all_labels.extend(labels)

roc_auc = roc_auc_score(all_labels, all_probs)
print(f"3 layer-NGF → MLPClassification Test ROC-AUC: {roc_auc:.4f}")


from data import get_lipo_gnn_datasets, LIPO_GNN
from models import MLPRegressor, MultiLayerGIN
from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import numpy as np


dataset = LIPO_GNN(root='data/LIPO_GNN')

train, test = get_lipo_gnn_datasets(root='data/LIPO_GNN')

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

gin = MultiLayerGIN(in_channels=dataset.num_features, hidden_channels=64, num_layers=3)
for p in gin.parameters(): 
    p.requires_grad = True

loader = DataLoader(dataset, batch_size=64, shuffle=False)

all_embs = []
with torch.no_grad():
    for data in loader:
        emb = gin(data)
        all_embs.append(emb.cpu())
all_embs = torch.cat(all_embs, dim=0) 
unique_embs = torch.unique(all_embs, dim=0)
print(f"Total graphs:             {all_embs.size(0)}")
print(f"Unique GNN embeddings:    {unique_embs.size(0)}")

reg = MLPRegressor(input_dim=64, hidden_dim=128)

optimizer = optim.Adam(list(gin.parameters())+ list(reg.parameters()), lr=0.001)
criterion = nn.MSELoss()
    
# 6) Training loop
for epoch in range(1, 201):
    reg.train()
    gin.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        #with torch.no_grad():
        emb = gin(data)             # shape [batch_size]
        logits = reg(emb)
        targets = data.y.view(-1)       # shape [batch_size]
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    total_loss /= len(train)
    print(f"Epoch {epoch:>2}  Train RMSE: {total_loss**0.5:.4f}")

# 7) Evaluation on test set
reg.eval()
gin.eval()
sq_err = 0.0
count  = 0
with torch.no_grad():
    for data in test_loader:
        emb   = gin(data)
        logits = reg(emb)
        targets = data.y.view(-1)
        sq_err += ((logits - targets)**2).sum().item()
        count  += data.num_graphs

rmse = (sq_err / count)**0.5
print(f"3 layer GIN → MLPRegressor RMSE = {rmse:.4f}")
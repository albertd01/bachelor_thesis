from data import get_lipo_gnn_datasets, LIPO_GNN
from models import MLPRegressor
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn.models import NeuralFingerprint
import numpy as np

dataset = LIPO_GNN(root='data/LIPO_GNN')

train, test = get_lipo_gnn_datasets(root='data/LIPO_GNN')

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

ngf = NeuralFingerprint(
    in_channels     = dataset.num_features,  
    hidden_channels = 128,
    out_channels    = 128,
    num_layers      = 3
)
    
reg = MLPRegressor(input_dim=128, hidden_dim=128)

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



optimizer = optim.Adam(list(ngf.parameters()) + list(reg.parameters()), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(1, 201):
    ngf.train(); reg.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        emb = ngf(data.x.float(), data.edge_index, data.batch)
        pred  = reg(emb).view(-1)
        target= data.y.view(-1)
        loss  = criterion(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    print(f"Epoch {epoch:>2}  Train RMSE: {(total_loss / len(train))**0.5:.4f}")

ngf.eval(); reg.eval()
sq_err, cnt = 0.0, 0
with torch.no_grad():
    for data in test_loader:
        emb    = ngf(data.x.float(), data.edge_index, data.batch)
        pred   = reg(emb).view(-1)
        target = data.y.view(-1)
        sq_err += ((pred - target)**2).sum().item()
        cnt    += data.num_graphs

rmse = (sq_err/cnt)**0.5
print(f"3 layer-NGF → MLPRegression Test RMSE: {rmse:.4f}")
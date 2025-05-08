from data import get_lipo_ecfp4_datasets, LIPO_ECFP, FingerprintDataset
from models import MLPRegressor
from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Analysis of expressivity
dataset    = LIPO_ECFP(root='data/LIPO_ECFP', radius=2, nBits=2048)
flat = dataset._data.x             
n    = len(dataset)                 
d    = flat.size(0) // n            

all_fps = flat.view(n, d)           
unique_fps = torch.unique(all_fps, dim=0)

print(f"Total molecules:           {all_fps.size(0)}")
print(f"Unique ECFP4 fingerprints: {unique_fps.size(0)}")

# Training the model
train_inmem, test_inmem = get_lipo_ecfp4_datasets(root='data/LIPO_ECFP')

train = FingerprintDataset(train_inmem)
test  = FingerprintDataset(test_inmem)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test,  batch_size=32)

model     = MLPRegressor(input_dim=2048, hidden_dim=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(1, 16):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        preds = model(x)             
        loss = criterion(preds, y.view(-1))   
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    total_loss /= len(train)
    print(f"Epoch {epoch:>2}  Train MSE: {total_loss:.4f}")

model.eval()
total_sq_err = 0.0
total_count  = 0
with torch.no_grad():
    for x, y in test_loader:
        preds   = model(x)  
        targets = y.view(-1)          

        total_sq_err += ((preds - targets) ** 2).sum().item()
        total_count  += targets.size(0)

rmse = np.sqrt(total_sq_err / total_count)
print(f"Test RMSE: {rmse:.4f}")
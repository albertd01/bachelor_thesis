from data import get_bace_ecfp4_datasets, BACE_ECFP
from models import MLPClassifier
from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

'''# Analysis of expressivity
dataset = BACE_ECFP(root='data/BACE_ECFP', radius=2, nBits=2048)
all_fps = dataset.data.x 
unique_fps = torch.unique(all_fps, dim=0)
print(f"Total molecules:         {all_fps.size(0)}")
print(f"Unique ECFP4 fingerprints: {unique_fps.size(0)}")'''

class FingerprintDataset(Dataset):
    """
    Wraps any InMemoryDataset of Data objects with .x = feature tensor
    and .y = label tensor into a simple (features, label) tuple dataset.
    """
    def __init__(self, inmem_subset):
        self.subset = inmem_subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        data = self.subset[idx]
        # data.x is already your precomputed fingerprint [2048]
        # data.y is [1], so we squeeze it to a scalar
        return data.x, data.y.view(-1)

# Training the model
train_inmem, test_inmem = get_bace_ecfp4_datasets(root='data/BACE_ECFP')

# 2) wrap them in our adapter
train = FingerprintDataset(train_inmem)
test  = FingerprintDataset(test_inmem)

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test,  batch_size=32)

model = MLPClassifier(input_dim=2048, hidden_dim=128, output_dim=1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

model.train()
for epoch in range(15):
    epoch_loss = 0.0
    for batch_feature, batch_label in train_loader:
        fingerprints = batch_feature
        labels       = batch_label.view(-1)     
        
        optimizer.zero_grad()
        outputs = model(fingerprints).view(-1)  
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/15  Loss: {epoch_loss/len(train_loader):.4f}")

model.eval()
all_labels = []
all_probs  = []
correct = 0
total   = 0

with torch.no_grad():
    for batch_feature, batch_label in test_loader:
        fingerprints = batch_feature
        labels       = batch_label.view(-1)          # [B]
        logits       = model(fingerprints).view(-1)
        probs        = torch.sigmoid(logits)      # [B]

        # accumulate for AUC
        all_probs .extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # compute accuracy too
        preds   = (probs > 0.5).float()
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

accuracy = correct / total
roc_auc   = roc_auc_score(np.array(all_labels), np.array(all_probs))

print(f"Test accuracy: {accuracy:.4f}")
print(f"Test ROC-AUC : {roc_auc:.4f}")





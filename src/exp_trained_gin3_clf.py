from data import get_bace_gnn_datasets, BACE_GNN
from models import MLPClassifier, MultiLayerGIN
from torch_geometric.loader import DataLoader
import torch
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.optim as optim
import numpy as np

dataset = BACE_GNN(root='data/BACE_GNN')

train, test = get_bace_gnn_datasets(root='data/BACE_GNN')

train_loader = DataLoader(train, batch_size=32, shuffle=True)
test_loader = DataLoader(test, batch_size=32, shuffle=False)

gin = MultiLayerGIN(in_channels=dataset.num_features, hidden_channels=128, num_layers=3)
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


clf = MLPClassifier(input_dim=128, hidden_dim=128, output_dim=1)

optimizer = optim.Adam(list(gin.parameters()) + list(clf.parameters()), lr=0.001) 
criterion = nn.BCEWithLogitsLoss()

num_epochs = 200
clf.train()
gin.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for data in train_loader:
        optimizer.zero_grad()
        emb = gin(data)  
        logits = clf(emb)
        labels = data.y.float()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")


clf.eval()
gin.eval()
all_labels = []
all_probs = []
with torch.no_grad():
    for data in test_loader:
        emb = gin(data)
        logits = clf(emb)
        predictions = (torch.sigmoid(logits) > 0.5).float()
        labels = data.y.float()
        all_probs.extend(predictions.numpy().flatten())
        all_labels.extend(labels.numpy().flatten())



roc_auc = roc_auc_score(np.array(all_labels), np.array(all_probs))
print(f"3 layer GIN → MLPClassifier ROC-AUC = {roc_auc:.4f}")

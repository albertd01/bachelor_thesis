import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import NeuralFingerprint
from data import BACE_GNN
from models import MultiLayerGIN

# 1) Load dataset & model (as before)
dataset = BACE_GNN(root='data/BACE_GNN')
loader  = DataLoader(dataset, batch_size=64, shuffle=False)

ngf = NeuralFingerprint(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=128,
    num_layers=3
)
for p in ngf.parameters(): p.requires_grad = False
ngf.eval()

# 2) Collect all embeddings
all_emb = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to('cpu')
        emb = ngf(batch.x.float(), batch.edge_index, batch.batch)
        all_emb.append(emb)
all_emb = torch.cat(all_emb, dim=0)  # [N, 128]

# 3) Sample 500 embeddings
N = all_emb.size(0)
idx = torch.randperm(N)[:500]
sampled = all_emb[idx]  # [500, 128]

# 4) Compute pairwise distances streaming
dists = []
for i in range(500):
    # compute distances from sampled[i] to all sampled[i+1:]
    vec = sampled[i].unsqueeze(0)       # [1, 128]
    rest = sampled[i+1:]                # [500-i-1, 128]
    # Euclidean distances
    dist = torch.norm(vec - rest, dim=1)  # [500-i-1]
    dists.append(dist)
dists = torch.cat(dists).cpu().numpy()  # [500*499/2]

# 5) Plot histogram
plt.figure()
plt.hist(dists, bins=50)
plt.title("Pairwise Euclidean Distance (n=500) of Frozen NGF Embeddings")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()

gin = MultiLayerGIN(in_channels=dataset.num_features, hidden_channels=64, num_layers=3)
for p in gin.parameters(): p.requires_grad = False
gin.eval()

# 2) Collect all embeddings
all_emb = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to('cpu')
        emb = gin(batch)
        all_emb.append(emb)
all_emb = torch.cat(all_emb, dim=0)  # [N, 128]

# 3) Sample 500 embeddings
N = all_emb.size(0)
idx = torch.randperm(N)[:500]
sampled = all_emb[idx]  # [500, 128]

# 4) Compute pairwise distances streaming
dists = []
for i in range(500):
    # compute distances from sampled[i] to all sampled[i+1:]
    vec = sampled[i].unsqueeze(0)       # [1, 128]
    rest = sampled[i+1:]                # [500-i-1, 128]
    # Euclidean distances
    dist = torch.norm(vec - rest, dim=1)  # [500-i-1]
    dists.append(dist)
dists = torch.cat(dists).cpu().numpy()  # [500*499/2]

plt.figure()
plt.hist(dists, bins=50)
plt.title("Pairwise Euclidean Distance (n=500) of Frozen GIN Embeddings")
plt.xlabel("Distance")
plt.ylabel("Frequency")
plt.show()
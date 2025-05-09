import torch
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import NeuralFingerprint
from data import BACE_GNN

# 1) Load the cached one-hot GNN dataset for BACE
dataset = BACE_GNN(root='data/BACE_GNN')
loader = DataLoader(dataset, batch_size=64, shuffle=False)

# 2) Instantiate and freeze the NGF model (same config as your experiments)
ngf = NeuralFingerprint(
    in_channels=dataset.num_node_features,
    hidden_channels=64,
    out_channels=128,
    num_layers=3
)
for param in ngf.parameters():
    param.requires_grad = False
ngf.eval()

# 3) Collect graph-level embeddings
embeddings = []
with torch.no_grad():
    for batch in loader:
        batch = batch.to('cpu')
        emb = ngf(batch.x.float(), batch.edge_index, batch.batch)
        embeddings.append(emb)
all_emb = torch.cat(embeddings, dim=0)  # [N_graphs, 128]

# 4a) Plot distribution of all fingerprint values
vals = all_emb.flatten().cpu().numpy()
plt.figure()
plt.hist(vals, bins=50)
plt.title("Frozen NGF Fingerprint Value Distribution")
plt.xlabel("Fingerprint Component Value")
plt.ylabel("Frequency")
plt.show()

# 4b) Plot distribution of L2 norms of each fingerprint
plt.figure()
norms = torch.norm(all_emb, dim=1).numpy()
plt.hist(norms, bins=50)
plt.title("L2 Norm Distribution of Frozen NGF Embeddings")
plt.xlabel("L2 Norm")
plt.ylabel("Frequency")
plt.show()


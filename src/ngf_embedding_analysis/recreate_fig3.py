from ngf_model import NeuralGraphFingerprint
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from itertools import combinations
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import hashlib
from rdkit.Chem import rdFingerprintGenerator
from alg1_ecfp import compute_ecfp_fp, algorithm1_duvenaud
from features import mol_to_duvenaud_graph
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn.models import NeuralFingerprint

class DuvenaudESOLDataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__('.', transform=None)
        self.data, self.slices = self.collate(data_list)

    @property
    def num_features(self):
        return self[0].x.shape[1]

dataset = MoleculeNet(root='data/MoleculeNet', name='ESOL')

# Step 2: Transform it with your feature function
duvenaud_graphs = []

for i in range(len(dataset)):
    smiles = dataset[i].smiles
    mol = Chem.MolFromSmiles(smiles)
    graph = mol_to_duvenaud_graph(mol)

    # Copy the label (target value)
    graph.y = dataset[i].y
    graph.smiles = smiles  # Optional: preserve SMILES for traceability
    duvenaud_graphs.append(graph)

dataset = DuvenaudESOLDataset(duvenaud_graphs)
# Step 3: Create a new DataLoader from transformed dataset
loader = DataLoader(dataset, batch_size=64, shuffle=False)
    

# =======================
# 2. Define NGF model
# =======================
model = NeuralGraphFingerprint(
    in_channels=dataset.num_features,
    hidden_dim=128,
    fingerprint_dim=2048,
    num_layers=2,
    weight_scale= 1.0
)

model = NeuralFingerprint(
    in_channels=dataset.num_features,
    hidden_channels=128,
    out_channels=2048,
    num_layers=2,
)

for p in model.parameters():
    p.requires_grad = False

model.eval()
all_embs = []
with torch.no_grad():
    for data in loader:
        #emb=model(data)
        emb = model(data.x.float(), data.edge_index, data.batch)  
        all_embs.append(emb.cpu())
emb_mat = torch.cat(all_embs, dim=0).numpy() 


def smiles_to_ecfp(smiles, radius=2, nBits=2048):
    return compute_ecfp_fp(smiles=smiles, radius = radius, nBits=nBits)

'''def smiles_to_ecfp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)'''

smiles_list = [dataset[i].smiles for i in range(len(dataset))]
fps_ecfp = [smiles_to_ecfp(smi) for smi in smiles_list]

# Build pairs
rng = np.random.default_rng(42)
all_pairs = list(combinations(range(len(fps_ecfp)), 2))
sample_idxs = rng.choice(len(all_pairs), size=1000, replace=False)
pairs = [all_pairs[i] for i in sample_idxs]


def cont_tanimoto_minmax(x, y):
    num = np.sum(np.minimum(x, y))
    denom = np.sum(np.maximum(x, y)) + 1e-8 
    return 1.0 - (num / denom)

ecfp_dists = np.array([
    cont_tanimoto_minmax(fps_ecfp[i], fps_ecfp[j])
    for i, j in pairs
])

ngf_dists = np.array([
    cont_tanimoto_minmax(emb_mat[i], emb_mat[j])
    for i, j in pairs
])

# Compute r
r, _ = pearsonr(ecfp_dists, ngf_dists)


plt.figure(figsize=(5, 5))
plt.scatter(
    ecfp_dists,
    ngf_dists,
    s=20,
    alpha=0.4,
    edgecolors='black',
    linewidths=0.2,
    facecolor='C0'
)
plt.xlabel("Circular fingerprint distances", fontsize=12)
plt.ylabel("Neural fingerprint distances", fontsize=12)
plt.xlim(0.5, 1.0)
plt.ylim(0.5, 1.0)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.title(f"Neural vs Circular distances, $r={r:.3f}$", fontsize=14)
plt.tight_layout()
plt.show()

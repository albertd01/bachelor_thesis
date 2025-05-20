from torch_geometric.datasets import MoleculeNet
from alg1_ecfp import algorithm1_duvenaud, compute_ecfp_array
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def compare_fingerprints_on_dataset(smiles_list, radius=2, nBits=2048):
    tanimoto_scores = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        f1 = algorithm1_duvenaud(mol, radius=radius, nBits=nBits)
        f2 = compute_ecfp_array(smiles, radius=radius, nBits=nBits)
        intersection = np.sum(np.logical_and(f1, f2))
        union = np.sum(np.logical_or(f1, f2))
        tanimoto = intersection / union if union > 0 else 0.0
        tanimoto_scores.append(tanimoto)
    return tanimoto_scores

# This automatically downloads and processes the ESOL dataset
dataset = MoleculeNet(root="data/MoleculeNet", name="ESOL")

# Each element is a PyG Data object
print(f"Number of molecules: {len(dataset)}")
print(dataset[0])

data = dataset[0]
smiles = dataset.smiles
print("Atom features shape:", data.x.shape)
print("Edge index shape:", data.edge_index.shape)
print("Edge attributes shape:", data.edge_attr.shape)
print("Target value (log solubility):", data.y)

# Run analysis
tanimoto_scores = compare_fingerprints_on_dataset(smiles)

# Plot results
sns.histplot(tanimoto_scores, bins=30, kde=True)
plt.title("Tanimoto Similarity: Duvenaud Algorithm 1 vs RDKit ECFP")
plt.xlabel("Tanimoto Similarity")
plt.ylabel("Number of Molecules")
plt.grid(True)
plt.show()
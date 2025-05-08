# data.py

import torch
import numpy as np
from torch.utils.data import random_split
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import MoleculeNet
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
import torch.nn.functional as F
from torch.utils.data import Dataset

# -------------------------------------------------------------------
# 1) Helper to compute ECFP bit arrays
# -------------------------------------------------------------------

def get_custom_invariants(mol):
    """Compute a list of integer invariants, one per atom in the molecule,
    using a tuple of properties. Adjust the tuple to include the properties you need."""
    invariants = []
    for atom in mol.GetAtoms():
        # Create a tuple of atomic properties:
        inv_tuple = (
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            str(atom.GetHybridization()),
            atom.GetIsAromatic(),
            atom.IsInRing(), 
        )
        # Hash the tuple into an integer.
        # The bitmask (& 0xFFFFFFFF) ensures the result is within a 32-bit range.
        invariants.append(hash(inv_tuple) & 0xFFFFFFFF)
    return invariants

def compute_ecfp_array(smiles: str, radius: int, nBits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=nBits, includeChirality=True)
    fp = mfpgen.GetFingerprint(mol, customAtomInvariants=get_custom_invariants(mol))
    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# -------------------------------------------------------------------
# 2) Generic BACE‐ECFP InMemoryDataset
# -------------------------------------------------------------------
class BACE_ECFP(InMemoryDataset):
    def __init__(self, root: str, radius: int, nBits: int = 2048):
        self.radius = radius
        self.nBits   = nBits
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []   # pull raw data via MoleculeNet

    @property
    def processed_file_names(self):
        # distinct file for each radius
        return [f'ecfp_r{self.radius}_b{self.nBits}.pt']

    def download(self):
        pass

    def process(self):
        molnet = MoleculeNet(self.root, name='bace')
        data_list = []
        for data in molnet:
            arr = compute_ecfp_array(data.smiles, self.radius, self.nBits)
            x   = torch.from_numpy(arr).float()           # [nBits]
            y   = data.y.clone().view(-1).float()         # [1]
            data_list.append(Data(x=x, y=y))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
class LIPO_ECFP(InMemoryDataset):
    def __init__(self, root: str, radius: int, nBits: int = 2048):
        self.radius = radius
        self.nBits   = nBits
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []   # pull raw data via MoleculeNet

    @property
    def processed_file_names(self):
        # distinct file for each radius
        return [f'ecfp_r{self.radius}_b{self.nBits}.pt']

    def download(self):
        pass

    def process(self):
        molnet = MoleculeNet(self.root, name='lipo')
        data_list = []
        for data in molnet:
            arr = compute_ecfp_array(data.smiles, self.radius, self.nBits)
            x   = torch.from_numpy(arr).float()           # [nBits]
            y   = data.y.clone().view(-1).float()         # [1]
            data_list.append(Data(x=x, y=y))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# -------------------------------------------------------------------
# 3) Helper to split BACE_ECFP into train/test
# -------------------------------------------------------------------
def _split_ecfp_bace(root: str, radius: int, nBits: int,
                     train_ratio: float, seed: int):
    full = BACE_ECFP(root, radius=radius, nBits=nBits)
    n_train = int(train_ratio * len(full))
    n_test  = len(full) - n_train
    g = torch.Generator().manual_seed(seed)
    return random_split(full, [n_train, n_test], generator=g)

def _split_ecfp_lipo(root: str, radius: int, nBits: int,
                     train_ratio: float, seed: int):
    full = LIPO_ECFP(root, radius=radius, nBits=nBits)
    n_train = int(train_ratio * len(full))
    n_test  = len(full) - n_train
    g = torch.Generator().manual_seed(seed)
    return random_split(full, [n_train, n_test], generator=g)

def get_lipo_ecfp4_datasets(root: str,
                            nBits: float = 2048,
                            train_ratio: float = 0.8,
                            seed: int = 42):
    return _split_ecfp_lipo(root, radius=2, nBits=nBits,
                            train_ratio=train_ratio, seed=seed)

def get_lipo_ecfp4_datasets(root: str,
                            nBits: float = 2048,
                            train_ratio: float = 0.8,
                            seed: int = 42):
    return _split_ecfp_lipo(root, radius=3, nBits=nBits,
                            train_ratio=train_ratio, seed=seed)

def get_bace_ecfp4_datasets(root: str,
                            nBits: float = 2048,
                            train_ratio: float = 0.8,
                            seed: int = 42):
    """
    Returns train/test Subsets for ECFP4 (radius=2).
    """
    return _split_ecfp_bace(root, radius=2, nBits=nBits,
                            train_ratio=train_ratio, seed=seed)

def get_bace_ecfp6_datasets(root: str,
                            nBits: float = 2048,
                            train_ratio: float = 0.8,
                            seed: int = 42):
    """
    Returns train/test Subsets for ECFP6 (radius=3).
    """
    return _split_ecfp_bace(root, radius=3, nBits=nBits,
                            train_ratio=train_ratio, seed=seed)
    
class OneHotEncodeFeatures(object):
    def __init__(self, feature_indices):
        """
        Args:
            feature_indices (dict):
              keys = column index in data.x,
              vals = (max_value + 1) for that column.
        """
        self.feature_indices = feature_indices

    def __call__(self, data):
        new_feats = []
        num_feats = data.x.size(1)
        for j in range(num_feats):
            if j in self.feature_indices:
                num_cats = self.feature_indices[j]
                col = data.x[:, j].long()
                oh  = F.one_hot(col, num_classes=num_cats).to(torch.float)
                new_feats.append(oh)
            else:
                new_feats.append(data.x[:, j].view(-1,1))
        data.x = torch.cat(new_feats, dim=1)
        return data
    
class LIPO_GNN(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['gnn_onehot.pt']

    def download(self):
        pass

    def process(self):
        raw = MoleculeNet(self.root, name='lipo')
        feature_indices = {0: 54, 1:3, 2:5, 3:7, 4:4, 6:5} 
        onehot = OneHotEncodeFeatures(feature_indices)
        data_list = []
        for data in raw:
            d2 = onehot(data)
            data_list.append(d2)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
class BACE_GNN(InMemoryDataset):
    def __init__(self, root):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['gnn_onehot.pt']

    def download(self):
        pass

    def process(self):
        # 1) load raw PyG BACE graphs
        raw = MoleculeNet(self.root, name='bace')
        # 2) compute global max per feature column
        feature_indices = {0: 54, 1:3, 2:5, 3:7, 4:4, 6:5} 
        onehot = OneHotEncodeFeatures(feature_indices)
        # 4) apply and collect
        data_list = []
        for data in raw:
            d2 = onehot(data)
            data_list.append(d2)
        # 5) collate & save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_bace_gnn_datasets(root, train_ratio=0.8, seed=42):
    full = BACE_GNN(root)
    n_train = int(train_ratio * len(full))
    n_test  = len(full) - n_train
    g = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(full, [n_train, n_test], generator=g)
    return train_ds, test_ds

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
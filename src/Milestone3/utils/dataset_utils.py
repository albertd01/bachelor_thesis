from torch_geometric.datasets import MoleculeNet
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
import torch
from utils.feature_extraction import mol_to_duvenaud_graph

class DuvenaudDataset(InMemoryDataset):
    def __init__(self, name, transform=None, pre_transform=None):
        self.name = name.lower()
        self.supported = ['esol', 'bace', 'lipo']
        if self.name not in self.supported:
            raise ValueError(f"Dataset '{self.name}' not supported. Choose from {self.supported}")
        
        self.smiles_list = []
        self.labels = []

        super().__init__(root=f"data/{self.name}", transform=transform, pre_transform=pre_transform)

        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f'{self.name}_duvenaud.pt']

    def process(self):
        dataset = MoleculeNet(root=self.root, name=self.name.upper() if self.name == 'esol' else self.name.capitalize())
        data_list = []

        for data in dataset:
            smiles = data.smiles
            mol = Chem.MolFromSmiles(smiles)
            graph = mol_to_duvenaud_graph(mol)
            graph.y = data.y
            graph.smiles = smiles

            self.smiles_list.append(smiles)
            self.labels.append(data.y.item())
            data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

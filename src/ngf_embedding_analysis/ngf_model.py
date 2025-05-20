from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn.models import NeuralFingerprint
import numpy as np
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F

class NeuralGraphFingerprint(nn.Module):
    def __init__(self, in_channels, hidden_dim, fingerprint_dim, num_layers=3, weight_scale=5.0):
        super().__init__()
        self.num_layers = num_layers
        self.weight_scale = weight_scale

        self.W_self = nn.ModuleList()
        self.W_neigh = nn.ModuleList()
        self.W_fp = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = in_channels if layer == 0 else hidden_dim
            self.W_self.append(nn.Linear(in_dim, hidden_dim, bias=True))
            self.W_neigh.append(nn.Linear(in_dim, hidden_dim, bias=True))
            self.W_fp.append(nn.Linear(hidden_dim, fingerprint_dim, bias=False))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in range(self.num_layers):
            nn.init.normal_(self.W_self[layer].weight, mean=0.0, std=self.weight_scale)
            nn.init.normal_(self.W_neigh[layer].weight, mean=0.0, std=self.weight_scale)
            nn.init.zeros_(self.W_self[layer].bias)
            nn.init.zeros_(self.W_neigh[layer].bias)
            nn.init.normal_(self.W_fp[layer].weight, mean=0.0, std=self.weight_scale)

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        fingerprint = x.new_zeros(batch.max().item() + 1, self.W_fp[0].out_features)

        for layer in range(self.num_layers):
            row, col = edge_index
            neigh_sum = torch.zeros_like(x)
            neigh_sum = neigh_sum.index_add(0, row, x[col])
            x = self.W_self[layer](x) + self.W_neigh[layer](neigh_sum)
            x = F.tanh(x)

            node_contrib = torch.softmax(self.W_fp[layer](x), dim=1)
            #node_contrib = F.gumbel_softmax(self.W_fp[layer](x),tau=0.01, hard=False,  dim=1)
            fingerprint_layer = global_add_pool(node_contrib, batch)
            fingerprint = fingerprint + fingerprint_layer

        return fingerprint

    
from torch_geometric.data import Data, Batch

# Build toy graph: 2 nodes, 1 edge
x = torch.tensor([[1.0],[2.0]])  # 1-dim features for simplicity
edge_index = torch.tensor([[0,1],[1,0]])
batch = torch.zeros(2, dtype=torch.long)
data = Data(x=x, edge_index=edge_index, batch=batch)

# Instantiate your NGF with known weights
model = NeuralGraphFingerprint(
    in_channels=1,
    hidden_dim=2,
    fingerprint_dim=2,
    num_layers=1
)
'''model.W_self[0].weight.data = torch.tensor([[1.0], [0.0]])  # shape [2, 1]
model.W_self[0].bias.data = torch.tensor([0.0, 0.0])
model.W_neigh[0].weight.data = torch.tensor([[0.0], [1.0]])  # shape [2, 1]
model.W_neigh[0].bias.data = torch.tensor([0.0, 0.0])
model.W_fp[0].weight.data = torch.tensor([
    [1.0, -1.0],   # First fingerprint dimension
    [0.5, 0.5]     # Second fingerprint dimension
])  # shape [2, 2]'''
# (Optionally: manually set ngf.W_self, W_neigh, W_fp to known small tensors)

my_fp = model(data)

print("My toy fingerprint:", my_fp)

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool


class MultiLayerGIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            dim_in = in_channels if layer == 0 else hidden_channels
            # each GIN block: MLP(in→hidden→hidden)
            mlp = nn.Sequential(
                nn.Linear(dim_in, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return global_add_pool(x, batch)
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=128, output_dim=1):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)
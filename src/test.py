from data import BACE_GNN

# 1) Load the cached one‐hot‐encoded GNN dataset
dataset = BACE_GNN(root='data/BACE_GNN')

# 2a) Using the Data object’s helper:
print("Node feature size (num_node_features):", dataset[0].num_node_features)

# 2b) Or by inspecting the x tensor directly:
print("x.shape for graph 0:", dataset[0].x.shape)
print("→ number of features per node:", dataset[0].x.shape[1])
from data import get_bace_ecfp4_datasets, get_bace_ecfp6_datasets, get_bace_gnn_datasets
from torch_geometric.loader import DataLoader

# ECFP4
train4, test4 = get_bace_ecfp4_datasets(root='data/BACE_ECFP')
loader4_tr = DataLoader(train4, batch_size=32, shuffle=True)
loader4_te = DataLoader(test4,  batch_size=32)

#GNN
gnn_train, gnn_test = get_bace_gnn_datasets( 'data/BACE_GNN')
loader_gnn_tr = DataLoader(gnn_train, batch_size=32, shuffle=True)
loader_gnn_te = DataLoader(gnn_test, batch_size=32, shuffle=True)

# ECFP6
train6, test6 = get_bace_ecfp6_datasets(root='data/BACE_ECFP')
loader6_tr = DataLoader(train6, batch_size=32, shuffle=True)
loader6_te = DataLoader(test6,  batch_size=32)


print(len(train4) + len(test4))
print(len(train6) + len(test6))
print(len(gnn_train) + len(gnn_test))
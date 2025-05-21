import yaml
import torch
from pathlib import Path
from utils.dataset_utils import DuvenaudDataset
from utils.ecfp_utils import compute_ecfp_bit_vectors
from models.ngf import NeuralGraphFingerprint
from utils.evaluation import run_pairwise_analysis, plot_pairwise_distances
from utils.downstream import run_downstream_task
import numpy as np
from torch_geometric.loader import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_experiment(config):
    # Load dataset + custom PyG graphs
    dataset = DuvenaudDataset(
        name=config['experiment']['dataset']
    )
    dataset.process()

    # Compute ECFP fingerprints
    fps_ecfp = compute_ecfp_bit_vectors(
        dataset.smiles_list,
        radius=config['experiment']['ECFP']['radius'],
        nBits=config['experiment']['ECFP']['fingerprint_dim'],
        count_fp=config['experiment']['ECFP']['count_fingerprint']
    )

    # Construct NGF model
    example_input = dataset[0]
    in_channels = example_input.x.shape[1]
    ngf = NeuralGraphFingerprint(
        in_channels=in_channels,
        hidden_dim=config['experiment']['ngf']['hidden_dim'],
        fingerprint_dim=config['experiment']['ngf']['fingerprint_dim'],
        num_layers=config['experiment']['ngf']['num_layers'],
        weight_scale=config['experiment']['ngf']['weight_scale'],
        sum_fn=config['experiment']['ngf']['sum_fn'],
        smooth_fn=config['experiment']['ngf']['smooth_fn'],
        sparsify_fn=config['experiment']['ngf']['sparsify_fn']
    )
    if config['experiment']['ngf']['frozen']:
        for p in ngf.parameters():
            p.requires_grad = False

    # Generate NGF embeddings
    ngf.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    emb_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to('cpu')
            emb = ngf(batch)
            emb_list.append(emb.cpu())
    emb_mat = torch.cat(emb_list, dim=0).numpy()
    print("Number of ECFP fingerprints:", len(fps_ecfp))
    print("NGF embeddings shape:", emb_mat.shape)
    
    # Run distance analysis
    ecfp_dists, ngf_dists, r = run_pairwise_analysis(
        emb_mat,
        fps_ecfp,
        num_pairs=config['experiment']['evaluation']['num_pairs']
    )

    # Plot
    plot_pairwise_distances(ecfp_dists, ngf_dists, r,
                            title=f"{config['experiment']['dataset']} ($r={r:.3f}$)")
    # Downstream Evaluation
    labels_np = np.array([data.y.item() for data in dataset])

    task_type = config['experiment']['evaluation']['downstream_task']
    dataset_name = config['experiment']['dataset']

    results = run_downstream_task(
        ecfp_array=fps_ecfp,
        ngf_array=emb_mat,
        labels=np.array(labels_np),
        task_type=task_type
    )

    print(f"\n[Downstream Evaluation] {task_type} on {dataset_name} dataset")
    print(f"ECFP   → mean: {results['ecfp'][0]:.4f}, std: {results['ecfp'][1]:.4f}")
    print(f"NGF    → mean: {results['ngf'][0]:.4f}, std: {results['ngf'][1]:.4f}")

if __name__ == "__main__":
    config_path = Path("config/experiment.yaml")
    config = load_config(config_path)
    run_experiment(config)

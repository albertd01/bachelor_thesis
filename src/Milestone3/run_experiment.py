import yaml
import torch
from pathlib import Path
from utils.dataset_utils import DuvenaudDataset
from utils.ecfp_utils import compute_ecfp_bit_vectors, compute_algorithm1_fps, compute_ecfp_count_vectors
from models.ngf import NeuralGraphFingerprint, NGFWithHead
from utils.evaluation import run_pairwise_analysis, plot_pairwise_distances
from utils.frozen_downstream import run_frozen_downstream_task
import numpy as np
from torch_geometric.loader import DataLoader
from utils.logging_utils import create_experiment_dir, save_results, save_distances_csv, save_distance_plot
from torch_geometric.nn.models import NeuralFingerprint
from models.ngf_adapter import NGFAdapter
from utils.end_to_end_downstream import run_end_to_end_training

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
    ecfp_impl = config['experiment']['ecfp']['implementation']
    radius = config['experiment']['ecfp']['radius']
    nBits = config['experiment']['ecfp']['fingerprint_dim']

    if ecfp_impl == 'rdkit_binary':
        fps_ecfp = compute_ecfp_bit_vectors(dataset.smiles_list, radius=radius, nBits=nBits)

    elif ecfp_impl == 'rdkit_count':
        fps_ecfp = compute_ecfp_count_vectors(dataset.smiles_list, radius=radius, nBits=nBits)

    elif ecfp_impl == 'algorithm1':
        fps_ecfp = compute_algorithm1_fps(dataset.smiles_list, radius=radius, nBits=nBits)

    else:
        raise ValueError("Invalid ECFP implementation.")

    # Construct NGF model
    example_input = dataset[0]
    in_channels = example_input.x.shape[1]
    if config['experiment']['ngf']['implementation'] == 'from_scratch':

        model_core = NeuralGraphFingerprint(
            in_channels=in_channels,
            hidden_dim=config['experiment']['ngf']['hidden_dim'],
            fingerprint_dim=config['experiment']['ngf']['fingerprint_dim'],
            num_layers=config['experiment']['ngf']['num_layers'],
            sum_fn=config['experiment']['ngf']['sum_fn'],
            smooth_fn=config['experiment']['ngf']['smooth_fn'],
            sparsify_fn=config['experiment']['ngf']['sparsify_fn'],
        )
        ngf = NGFAdapter(model_core,mode="custom")
        
    elif config['experiment']['ngf']['implementation'] == 'pytorch_geometric':

        model_core = NeuralFingerprint(
            in_channels=in_channels,
            hidden_channels=config['experiment']['ngf']['hidden_dim'],
            out_channels=config['experiment']['ngf']['fingerprint_dim'],
            num_layers=config['experiment']['ngf']['num_layers']
        )
        ngf = NGFAdapter(model_core, mode="pytorch_geometric")
    else:
        raise ValueError("Invalid NGF implementation.")
    
    
    if config['experiment']['ngf']['frozen']:
        for p in ngf.model.parameters():
            p.requires_grad = False

    # Generate NGF embeddings
    ngf.model.eval()
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
    ecfp_all, ngf_all, ecfp_sample, ngf_sample, r = run_pairwise_analysis(
        emb_mat, fps_ecfp,
        sample_size=config['experiment']['evaluation']['num_pairs']
    )

    # Plot
    plot_pairwise_distances(ecfp_sample, ngf_sample, r, title=config['experiment']['dataset'])
    
    # Downstream Evaluation
    labels_np = np.array([data.y.item() for data in dataset])

    task_type = config['experiment']['evaluation']['downstream_task']
    dataset_name = config['experiment']['dataset']

    results_frozen = run_frozen_downstream_task(
        ecfp_array=fps_ecfp,
        ngf_array=emb_mat,
        labels=np.array(labels_np),
        task_type=task_type
    )

    print(f"\n[Frozen Downstream Evaluation] {task_type} on {dataset_name} dataset")
    print(f"ECFP   → mean: {results_frozen['ecfp'][0]:.4f}, std: {results_frozen['ecfp'][1]:.4f}")
    print(f"NGF    → mean: {results_frozen['ngf'][0]:.4f}, std: {results_frozen['ngf'][1]:.4f}")
    
    if config['experiment']['evaluation'].get("train_end_to_end", False):    
        full_model = NGFWithHead(
            ngf_base=ngf.model,
            task_type=config['experiment']['evaluation']['downstream_task'],
            hidden_dim=config['experiment']['ngf']['hidden_dim']  # same as NGF base
        )
        print("Running end-to-end training with NGF...")
        results_trained = run_end_to_end_training(full_model, dataset, task_type)
        print(f"\n[Frozen Downstream Evaluation] {task_type} on {dataset_name} dataset")
        print(f"NGF    → mean: {results_trained[0]:.4f}, std: {results_trained[1]:.4f}")
    
    # Save results 
    log_dir = create_experiment_dir(config['experiment']['dataset'])
    save_distance_plot(log_dir, ecfp_sample, ngf_sample, r, title=config['experiment']['dataset'])
    save_distances_csv(log_dir, ecfp_all, ngf_all)
    results_to_log = {
        "dataset": config['experiment']['dataset'],
        "pearson_r": r,
        "frozen downstream task": {
            "task": config['experiment']['evaluation']['downstream_task'],
            "ecfp_mean": float(results_frozen['ecfp'][0]),
            "ecfp_std": float(results_frozen['ecfp'][1]),
            "ngf_mean": float(results_frozen['ngf'][0]),
            "ngf_std": float(results_frozen['ngf'][1])
        },
        "end to end trained downstream task": {
        },
        "config": config['experiment']  
    }
    save_results(log_dir, results_to_log)

if __name__ == "__main__":
    config_path = Path("config/experiment.yaml")
    config = load_config(config_path)
    run_experiment(config)

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr
from rdkit import DataStructs


def cont_tanimoto_minmax(x, y):
    """Continuous generalization of Tanimoto distance using min/max."""
    num = np.sum(np.minimum(x, y))
    denom = np.sum(np.maximum(x, y)) + 1e-8
    return 1.0 - (num / denom)


def run_pairwise_analysis(ngf_embeddings, ecfp_fps, num_pairs=1000, seed=42):
    """
    Compute pairwise distances and Pearson correlation between NGF and ECFP.
    
    Args:
        ngf_embeddings (np.ndarray): [N, D] array of fingerprint vectors
        ecfp_fps (list): List of RDKit fingerprint objects (ExplicitBitVect or array-like)
        num_pairs (int): Number of unique (i, j) pairs to sample
        seed (int): Random seed
    
    Returns:
        ecfp_dists, ngf_dists (np.ndarray), r (float)
    """
    N = len(ngf_embeddings)
    rng = np.random.default_rng(seed)
    all_pairs = list(combinations(range(N), 2))
    sample_idxs = rng.choice(len(all_pairs), size=num_pairs, replace=False)
    pairs = [all_pairs[i] for i in sample_idxs]

    ecfp_dists = np.array([
        cont_tanimoto_minmax(
            np.array(ecfp_fps[i], dtype=np.float32),
            np.array(ecfp_fps[j], dtype=np.float32)
        )
        for i, j in pairs
    ])

    ngf_dists = np.array([
        cont_tanimoto_minmax(ngf_embeddings[i], ngf_embeddings[j])
        for i, j in pairs
    ])

    r, _ = pearsonr(ecfp_dists, ngf_dists)
    return ecfp_dists, ngf_dists, r


def plot_pairwise_distances(ecfp_dists, ngf_dists, r, title='NGF vs ECFP Distances'):
    """
    Generate Figure 3-style plot comparing NGF and ECFP distances.

    Args:
        ecfp_dists (np.ndarray): Distance vector for ECFP pairs
        ngf_dists (np.ndarray): Distance vector for NGF pairs
        r (float): Pearson correlation coefficient
        title (str): Plot title
    """
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
    plt.title(f"{title}\n$r={r:.3f}$", fontsize=14)
    plt.tight_layout()
    plt.show()

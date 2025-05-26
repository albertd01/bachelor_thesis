import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr
from rdkit import DataStructs


def cont_tanimoto_minmax(x, y):
    num = np.sum(np.minimum(x, y))
    denom = np.sum(np.maximum(x, y)) + 1e-8
    return 1.0 - (num / denom)


def run_pairwise_analysis(ngf_embeddings, ecfp_fps, sample_size=2000, seed=42):
    """
    Compute pairwise distances between all pairs of NGF and ECFP representations.
    Use all distances to compute Pearson r, but only sample a subset for plotting.

    Args:
        ngf_embeddings (np.ndarray): shape [N, D]
        ecfp_fps (List[ExplicitBitVect]): RDKit bit vectors
        sample_size (int): number of pairs to sample for plotting
        seed (int): RNG seed for reproducibility

    Returns:
        ecfp_dists_all, ngf_dists_all (np.ndarray): distances over all pairs
        ecfp_dists_sampled, ngf_dists_sampled (np.ndarray): sampled distances for plotting
        r (float): Pearson correlation over all pairs
    """
    N = len(ngf_embeddings)
    all_pairs = list(combinations(range(N), 2))

    # Compute all distances
    ecfp_dists_all = np.empty(len(all_pairs))
    ngf_dists_all = np.empty(len(all_pairs))

    for idx, (i, j) in enumerate(all_pairs):
        ecfp_dists_all[idx] = cont_tanimoto_minmax(
            np.array(ecfp_fps[i], dtype=np.float32),
            np.array(ecfp_fps[j], dtype=np.float32)
        )
        ngf_dists_all[idx] = cont_tanimoto_minmax(ngf_embeddings[i], ngf_embeddings[j])

    # Compute Pearson r over all pairs
    r, _ = pearsonr(ecfp_dists_all, ngf_dists_all)

    # Sample subset for plotting
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(all_pairs), size=min(sample_size, len(all_pairs)), replace=False)
    ecfp_dists_sampled = ecfp_dists_all[sampled_indices]
    ngf_dists_sampled = ngf_dists_all[sampled_indices]

    return ecfp_dists_all, ngf_dists_all, ecfp_dists_sampled, ngf_dists_sampled, r



def plot_pairwise_distances(ecfp_sampled, ngf_sampled, r, title='NGF vs ECFP Distances'):
    plt.figure(figsize=(5, 5))
    plt.scatter(
        ecfp_sampled,
        ngf_sampled,
        s=20,
        alpha=0.4,
        edgecolors='black',
        linewidths=0.2,
        facecolor='C0'
    )
    plt.xlabel("Circular fingerprint distances")
    plt.ylabel("Neural fingerprint distances")
    plt.xlim(0.5, 1.0)
    plt.ylim(0.5, 1.0)
    plt.grid(True, linestyle=':', linewidth=0.5)
    plt.title(f"{title}\n$r={r:.3f}$", fontsize=14)
    plt.tight_layout()
    plt.show()

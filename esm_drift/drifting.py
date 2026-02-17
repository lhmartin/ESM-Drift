"""Drifting field computation and loss.

Implements the core algorithm from "Generative Modeling via Drifting"
(Deng et al., 2026, arXiv:2602.04770).

The drifting field V(x) combines attraction toward real data (positives)
and repulsion from generated samples (negatives):

    V(x) = V⁺(x) - V⁻(x)

where V⁺/V⁻ are kernel-weighted mean shifts.

We use a cosine-similarity kernel k(x,y) = exp(cos(x,y)/τ) which is
scale-invariant and works well in high-dimensional embedding spaces.

Anti-symmetry (N_pos == N_neg, same kernel) is critical for convergence.

To handle the curse of dimensionality in 1024D embedding space, we compute
the kernel in a PCA-projected lower-dimensional space (e.g., 64D) where
distance concentration is much less severe. The drifting field is computed
in the full space so that gradients flow through all 1024 dimensions.
"""

import torch
import torch.nn.functional as F


def compute_pca(features: torch.Tensor, n_components: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute PCA projection matrix from features.

    Args:
        features: [N, D] feature vectors
        n_components: number of PCA components

    Returns:
        pca_mean: [D] mean of features
        pca_components: [D, n_components] projection matrix (columns are eigenvectors)
    """
    with torch.no_grad():
        pca_mean = features.mean(dim=0)
        centered = features - pca_mean
        # SVD on centered features
        U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        # Top n_components eigenvectors
        pca_components = Vt[:n_components].T  # [D, n_components]
        # Log variance explained
        total_var = (S ** 2).sum()
        explained_var = (S[:n_components] ** 2).sum()
        print(f"PCA: {n_components} components explain {explained_var/total_var:.1%} of variance")
    return pca_mean, pca_components


def compute_drifting_field(
    x: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    tau: float = 0.1,
    pca_mean: torch.Tensor | None = None,
    pca_components: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the drifting field V(x) using cosine similarity kernel.

    If pca_mean and pca_components are provided, the kernel similarity is
    computed in the PCA-projected space (to avoid curse of dimensionality),
    but the mean shift is computed in the original full space.

    Args:
        x: [N, D] feature vectors of generated samples
        positives: [M, D] feature vectors of real data
        negatives: [K, D] feature vectors of generated samples
        tau: temperature for cosine similarity kernel
        pca_mean: [D] mean for PCA centering (optional)
        pca_components: [D, d_pca] PCA projection matrix (optional)

    Returns:
        V: [N, D] drifting field at each generated sample
    """
    if pca_components is not None:
        # Project to PCA space for kernel computation
        x_proj = (x - pca_mean) @ pca_components  # [N, d_pca]
        pos_proj = (positives - pca_mean) @ pca_components  # [M, d_pca]
        neg_proj = (negatives - pca_mean) @ pca_components  # [K, d_pca]
        # Cosine similarity in PCA space
        x_norm = F.normalize(x_proj, dim=-1)
        pos_norm = F.normalize(pos_proj, dim=-1)
        neg_norm = F.normalize(neg_proj, dim=-1)
    else:
        # Cosine similarity in original space
        x_norm = F.normalize(x, dim=-1)
        pos_norm = F.normalize(positives, dim=-1)
        neg_norm = F.normalize(negatives, dim=-1)

    sim_pos = x_norm @ pos_norm.T  # [N, M]
    sim_neg = x_norm @ neg_norm.T  # [N, K]

    # Softmax weights
    weights_pos = F.softmax(sim_pos / tau, dim=1)  # [N, M]
    weights_neg = F.softmax(sim_neg / tau, dim=1)  # [N, K]

    # Mean shift in the ORIGINAL (full-dimensional) space
    v_plus = weights_pos @ positives - x   # [N, D]
    v_minus = weights_neg @ negatives - x  # [N, D]

    return v_plus - v_minus


def drifting_loss(
    generated_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    tau: float = 0.1,
    pca_mean: torch.Tensor | None = None,
    pca_components: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the drifting training loss.

    L = E[||φ(x) - sg(φ(x) + V(φ(x)))||²]

    The stop-gradient ensures we don't differentiate through V.
    """
    V = compute_drifting_field(
        generated_features.detach(),
        positive_features,
        negative_features,
        tau=tau,
        pca_mean=pca_mean,
        pca_components=pca_components,
    )
    target = (generated_features.detach() + V).detach()
    return F.mse_loss(generated_features, target)


def multi_tau_drifting_loss(
    generated_features: torch.Tensor,
    positive_features: torch.Tensor,
    negative_features: torch.Tensor,
    taus: list[float] = (0.05, 0.1, 0.3),
    pca_mean: torch.Tensor | None = None,
    pca_components: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drifting loss averaged over multiple kernel temperatures."""
    total = torch.tensor(0.0, device=generated_features.device)
    for tau in taus:
        total = total + drifting_loss(
            generated_features, positive_features, negative_features, tau,
            pca_mean=pca_mean, pca_components=pca_components,
        )
    return total / len(taus)


def adaptive_taus(
    features: torch.Tensor,
    multipliers: tuple[float, ...] = (0.25, 0.5, 1.0),
    max_samples: int = 1000,
    pca_mean: torch.Tensor | None = None,
    pca_components: torch.Tensor | None = None,
) -> list[float]:
    """Compute tau values based on median cosine similarity.

    If PCA components are provided, computes similarity in PCA space.
    """
    with torch.no_grad():
        if features.shape[0] > max_samples:
            idx = torch.randperm(features.shape[0], device=features.device)[:max_samples]
            features = features[idx]
        if pca_components is not None:
            features = (features - pca_mean) @ pca_components
        normed = F.normalize(features, dim=-1)
        sims = normed @ normed.T
        mask = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        median_sim = sims[mask].median().item()
        base_tau = max(1.0 - median_sim, 0.01)
    return [base_tau * m for m in multipliers]

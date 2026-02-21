"""Drifting field computation and loss.

Implements the core algorithm from "Generative Modeling via Drifting"
(Deng et al., 2026, arXiv:2602.04770).

The drifting field V(x) combines attraction toward real data (positives)
and repulsion from other generated samples (negatives):

    V(x) = V⁺(x) - V⁻(x)

The kernel is k(x, y) = exp(-||x-y|| / τ), an L2-distance (Laplacian/Yukawa)
kernel. Negatives are the generated samples themselves — the same batch serves
as both query points and negatives. This is required by the paper.

The kernel is doubly normalised (over both the x-rows and y-columns) to
implement the joint Z_p(x) · Z_q(x) denominator from Eq. 11 of the paper.

Anti-symmetry (N_pos == N_neg, same kernel) is non-negotiable.
"""

import torch
import torch.nn.functional as F


def compute_drifting_field(
    gen: torch.Tensor,
    positives: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Compute the drifting field V(x) using the L2-distance kernel.

    Negatives are the generated samples themselves (gen). This matches
    the paper's Algorithm 1 and the official demo notebook.

    The kernel is doubly normalised:
        K_norm[i, j] = k(i, j) / sqrt(row_sum[i] * col_sum[j])
    which correctly implements the joint Z_p(x) · Z_q(x) denominator
    from Eq. 11 of the paper.

    Args:
        gen:       [N, D] generated feature vectors (also used as negatives)
        positives: [M, D] real data feature vectors  (M should ≈ N)
        tau:       temperature for exp(-dist/tau) kernel

    Returns:
        V: [N, D] drifting field at each generated sample
    """
    N = gen.shape[0]
    targets = torch.cat([gen, positives], dim=0)  # [N+M, D]

    dist = torch.cdist(gen, targets)               # [N, N+M], L2 distances

    # Mask self-interactions: a sample should not repel itself
    dist[:, :N].fill_diagonal_(float("inf"))

    kernel = (-dist / tau).exp()                   # [N, N+M]

    # Doubly-normalised kernel: K_norm[i,j] = k(i,j) / sqrt(row_sum[i] * col_sum[j])
    row_sum = kernel.sum(dim=1, keepdim=True)       # [N, 1]
    col_sum = kernel.sum(dim=0, keepdim=True)       # [1, N+M]
    normalizer = (row_sum * col_sum).clamp_min(1e-12).sqrt()
    K = kernel / normalizer                         # [N, N+M]

    neg_K = K[:, :N]   # [N, N] — gen vs gen  (repulsion)
    pos_K = K[:, N:]   # [N, M] — gen vs pos  (attraction)

    # V(x_i) = Z_neg(x_i) * weighted_mean(pos) - Z_pos(x_i) * weighted_mean(neg)
    neg_sum = neg_K.sum(dim=1, keepdim=True)        # [N, 1]
    pos_sum = pos_K.sum(dim=1, keepdim=True)        # [N, 1]

    pos_V = (pos_K * neg_sum) @ positives           # [N, D]
    neg_V = (neg_K * pos_sum) @ gen                 # [N, D]

    return pos_V - neg_V


def drifting_loss(
    generated_features: torch.Tensor,
    positive_features: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    """Compute the drifting training loss.

    L = E[||φ(x) - sg(φ(x) + V(φ(x)))||²]

    V is computed with stop-gradient on both gen and pos so that the
    drifting field is treated as a fixed target (analogous to SimSiam).

    Args:
        generated_features: [N, D] generated embeddings (gradient flows here)
        positive_features:  [M, D] real embeddings (M should ≈ N)
        tau:                kernel temperature
    """
    with torch.no_grad():
        V = compute_drifting_field(
            generated_features.detach(),
            positive_features.detach(),
            tau=tau,
        )
        target = generated_features.detach() + V   # [N, D], fully stop-grad
    return F.mse_loss(generated_features, target)


def multi_tau_drifting_loss(
    generated_features: torch.Tensor,
    positive_features: torch.Tensor,
    taus: list[float],
) -> torch.Tensor:
    """Drifting loss averaged over multiple kernel temperatures."""
    total = torch.tensor(0.0, device=generated_features.device)
    for tau in taus:
        total = total + drifting_loss(generated_features, positive_features, tau)
    return total / len(taus)


def per_protein_drifting_loss(
    gen_s_s: torch.Tensor,
    pos_s_s: torch.Tensor,
    pos_mask: torch.Tensor,
    taus: list[float],
) -> torch.Tensor:
    """Per-protein drifting loss — no cross-protein residue mixing.

    For each pair (gen_i, real_i), the drifting field is computed using only
    gen_i's residues as generators and real_i's valid residues as positives.
    The negatives are gen_i's own residues (within-protein self-repulsion),
    so the field pushes each generated protein toward its matched real protein
    without mixing residues across different proteins.

    Args:
        gen_s_s:  [B, L, D]   generated protein embeddings
        pos_s_s:  [B, L, D]   padded real protein embeddings (matched 1-to-1)
        pos_mask: [B, L]      True where real residues are valid (not padding)
        taus:     list of kernel temperatures

    Returns:
        scalar loss, mean over B proteins
    """
    B = gen_s_s.shape[0]
    losses = []
    for i in range(B):
        gen_i = gen_s_s[i]              # [L, D]
        pos_i = pos_s_s[i, pos_mask[i]] # [L_i, D]  valid residues only
        if pos_i.shape[0] == 0:
            continue
        losses.append(multi_tau_drifting_loss(gen_i, pos_i, taus))
    if not losses:
        return torch.tensor(0.0, device=gen_s_s.device)
    return torch.stack(losses).mean()


def adaptive_taus(
    features: torch.Tensor,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
    max_samples: int = 1000,
) -> list[float]:
    """Compute tau values based on the median pairwise L2 distance.

    Uses real data features to calibrate tau so that the kernel exp(-d/tau)
    gives meaningful weights at typical inter-sample distances.
    """
    with torch.no_grad():
        if features.shape[0] > max_samples:
            idx = torch.randperm(features.shape[0], device=features.device)[:max_samples]
            features = features[idx]
        # Pairwise L2 distances
        dists = torch.cdist(features, features)  # [N, N]
        mask = ~torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
        median_dist = dists[mask].median().item()
        # tau = median_dist so that exp(-1) ≈ 0.37 weight at the median distance
        base_tau = max(median_dist, 1e-3)
    return [base_tau * m for m in multipliers]

#!/usr/bin/env python3
"""Training script for ESM-Drift (per-residue drifting).

Trains a drifting generator that maps noise to ESMFold s_s embeddings.

The drifting field operates on individual residue embeddings (not pooled),
giving each residue its own gradient signal. All valid residues from generated
and real proteins are flattened into bags and matched via the cosine kernel.

Usage:
    uv run python scripts/train.py --data_dir data/ --max_seq_len 64 --device cuda
    uv run python scripts/train.py --data_dir data/ --device cuda --epochs 5000 --lr 3e-4
"""

import argparse
import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from esm_drift.data.dataset import EmbeddingDataset, pad_collate
from esm_drift.drifting import adaptive_taus, multi_tau_drifting_loss
from esm_drift.model import DriftingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_real_residues(
    dataset: EmbeddingDataset, max_len: int, device: torch.device
) -> torch.Tensor:
    """Load all real data and return flattened raw residue embeddings.

    Returns:
        real_residues: [n_total_valid, 1024] flattened valid residues (raw)
    """
    collate = partial(pad_collate, max_len=max_len)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    real_s_s = batch["s_s"].to(device)
    real_mask = batch["mask"].to(device)

    real_residues = real_s_s[real_mask]  # [n_valid, 1024]

    log.info(
        "Real data: %d proteins, %d total residues, padded to L=%d",
        real_s_s.shape[0], real_residues.shape[0], max_len,
    )
    log.info(
        "  Per-residue norm: mean=%.2f, std=%.2f",
        real_residues.norm(dim=-1).mean().item(),
        real_residues.norm(dim=-1).std().item(),
    )
    return real_residues.detach()


def train(
    generator: DriftingGenerator,
    real_residues: torch.Tensor,
    max_len: int,
    device: torch.device,
    epochs: int = 5000,
    batch_size: int = 32,
    lr: float = 3e-4,
    taus: list[float] | None = None,
    eval_every: int = 200,
    save_dir: str = "checkpoints",
):
    """Train the drifting generator with per-residue matching.

    Uses cosine kernel in raw (unstandardized) 1024D space — the cosine kernel
    is naturally scale-invariant and works well for direction matching. An
    auxiliary norm-matching loss ensures generated residues have the right scale.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    n_real = real_residues.shape[0]
    real_mean_norm = real_residues.norm(dim=-1).mean()
    real_norm_std = real_residues.norm(dim=-1).std()

    # Compute adaptive tau from real residue cosine similarities
    if taus is None:
        taus = adaptive_taus(real_residues, multipliers=(0.5, 1.0, 2.0))
    log.info(
        "Training: %d epochs, batch=%d, lr=%s, taus=%s",
        epochs, batch_size, lr, [f"{t:.4f}" for t in taus],
    )
    log.info("  Real norm: mean=%.1f, std=%.1f", real_mean_norm.item(), real_norm_std.item())

    generator.train()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # All generated positions are valid (no padding for generated)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
    n_gen_residues = batch_size * max_len

    best_loss = float("inf")
    norm_weight = 5000.0  # weight for norm-matching penalty (must compete with drift loss ~10k)

    for epoch in range(1, epochs + 1):
        # Generate
        noise = generator.sample_noise(batch_size, max_len, device)
        gen_s_s = generator(noise, mask)  # [B, L, 1024]

        # Flatten all generated residues (gradient flows through)
        gen_residues = gen_s_s.reshape(-1, 1024)  # [B*L, 1024]

        # Anti-symmetry: N_pos = N_neg
        n_ref = min(n_real, n_gen_residues)
        neg_idx = torch.randperm(n_gen_residues, device=device)[:n_ref]
        neg_feat = gen_residues[neg_idx].detach()

        # Add noise to negatives to break collapse equilibrium.
        # Without this, if all gen residues collapse to the same point,
        # V⁻ = 0 and there's no repulsive signal to restore diversity.
        neg_noise_scale = real_norm_std * 2  # ~100, proportional to real spread
        neg_feat = neg_feat + neg_noise_scale * torch.randn_like(neg_feat)

        if n_real >= n_ref:
            pos_idx = torch.randperm(n_real, device=device)[:n_ref]
        else:
            pos_idx = torch.randint(n_real, (n_ref,), device=device)
        pos_feat = real_residues[pos_idx]

        # Drifting loss (cosine kernel in raw space)
        drift_loss = multi_tau_drifting_loss(
            gen_residues, pos_feat, neg_feat, taus=list(taus),
        )

        # Norm-matching: MSE on mean norms (large enough to matter)
        gen_norms = gen_residues.norm(dim=-1)
        norm_loss = ((gen_norms.mean() - real_mean_norm) / real_mean_norm) ** 2

        loss = drift_loss + norm_weight * norm_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/loss": loss.item(),
            "train/drift_loss": drift_loss.item(),
            "train/norm_loss": norm_loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/epoch": epoch,
        })

        if epoch % eval_every == 0:
            with torch.no_grad():
                # Per-residue distance metrics
                cross = torch.cdist(
                    gen_residues[:200],
                    real_residues[:200],
                )
                mean_dist = cross.mean().item()
                min_dist = cross.min(dim=1).values.mean().item()

                gen_norm_val = gen_norms.mean().item()
                real_norm_val = real_mean_norm.item()

                # Per-residue cosine similarity
                gen_normed = F.normalize(gen_residues[:200], dim=-1)
                real_normed = F.normalize(real_residues[:200], dim=-1)
                cos_sim = (gen_normed @ real_normed.T).mean().item()

                # Diversity: variance across generated residues
                gen_std = gen_residues.std(dim=0).mean().item()
                real_std = real_residues.std(dim=0).mean().item()

            log.info(
                "Epoch %d/%d  loss=%.4f (drift=%.4f norm=%.4f)  grad=%.4f  "
                "cos_sim=%.4f  mean_dist=%.1f  min_dist=%.1f  "
                "gen_norm=%.0f  real_norm=%.0f  gen_std=%.2f  real_std=%.2f",
                epoch, epochs, loss.item(), drift_loss.item(), norm_loss.item(),
                grad_norm.item(), cos_sim, mean_dist, min_dist,
                gen_norm_val, real_norm_val, gen_std, real_std,
            )
            wandb.log({
                "eval/cos_sim": cos_sim,
                "eval/mean_dist": mean_dist,
                "eval/min_dist": min_dist,
                "eval/gen_norm": gen_norm_val,
                "eval/gen_std": gen_std,
                "eval/real_std": real_std,
            })

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "generator": generator.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                    "config": {
                        "max_len": max_len,
                        "taus": list(taus),
                        "batch_size": batch_size,
                    },
                }, save_path / "best.pt")

    # Save final
    torch.save({
        "generator": generator.state_dict(),
        "epoch": epochs,
        "loss": loss.item(),
        "config": {"max_len": max_len, "taus": list(taus), "batch_size": batch_size},
    }, save_path / "final.pt")
    log.info("Training complete. Best loss: %.6f", best_loss)


def main():
    parser = argparse.ArgumentParser(description="Train ESM-Drift (per-residue)")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--taus", type=float, nargs="+", default=None)

    parser.add_argument("--d_noise", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--wandb_project", type=str, default="esm-drift")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # Load dataset
    dataset = EmbeddingDataset(args.data_dir, max_seq_len=args.max_seq_len)
    log.info("Dataset: %d samples, max_len=%d", len(dataset), args.max_seq_len)
    if len(dataset) == 0:
        log.error("No samples found!")
        return

    # Load and cache real residue embeddings (raw)
    real_residues = load_real_residues(dataset, args.max_seq_len, device)

    # Create generator
    generator = DriftingGenerator(
        d_noise=args.d_noise,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        s_s_dim=1024,
        max_len=args.max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in generator.parameters())
    log.info("Generator: %d params (%.1fM)", n_params, n_params / 1e6)

    train(
        generator, real_residues,
        max_len=args.max_seq_len,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        taus=args.taus,
        save_dir=args.save_dir,
    )
    wandb.finish()


if __name__ == "__main__":
    main()

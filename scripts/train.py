#!/usr/bin/env python3
"""Training script for ESM-Drift (per-residue drifting, no feature encoder).

Trains a DriftingGeneratorUNet that maps noise → ESMFold s_s embeddings.

The drifting field operates directly on per-residue s_s embeddings, rescaled
by the real data's mean norm so that pairwise distances are O(1) and the
L2 kernel exp(-d/tau) gives useful gradients. No separate feature encoder is
needed: the UNet's internal bottleneck (d_noise→d_model→d_mid→d_bottleneck→...)
is the learned compression; the rescaling is a fixed, parameter-free operation.

Why no FeatureEncoder:
  - ESMFold s_s is already feature-rich (36-layer ESM-2)
  - A separate encoder creates a chicken-and-egg collapse: it learns real data
    well but maps all (initially poor) gen vectors to the same point, making
    the drifting field V≈0 and blocking the generator from learning
  - The scale problem (norms ~4000 → kernel≈0) is solved by fixed rescaling,
    not by a learned compression layer

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
from esm_drift.drifting import (
    adaptive_taus,
    masked_mean_pool,
    multi_tau_drifting_loss,
)
from esm_drift.model import DriftingGeneratorUNet, SeqHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_real_residues(
    dataset: EmbeddingDataset, max_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load all real data.

    Returns:
        real_residues: [n_total_valid, 1024] flattened valid residues
        real_s_s:      [N, max_len, 1024] padded protein embeddings
        real_aa:       [N, max_len] LongTensor amino acid indices (0-19, 20=unk)
        real_mask:     [N, max_len] boolean mask
        real_means:    [N, 1024] per-protein mean-pooled embeddings
        real_seq_lens: [N] actual sequence lengths (for dynamic batching)
    """
    collate = partial(pad_collate, max_len=max_len)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    real_s_s = batch["s_s"].to(device)
    real_mask = batch["mask"].to(device)
    real_aa = batch["aa_indices"].to(device)
    real_seq_lens = batch["seq_lens"].to(device)  # [N] actual lengths

    real_residues = real_s_s[real_mask]  # [n_valid, 1024]
    real_means = masked_mean_pool(real_s_s, real_mask)  # [N, 1024]

    log.info(
        "Real data: %d proteins, %d total residues, padded to L=%d",
        real_s_s.shape[0], real_residues.shape[0], max_len,
    )
    log.info(
        "  Sequence lengths: min=%d, median=%d, max=%d",
        real_seq_lens.min().item(),
        real_seq_lens.median().item(),
        real_seq_lens.max().item(),
    )
    log.info(
        "  Per-residue norm: mean=%.2f, std=%.2f",
        real_residues.norm(dim=-1).mean().item(),
        real_residues.norm(dim=-1).std().item(),
    )
    log.info(
        "  Per-protein mean norm: mean=%.2f, std=%.2f",
        real_means.norm(dim=-1).mean().item(),
        real_means.norm(dim=-1).std().item(),
    )
    return (
        real_residues.detach(),
        real_s_s.detach(),
        real_aa.detach(),
        real_mask.detach(),
        real_means.detach(),
        real_seq_lens.detach(),
    )


def train(
    generator: DriftingGeneratorUNet,
    real_residues: torch.Tensor,
    real_s_s: torch.Tensor,
    real_aa: torch.Tensor,
    real_mask: torch.Tensor,
    real_means: torch.Tensor,
    real_seq_lens: torch.Tensor,
    max_len: int,
    device: torch.device,
    epochs: int = 5000,
    batch_size: int = 32,
    lr: float = 3e-4,
    taus: list[float] | None = None,
    eval_every: int = 200,
    save_dir: str = "checkpoints",
    max_grad_norm: float = 2.0,
    eta_min: float = 1e-5,
    warmup_T0: int = 2000,
    tau_recal_every: int = 1000,
    seq_ce_weight: float = 0.05,
):
    """Train the drifting generator with global residue-level drifting.

    Loss terms:
      1. drift_loss: global drifting on residues. gen_residues [B*L, 1024] are attracted
         toward B*L randomly-sampled real residues (positives) and repelled from each other
         (negatives). Taus calibrated on real residue pairwise distances in raw 1024D.
         Cross-protein repulsion IS present here: gen residues from protein i repel gen
         residues from protein j, providing natural diversity.
      2. norm_loss: drives gen_norm toward real_norm (needed for ESMFold decoding).
      3. gen_ce_loss / real_ce_loss: sequence cross-entropy.

    Architecture notes:
      - protein_cond in DriftingGeneratorUNet: projects noise.mean(dim=1) to s_s space,
        providing a protein-specific offset. z_protein broadcast ensures each protein has
        a unique identity that survives mean-pooling.
      - noise_skip: per-position diversity.
      - output_scale: learnable global amplitude matching.

    warmup_T0: restart period for CosineAnnealingWarmRestarts (0 = plain cosine).
    tau_recal_every: re-calibrate taus from current gen distribution every N epochs.
    seq_ce_weight: weight for CE losses.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    n_real = real_residues.shape[0]
    n_proteins = real_s_s.shape[0]

    real_mean_norm = real_residues.norm(dim=-1).mean()

    if taus is None:
        taus = adaptive_taus(real_residues, multipliers=(0.5, 1.0, 2.0))
    taus = list(taus)

    log.info(
        "Training: %d epochs, batch=%d, lr=%s, taus=%s",
        epochs, batch_size, lr, [f"{t:.1f}" for t in taus],
    )
    log.info("  Warm restarts T0=%d, tau recal every=%d", warmup_T0, tau_recal_every)

    seq_head = SeqHead(s_s_dim=1024).to(device)

    generator.train()
    seq_head.train()
    optimizer = torch.optim.AdamW(
        list(generator.parameters()) + list(seq_head.parameters()),
        lr=lr, weight_decay=1e-4,
    )
    if warmup_T0 > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=warmup_T0, T_mult=1, eta_min=eta_min
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=eta_min
        )

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # Sample B real proteins, then generate at their actual max length.
        # This eliminates padding waste: a batch of short proteins runs the
        # Transformer at L_batch << max_len, giving 2-4x speedup on short data.
        if n_proteins >= batch_size:
            prot_idx = torch.randperm(n_proteins, device=device)[:batch_size]
        else:
            prot_idx = torch.randint(n_proteins, (batch_size,), device=device)

        # L_batch = longest real protein in this batch (no unnecessary padding)
        L_batch = int(real_seq_lens[prot_idx].max().item())

        gen_mask = torch.ones(batch_size, L_batch, dtype=torch.bool, device=device)

        noise = generator.sample_noise(batch_size, L_batch, device)
        # z_protein: per-protein broadcast offset that survives mean-pooling,
        # giving protein_cond(noise.mean(dim=1)) a protein-specific signal.
        z_protein = torch.randn(batch_size, 1, generator.d_noise, device=device)
        noise = noise + z_protein
        gen_s_s = generator(noise, gen_mask)  # [B, L_batch, 1024]

        gen_residues = gen_s_s.reshape(-1, 1024)  # [B*L_batch, 1024]

        prot_s_s   = real_s_s[prot_idx, :L_batch, :]    # [B, L_batch, 1024]
        prot_mask  = real_mask[prot_idx, :L_batch]       # [B, L_batch]
        # Valid real residues from this batch — used as drifting positives
        pos_residues = prot_s_s[prot_mask]  # [n_valid, 1024]

        # Global drifting: all B*L gen residues are attracted toward the batch's
        # real residues (positives) and repelled from each other (negatives).
        # Cross-protein repulsion is implicit: gen_residue from protein i repels
        # gen_residues from protein j, driving protein-level diversity.
        drift_loss = multi_tau_drifting_loss(gen_residues, pos_residues, taus)

        # Norm-matching
        gen_norms = gen_residues.norm(dim=-1)
        norm_loss = ((gen_norms.mean() - real_mean_norm) / real_mean_norm) ** 2

        # CE: generated s_s should predict the sequence of the paired real protein
        batch_aa        = real_aa[prot_idx, :L_batch]
        batch_real_mask = real_mask[prot_idx, :L_batch]

        known_aa = batch_real_mask & (batch_aa < 20)

        if known_aa.any():
            gen_ce_loss = F.cross_entropy(
                seq_head(gen_s_s)[known_aa],
                batch_aa[known_aa],
            )
            real_ce_loss = F.cross_entropy(
                seq_head(prot_s_s)[known_aa],
                batch_aa[known_aa],
            )
        else:
            gen_ce_loss = real_ce_loss = torch.tensor(0.0, device=device)

        loss = (drift_loss
                + 200.0 * norm_loss
                + seq_ce_weight * gen_ce_loss
                + 0.5 * seq_ce_weight * real_ce_loss)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(generator.parameters()) + list(seq_head.parameters()),
            max_norm=max_grad_norm,
        )
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/loss": loss.item(),
            "train/drift_loss": drift_loss.item(),
            "train/norm_loss": norm_loss.item(),
            "train/gen_ce_loss": gen_ce_loss.item(),
            "train/real_ce_loss": real_ce_loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/epoch": epoch,
        })

        if epoch % eval_every == 0:
            with torch.no_grad():
                cross = torch.cdist(gen_residues[:200], real_residues[:200])
                mean_dist = cross.mean().item()
                min_dist  = cross.min(dim=1).values.mean().item()

                gen_norm_val  = gen_norms.mean().item()
                real_norm_val = real_mean_norm.item()

                gen_normed  = F.normalize(gen_residues[:200], dim=-1)
                real_normed = F.normalize(real_residues[:200], dim=-1)
                cos_sim = (gen_normed @ real_normed.T).mean().item()

                gen_std  = gen_residues.std(dim=0).mean().item()
                real_std = real_residues.std(dim=0).mean().item()

                # Protein-level diversity: pairwise L2 of per-protein mean-pooled embeddings
                gen_means = masked_mean_pool(gen_s_s, gen_mask)  # [B, 1024]
                gen_pairwise = torch.cdist(gen_means, gen_means)
                off_diag_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
                gen_pairwise_l2 = gen_pairwise[off_diag_mask].mean().item()
                # Compare against real protein mean pairwise L2
                real_means_batch = real_means[prot_idx]
                real_pairwise = torch.cdist(real_means_batch, real_means_batch)
                real_pairwise_l2 = real_pairwise[off_diag_mask].mean().item()

                # Nearest real protein index for each gen protein
                all_dists = torch.cdist(gen_means, real_means)  # [B, N_real]
                nearest_idx = all_dists.argmin(dim=1).tolist()
                n_unique = len(set(nearest_idx))

                # Sequence recovery
                eval_aa_flat  = real_aa[real_mask][:500]
                eval_emb_flat = real_residues[:500]
                valid = eval_aa_flat < 20
                if valid.any():
                    logits  = seq_head(eval_emb_flat[valid])
                    seq_acc = (logits.argmax(-1) == eval_aa_flat[valid]).float().mean().item()
                else:
                    seq_acc = 0.0

            log.info(
                "Epoch %d/%d  loss=%.4f (drift=%.4f norm=%.4f gen_ce=%.4f real_ce=%.4f)  "
                "grad=%.4f  cos_sim=%.4f  mean_dist=%.3f  min_dist=%.3f  "
                "gen_norm=%.0f  real_norm=%.0f  gen_std=%.2f  real_std=%.2f  seq_acc=%.3f  "
                "prot_L2=%.1f (real=%.1f)  unique=%d/%d",
                epoch, epochs, loss.item(), drift_loss.item(),
                norm_loss.item(), gen_ce_loss.item(), real_ce_loss.item(),
                grad_norm.item(), cos_sim, mean_dist, min_dist,
                gen_norm_val, real_norm_val, gen_std, real_std, seq_acc,
                gen_pairwise_l2, real_pairwise_l2, n_unique, batch_size,
            )
            wandb.log({
                "eval/cos_sim": cos_sim,
                "eval/mean_dist": mean_dist,
                "eval/min_dist": min_dist,
                "eval/gen_norm": gen_norm_val,
                "eval/gen_std": gen_std,
                "eval/real_std": real_std,
                "eval/seq_acc": seq_acc,
                "eval/gen_pairwise_l2": gen_pairwise_l2,
                "eval/real_pairwise_l2": real_pairwise_l2,
                "eval/unique_nearest": n_unique,
                "eval/L_batch": L_batch,
            })

            if tau_recal_every > 0 and epoch % tau_recal_every == 0:
                new_taus = adaptive_taus(gen_residues.detach(), multipliers=(0.5, 1.0, 2.0))
                taus = list(new_taus)
                log.info(
                    "  [tau recal @ epoch %d]  new taus: %s",
                    epoch, [f"{t:.3f}" for t in taus],
                )
                wandb.log({"train/tau_base": taus[1], "train/epoch": epoch})

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save({
                    "generator": generator.state_dict(),
                    "seq_head": seq_head.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                    "config": {
                        "model_type": "unet",
                        "max_len": max_len,
                        "taus": list(taus),
                        "batch_size": batch_size,
                        "d_noise": generator.d_noise,
                        "d_model": generator.d_model,
                        "d_bottleneck": getattr(generator, "d_bottleneck", None),
                        "nhead": generator.nhead,
                        "enc_layers": generator.enc_layers,
                        "dec_layers": generator.dec_layers,
                        "num_layers": generator.num_layers,
                    },
                }, save_path / "best.pt")

    torch.save({
        "generator": generator.state_dict(),
        "seq_head": seq_head.state_dict(),
        "epoch": epochs,
        "loss": loss.item(),
        "config": {
            "model_type": "unet",
            "max_len": max_len,
            "taus": list(taus),
            "batch_size": batch_size,
            "d_noise": generator.d_noise,
            "d_model": generator.d_model,
            "d_bottleneck": getattr(generator, "d_bottleneck", None),
            "nhead": generator.nhead,
            "num_layers": generator.num_layers,
        },
    }, save_path / "final.pt")
    log.info("Training complete. Best loss: %.6f", best_loss)


def main():
    parser = argparse.ArgumentParser(description="Train ESM-Drift (per-residue, no feature encoder)")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="checkpoints")

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--warmup_T0", type=int, default=2000,
                        help="LR warm restart period (epochs). 0 = no restarts.")
    parser.add_argument("--tau_recal_every", type=int, default=1000,
                        help="Re-calibrate taus from current gen distribution every N epochs (0=off).")
    parser.add_argument("--taus", type=float, nargs="+", default=None)

    parser.add_argument("--d_noise", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_bottleneck", type=int, default=128,
                        help="Bottleneck dim for unet model.")
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Layers per half of the U-Net (enc=dec=num_layers).")

    parser.add_argument("--seq_ce_weight", type=float, default=0.05)

    parser.add_argument("--wandb_project", type=str, default="esm-drift")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    device = torch.device(args.device)

    if args.no_wandb:
        wandb.init(mode="disabled")
    else:
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    dataset = EmbeddingDataset(args.data_dir, max_seq_len=args.max_seq_len)
    log.info("Dataset: %d samples, max_len=%d", len(dataset), args.max_seq_len)
    if len(dataset) == 0:
        log.error("No samples found!")
        return

    real_residues, real_s_s, real_aa, real_mask, real_means, real_seq_lens = load_real_residues(dataset, args.max_seq_len, device)

    generator = DriftingGeneratorUNet(
        d_noise=args.d_noise,
        d_model=args.d_model,
        d_bottleneck=args.d_bottleneck,
        nhead=args.nhead,
        enc_layers=args.num_layers,
        dec_layers=args.num_layers,
        s_s_dim=1024,
        max_len=args.max_seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in generator.parameters())
    log.info("Generator: %d params (%.1fM)", n_params, n_params / 1e6)

    train(
        generator, real_residues, real_s_s, real_aa, real_mask, real_means, real_seq_lens,
        max_len=args.max_seq_len,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        taus=args.taus,
        save_dir=args.save_dir,
        max_grad_norm=args.max_grad_norm,
        eta_min=args.eta_min,
        warmup_T0=args.warmup_T0,
        tau_recal_every=args.tau_recal_every,
        seq_ce_weight=args.seq_ce_weight,
    )
    wandb.finish()


if __name__ == "__main__":
    main()

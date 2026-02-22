#!/usr/bin/env python3
"""Training script for ESM-Drift (per-residue drifting, no feature encoder).

Trains a DriftingGeneratorUNet that maps noise → ESMFold s_s embeddings.

Usage:
    # Start from defaults
    uv run python scripts/train.py --save_dir checkpoints/v14

    # Load a config file
    uv run python scripts/train.py --config configs/v14.yaml

    # Load a config file and override specific fields
    uv run python scripts/train.py --config configs/v14.yaml --lr 1e-3 --wandb_name v14-hi-lr

    # Quick ablation: disable a feature
    uv run python scripts/train.py --config configs/v14.yaml --no_use_length_cond
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

from esm_drift.config import Config
from esm_drift.data.dataset import EmbeddingDataset, pad_collate
from esm_drift.drifting import (
    adaptive_taus,
    masked_mean_pool,
    multi_tau_drifting_loss,
    protein_level_drifting_loss,
)
from esm_drift.model import DriftingGeneratorUNet, SeqHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_real_residues(
    dataset: EmbeddingDataset, max_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load all real data into GPU memory.

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
    real_seq_lens = batch["seq_lens"].to(device)

    real_residues = real_s_s[real_mask]  # [n_valid, 1024]
    real_means = masked_mean_pool(real_s_s, real_mask)  # [N, 1024]

    log.info(
        "Real data: %d proteins, %d total residues, padded to L=%d",
        real_s_s.shape[0], real_residues.shape[0], max_len,
    )
    log.info(
        "  Sequence lengths: min=%d, median=%d, max=%d",
        real_seq_lens.min().item(), real_seq_lens.median().item(), real_seq_lens.max().item(),
    )
    log.info(
        "  Per-residue norm: mean=%.2f, std=%.2f",
        real_residues.norm(dim=-1).mean().item(), real_residues.norm(dim=-1).std().item(),
    )
    log.info(
        "  Per-protein mean norm: mean=%.2f, std=%.2f",
        real_means.norm(dim=-1).mean().item(), real_means.norm(dim=-1).std().item(),
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
    cfg: Config,
    device: torch.device,
):
    """Train the drifting generator. All hyperparameters come from cfg."""
    save_path = Path(cfg.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    n_proteins = real_s_s.shape[0]
    real_mean_norm = real_residues.norm(dim=-1).mean()

    taus = list(cfg.taus) if cfg.taus else adaptive_taus(real_residues, multipliers=(0.5, 1.0, 2.0))

    log.info(
        "Training: %d epochs, batch=%d, lr=%s, taus=%s",
        cfg.epochs, cfg.batch_size, cfg.lr, [f"{t:.1f}" for t in taus],
    )
    log.info("  warmup_T0=%d  T_mult=%d  prot_drift_weight=%.2f",
             cfg.warmup_T0, cfg.warmup_T_mult, cfg.prot_drift_weight)
    log.info("  use_length_cond=%s  use_strict_antisymmetry=%s  use_dynamic_len=%s",
             generator.use_length_cond, cfg.use_strict_antisymmetry, cfg.use_dynamic_len)

    seq_head = SeqHead(s_s_dim=1024).to(device)
    generator.train()
    seq_head.train()

    optimizer = torch.optim.AdamW(
        list(generator.parameters()) + list(seq_head.parameters()),
        lr=cfg.lr, weight_decay=1e-4,
    )
    if cfg.warmup_T0 > 0:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.warmup_T0, T_mult=cfg.warmup_T_mult, eta_min=cfg.eta_min
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
        )

    best_loss = float("inf")

    def _save(epoch: int, loss_val: float, filename: str):
        torch.save({
            "generator": generator.state_dict(),
            "seq_head": seq_head.state_dict(),
            "epoch": epoch,
            "loss": loss_val,
            "config": cfg.to_dict(),
        }, save_path / filename)

    for epoch in range(1, cfg.epochs + 1):
        # Sample B real proteins, generate at their actual max length (no wasted padding)
        if n_proteins >= cfg.batch_size:
            prot_idx = torch.randperm(n_proteins, device=device)[:cfg.batch_size]
        else:
            prot_idx = torch.randint(n_proteins, (cfg.batch_size,), device=device)

        if cfg.use_dynamic_len:
            L_batch = int(real_seq_lens[prot_idx].max().item())
        else:
            L_batch = cfg.max_len

        gen_mask  = torch.ones(cfg.batch_size, L_batch, dtype=torch.bool, device=device)
        prot_lens = real_seq_lens[prot_idx]  # [B] for length conditioning

        noise     = generator.sample_noise(cfg.batch_size, L_batch, device)
        z_protein = torch.randn(cfg.batch_size, 1, generator.d_noise, device=device)
        noise     = noise + z_protein
        gen_s_s   = generator(noise, gen_mask, lengths=prot_lens)  # [B, L_batch, 1024]

        gen_residues = gen_s_s.reshape(-1, 1024)  # [B*L_batch, 1024]
        prot_s_s     = real_s_s[prot_idx, :L_batch, :]   # [B, L_batch, 1024]
        prot_mask    = real_mask[prot_idx, :L_batch]      # [B, L_batch]
        pos_residues = prot_s_s[prot_mask]                # [n_valid, 1024]

        # Strict anti-symmetry: paper requires N_pos == N_neg
        if cfg.use_strict_antisymmetry:
            n_pos, n_gen = pos_residues.shape[0], gen_residues.shape[0]
            if n_gen > n_pos:
                gen_for_drift = gen_residues[torch.randperm(n_gen, device=device)[:n_pos]]
            else:
                gen_for_drift = gen_residues
        else:
            gen_for_drift = gen_residues
        drift_loss = multi_tau_drifting_loss(gen_for_drift, pos_residues, taus)

        # Protein-level drifting on unit-sphere mean-pooled embeddings
        if cfg.prot_drift_weight > 0.0:
            prot_drift_loss = protein_level_drifting_loss(
                gen_s_s, prot_s_s, prot_mask, gen_mask, taus=[0.7, 1.4, 2.8]
            )
        else:
            prot_drift_loss = torch.tensor(0.0, device=device)

        # Norm-matching
        gen_norms = gen_residues.norm(dim=-1)
        norm_loss = ((gen_norms.mean() - real_mean_norm) / real_mean_norm) ** 2

        # Sequence CE
        batch_aa     = real_aa[prot_idx, :L_batch]
        batch_mask   = real_mask[prot_idx, :L_batch]
        known_aa     = batch_mask & (batch_aa < 20)
        if known_aa.any():
            gen_ce_loss  = F.cross_entropy(seq_head(gen_s_s)[known_aa], batch_aa[known_aa])
            real_ce_loss = F.cross_entropy(seq_head(prot_s_s)[known_aa], batch_aa[known_aa])
        else:
            gen_ce_loss = real_ce_loss = torch.tensor(0.0, device=device)

        loss = (drift_loss
                + cfg.prot_drift_weight * prot_drift_loss
                + 200.0 * norm_loss
                + cfg.seq_ce_weight * gen_ce_loss
                + 0.5 * cfg.seq_ce_weight * real_ce_loss)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(generator.parameters()) + list(seq_head.parameters()),
            max_norm=cfg.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()

        wandb.log({
            "train/loss": loss.item(),
            "train/drift_loss": drift_loss.item(),
            "train/prot_drift_loss": prot_drift_loss.item(),
            "train/norm_loss": norm_loss.item(),
            "train/gen_ce_loss": gen_ce_loss.item(),
            "train/real_ce_loss": real_ce_loss.item(),
            "train/grad_norm": grad_norm.item(),
            "train/lr": scheduler.get_last_lr()[0],
            "train/epoch": epoch,
        })

        if epoch % cfg.eval_every == 0:
            with torch.no_grad():
                cross = torch.cdist(gen_residues[:200], real_residues[:200])
                mean_dist = cross.mean().item()
                min_dist  = cross.min(dim=1).values.mean().item()

                gen_normed  = F.normalize(gen_residues[:200], dim=-1)
                real_normed = F.normalize(real_residues[:200], dim=-1)
                cos_sim     = (gen_normed @ real_normed.T).mean().item()
                gen_std     = gen_residues.std(dim=0).mean().item()
                real_std    = real_residues.std(dim=0).mean().item()

                gen_means      = masked_mean_pool(gen_s_s, gen_mask)  # [B, 1024]
                off_diag       = ~torch.eye(cfg.batch_size, dtype=torch.bool, device=device)
                gen_pairwise   = torch.cdist(gen_means, gen_means)[off_diag].mean().item()
                real_pairwise  = torch.cdist(real_means[prot_idx], real_means[prot_idx])[off_diag].mean().item()

                nearest_idx = torch.cdist(gen_means, real_means).argmin(dim=1).tolist()
                n_unique    = len(set(nearest_idx))

                eval_aa_flat  = real_aa[real_mask][:500]
                eval_emb_flat = real_residues[:500]
                valid = eval_aa_flat < 20
                seq_acc = (seq_head(eval_emb_flat[valid]).argmax(-1) == eval_aa_flat[valid]).float().mean().item() if valid.any() else 0.0

            log.info(
                "Epoch %d/%d  loss=%.4f (drift=%.4f prot=%.4f norm=%.4f gen_ce=%.4f)  "
                "grad=%.4f  cos_sim=%.4f  gen_norm=%.0f  gen_std=%.2f  seq_acc=%.3f  "
                "prot_L2=%.1f (real=%.1f)  unique=%d/%d  L=%d",
                epoch, cfg.epochs, loss.item(), drift_loss.item(),
                prot_drift_loss.item(), norm_loss.item(), gen_ce_loss.item(),
                grad_norm.item(), cos_sim, gen_norms.mean().item(), gen_std, seq_acc,
                gen_pairwise, real_pairwise, n_unique, cfg.batch_size, L_batch,
            )
            wandb.log({
                "eval/cos_sim": cos_sim,
                "eval/mean_dist": mean_dist,
                "eval/min_dist": min_dist,
                "eval/gen_norm": gen_norms.mean().item(),
                "eval/gen_std": gen_std,
                "eval/real_std": real_std,
                "eval/seq_acc": seq_acc,
                "eval/gen_pairwise_l2": gen_pairwise,
                "eval/real_pairwise_l2": real_pairwise,
                "eval/unique_nearest": n_unique,
                "eval/L_batch": L_batch,
            })

            if cfg.tau_recal_every > 0 and epoch % cfg.tau_recal_every == 0:
                taus = list(adaptive_taus(gen_residues.detach(), multipliers=(0.5, 1.0, 2.0)))
                log.info("  [tau recal @ %d]  %s", epoch, [f"{t:.3f}" for t in taus])
                wandb.log({"train/tau_base": taus[1], "train/epoch": epoch})

            if loss.item() < best_loss:
                best_loss = loss.item()
                _save(epoch, best_loss, "best.pt")

    _save(cfg.epochs, loss.item(), "final.pt")
    log.info("Training complete. Best loss: %.6f", best_loss)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. All fields mirror Config with default=None
    so we can detect which were explicitly provided (vs. coming from the config file)."""
    parser = argparse.ArgumentParser(
        description="Train ESM-Drift",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file. CLI args override it.")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir", type=str, default=None)

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--d_noise", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_bottleneck", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Layers per half of the U-Net (enc=dec=num_layers).")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--use_length_cond", action=argparse.BooleanOptionalAction, default=None,
                        help="Length conditioning in the generator. --no_use_length_cond to disable.")

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--eta_min", type=float, default=None)
    parser.add_argument("--eval_every", type=int, default=None)

    # ── LR schedule ───────────────────────────────────────────────────────────
    parser.add_argument("--warmup_T0", type=int, default=None,
                        help="Restart period (epochs). 0 = plain cosine.")
    parser.add_argument("--warmup_T_mult", type=int, default=None,
                        help="Cycle multiplier. 1 = fixed period, 2 = doubling.")

    # ── Kernel ────────────────────────────────────────────────────────────────
    parser.add_argument("--taus", type=float, nargs="+", default=None)
    parser.add_argument("--tau_recal_every", type=int, default=None)

    # ── Loss weights / ablations ───────────────────────────────────────────────
    parser.add_argument("--seq_ce_weight", type=float, default=None)
    parser.add_argument("--prot_drift_weight", type=float, default=None,
                        help="Weight for protein-level drifting loss. 0.0 = disabled.")
    parser.add_argument("--use_strict_antisymmetry", action=argparse.BooleanOptionalAction, default=None,
                        help="Enforce N_pos==N_neg in drifting kernel.")
    parser.add_argument("--use_dynamic_len", action=argparse.BooleanOptionalAction, default=None,
                        help="Generate at batch max length instead of global max_len.")

    # ── Infrastructure ────────────────────────────────────────────────────────
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load base config from file, then apply any explicit CLI overrides
    cfg = Config.from_yaml(args.config) if args.config else Config()
    for key, val in vars(args).items():
        if key in ("config", "no_wandb") or val is None:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    # Resolve device default
    if cfg.device == "cuda" and not torch.cuda.is_available():
        cfg.device = "cpu"
    device = torch.device(cfg.device)

    # max_len alias: train.py historically used --max_seq_len; Config uses max_len
    if wandb is not None:
        if args.no_wandb:
            wandb.init(mode="disabled")
        else:
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg.to_dict())

    # Save config snapshot alongside the checkpoints
    cfg.save_yaml(Path(cfg.save_dir) / "config.yaml")
    log.info("Config saved to %s/config.yaml", cfg.save_dir)
    log.info("Config: %s", cfg.to_dict())

    dataset = EmbeddingDataset(cfg.data_dir, max_seq_len=cfg.max_len)
    log.info("Dataset: %d samples, max_len=%d", len(dataset), cfg.max_len)
    if len(dataset) == 0:
        log.error("No samples found in %s", cfg.data_dir)
        return

    real_residues, real_s_s, real_aa, real_mask, real_means, real_seq_lens = \
        load_real_residues(dataset, cfg.max_len, device)

    generator = DriftingGeneratorUNet(
        d_noise=cfg.d_noise,
        d_model=cfg.d_model,
        d_bottleneck=cfg.d_bottleneck,
        nhead=cfg.nhead,
        enc_layers=cfg.num_layers,
        dec_layers=cfg.num_layers,
        s_s_dim=1024,
        max_len=cfg.max_len,
        use_length_cond=cfg.use_length_cond,
    ).to(device)

    n_params = sum(p.numel() for p in generator.parameters())
    log.info("Generator: %d params (%.1fM)", n_params, n_params / 1e6)

    train(
        generator, real_residues, real_s_s, real_aa, real_mask, real_means, real_seq_lens,
        cfg=cfg, device=device,
    )
    wandb.finish()


if __name__ == "__main__":
    main()

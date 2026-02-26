#!/usr/bin/env python3
"""Training script for ESM-Drift (per-residue drifting, no feature encoder).

Trains a DriftingGeneratorUNet that maps noise → ESMFold s_s / CHEAP z embeddings.

Single-GPU:
    uv run python scripts/train.py --config configs/v14-cheap.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 scripts/train.py --config configs/v15-cheap64.yaml

Override specific fields:
    torchrun --nproc_per_node=4 scripts/train.py --config configs/v15.yaml --lr 1e-3
"""

import argparse
import logging
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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


def setup_ddp() -> tuple[int, int, bool]:
    """Initialize DDP if running under torchrun. Returns (local_rank, world_size, is_main)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, True

    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    is_main = (local_rank == 0)
    return local_rank, world_size, is_main


def load_real_residues(
    dataset: EmbeddingDataset, max_len: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load all real data into GPU memory.

    Returns:
        real_residues: [n_total_valid, s_s_dim] flattened valid residues
        real_s_s:      [N, max_len, s_s_dim] padded protein embeddings
        real_aa:       [N, max_len] LongTensor amino acid indices (0-19, 20=unk)
        real_mask:     [N, max_len] boolean mask
        real_means:    [N, s_s_dim] per-protein mean-pooled embeddings
        real_seq_lens: [N] actual sequence lengths (for dynamic batching)
    """
    collate = partial(pad_collate, max_len=max_len)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    real_s_s = batch["s_s"].to(device)
    real_mask = batch["mask"].to(device)
    real_aa = batch["aa_indices"].to(device)
    real_seq_lens = batch["seq_lens"].to(device)

    real_residues = real_s_s[real_mask]  # [n_valid, s_s_dim]
    real_means = masked_mean_pool(real_s_s, real_mask)  # [N, s_s_dim]

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
    generator: nn.Module,
    real_residues: torch.Tensor,
    real_s_s: torch.Tensor,
    real_aa: torch.Tensor,
    real_mask: torch.Tensor,
    real_means: torch.Tensor,
    real_seq_lens: torch.Tensor,
    cfg: Config,
    device: torch.device,
    is_main: bool,
):
    """Train the drifting generator. All hyperparameters come from cfg."""
    save_path = Path(cfg.save_dir)
    if is_main:
        save_path.mkdir(parents=True, exist_ok=True)

    # Unwrapped generator for attribute access and saving
    raw_gen = generator.module if isinstance(generator, DDP) else generator

    n_proteins = real_s_s.shape[0]
    real_mean_norm = real_residues.norm(dim=-1).mean()

    taus = list(cfg.taus) if cfg.taus else adaptive_taus(real_residues[:2000], multipliers=(0.5, 1.0, 2.0))

    if is_main:
        log.info(
            "Training: %d epochs, batch=%d, lr=%s, taus=%s",
            cfg.epochs, cfg.batch_size, cfg.lr, [f"{t:.1f}" for t in taus],
        )
        log.info("  warmup_T0=%d  T_mult=%d  prot_drift_weight=%.2f",
                 cfg.warmup_T0, cfg.warmup_T_mult, cfg.prot_drift_weight)
        log.info("  use_length_cond=%s  use_strict_antisymmetry=%s  use_dynamic_len=%s",
                 raw_gen.use_length_cond, cfg.use_strict_antisymmetry, cfg.use_dynamic_len)

    seq_head = SeqHead(s_s_dim=cfg.s_s_dim).to(device)
    generator.train()
    seq_head.train()

    optimizer = torch.optim.AdamW(
        list(raw_gen.parameters()) + list(seq_head.parameters()),
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
            "generator": raw_gen.state_dict(),
            "seq_head": seq_head.state_dict(),
            "epoch": epoch,
            "loss": loss_val,
            "config": cfg.to_dict(),
        }, save_path / filename)

    for epoch in range(1, cfg.epochs + 1):
        # Each rank independently samples a different random batch — gradients are averaged via DDP
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

        noise     = raw_gen.sample_noise(cfg.batch_size, L_batch, device)
        z_protein = torch.randn(cfg.batch_size, 1, raw_gen.d_noise, device=device)
        noise     = noise + z_protein
        gen_s_s   = generator(noise, gen_mask, lengths=prot_lens)  # [B, L_batch, s_s_dim]

        gen_residues = gen_s_s.reshape(-1, cfg.s_s_dim)  # [B*L_batch, s_s_dim]
        prot_s_s     = real_s_s[prot_idx, :L_batch, :]   # [B, L_batch, s_s_dim]
        prot_mask    = real_mask[prot_idx, :L_batch]      # [B, L_batch]
        pos_residues = prot_s_s[prot_mask]                # [n_valid, s_s_dim]

        # Strict anti-symmetry: paper requires N_pos == N_neg
        if cfg.use_strict_antisymmetry:
            n_pos, n_gen = pos_residues.shape[0], gen_residues.shape[0]
            if n_gen > n_pos:
                gen_for_drift = gen_residues[torch.randperm(n_gen, device=device)[:n_pos]]
            else:
                gen_for_drift = gen_residues
        else:
            gen_for_drift = gen_residues

        # Cap residues fed to cdist to prevent OOM at large batch×len
        if cfg.max_drift_residues > 0:
            n_cap = cfg.max_drift_residues
            if gen_for_drift.shape[0] > n_cap:
                gen_for_drift = gen_for_drift[torch.randperm(gen_for_drift.shape[0], device=device)[:n_cap]]
            if pos_residues.shape[0] > n_cap:
                pos_residues = pos_residues[torch.randperm(pos_residues.shape[0], device=device)[:n_cap]]

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
            # Detach gen_s_s: seq_head trains on generated embeddings but noise gradient
            # from gen_ce does NOT flow back into the generator (gen_ce ≈ 3.0 = random).
            gen_ce_loss  = F.cross_entropy(seq_head(gen_s_s.detach())[known_aa], batch_aa[known_aa])
            real_ce_loss = F.cross_entropy(seq_head(prot_s_s)[known_aa], batch_aa[known_aa])
        else:
            gen_ce_loss = real_ce_loss = torch.tensor(0.0, device=device)

        loss = (drift_loss
                + cfg.prot_drift_weight * prot_drift_loss
                + cfg.norm_loss_weight * norm_loss
                + cfg.seq_ce_weight * gen_ce_loss
                + 0.5 * cfg.seq_ce_weight * real_ce_loss)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            list(raw_gen.parameters()) + list(seq_head.parameters()),
            max_norm=cfg.max_grad_norm,
        )
        optimizer.step()
        scheduler.step()

        if is_main:
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

        if is_main and epoch % cfg.eval_every == 0:
            with torch.no_grad():
                cross = torch.cdist(gen_residues[:200], real_residues[:200])
                mean_dist = cross.mean().item()
                min_dist  = cross.min(dim=1).values.mean().item()

                gen_normed  = F.normalize(gen_residues[:200], dim=-1)
                real_normed = F.normalize(real_residues[:200], dim=-1)
                cos_sim     = (gen_normed @ real_normed.T).mean().item()
                gen_std     = gen_residues.std(dim=0).mean().item()
                real_std    = real_residues.std(dim=0).mean().item()

                gen_means      = masked_mean_pool(gen_s_s, gen_mask)  # [B, s_s_dim]
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
                "grad=%.4f  cos_sim=%.4f  gen_norm=%.3f  gen_std=%.4f  seq_acc=%.3f  "
                "prot_L2=%.3f (real=%.3f)  unique=%d/%d  L=%d",
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
                # Use a fixed slice of real residues for consistent tau calibration across ranks
                taus = list(adaptive_taus(real_residues[:2000], multipliers=(0.5, 1.0, 2.0)))
                log.info("  [tau recal @ %d]  %s", epoch, [f"{t:.3f}" for t in taus])
                wandb.log({"train/tau_base": taus[1], "train/epoch": epoch})

            if loss.item() < best_loss:
                best_loss = loss.item()
                _save(epoch, best_loss, "best.pt")

    if is_main:
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
    parser.add_argument("--min_ptm", type=float, default=None)

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
    parser.add_argument("--norm_loss_weight", type=float, default=None)
    parser.add_argument("--max_drift_residues", type=int, default=None)
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
    local_rank, world_size, is_main = setup_ddp()

    parser = build_parser()
    args = parser.parse_args()

    # Load base config from file, then apply any explicit CLI overrides
    cfg = Config.from_yaml(args.config) if args.config else Config()
    for key, val in vars(args).items():
        if key in ("config", "no_wandb") or val is None:
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    # Resolve device: torchrun sets LOCAL_RANK, use that GPU; otherwise use cfg.device
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        if cfg.device == "cuda" and not torch.cuda.is_available():
            cfg.device = "cpu"
        device = torch.device(cfg.device)

    if is_main:
        if args.no_wandb:
            wandb.init(mode="disabled")
        else:
            wandb.init(project=cfg.wandb_project, name=cfg.wandb_name, config=cfg.to_dict())

        # Save config snapshot alongside the checkpoints
        cfg.save_yaml(Path(cfg.save_dir) / "config.yaml")
        log.info("Config saved to %s/config.yaml", cfg.save_dir)
        log.info("Config: %s", cfg.to_dict())

    dataset = EmbeddingDataset(cfg.data_dir, max_seq_len=cfg.max_len, min_ptm=cfg.min_ptm)
    if is_main:
        log.info("Dataset: %d samples, max_len=%d, min_ptm=%.2f",
                 len(dataset), cfg.max_len, cfg.min_ptm)
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
        s_s_dim=cfg.s_s_dim,
        max_len=cfg.max_len,
        use_length_cond=cfg.use_length_cond,
        output_scale_init=cfg.output_scale_init,
        protein_cond_std=cfg.protein_cond_std,
    ).to(device)

    if world_size > 1:
        generator = DDP(generator, device_ids=[local_rank])

    if is_main:
        raw_gen = generator.module if isinstance(generator, DDP) else generator
        n_params = sum(p.numel() for p in raw_gen.parameters())
        log.info("Generator: %d params (%.1fM)", n_params, n_params / 1e6)
        log.info("World size: %d GPU(s)", world_size)

    train(
        generator, real_residues, real_s_s, real_aa, real_mask, real_means, real_seq_lens,
        cfg=cfg, device=device, is_main=is_main,
    )

    if is_main:
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

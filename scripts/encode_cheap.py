#!/usr/bin/env python3
"""Encode ESMFold s_s [L, 1024] → CHEAP z [L, dim].

Reads existing .pt files from --input_dir, compresses each protein to a
CHEAP latent z [L, dim], and saves to --output_dir.

Multi-GPU (fastest):
    torchrun --nproc_per_node=4 scripts/encode_cheap.py \\
        --input_dir data/ --output_dir data/cheap64/ --s_s_dim 64 --min_ptm 0.7

Single-GPU:
    uv run python scripts/encode_cheap.py \\
        --input_dir data/ --output_dir data/cheap64/ --s_s_dim 64 --min_ptm 0.7

Calibration (prints output_scale_init / protein_cond_std for training config):
    uv run python scripts/encode_cheap.py --input_dir data/cheap64/ --s_s_dim 64 --calibrate

Reconstruction test:
    uv run python scripts/encode_cheap.py --input_dir data/ --s_s_dim 64 --test
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, bool]:
    """Init DDP if launched under torchrun. Returns (local_rank, world_size, is_main)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, True
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_world_size(), local_rank == 0


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_channel_stats(input_dir: Path, max_files: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel min/max across s_s embeddings for normalisation."""
    files = sorted(input_dir.glob("*.pt"))[:max_files]
    log.info("Computing per-channel stats from %d files...", len(files))
    all_s_s = []
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
            if "s_s" in d:
                all_s_s.append(d["s_s"])
        except Exception:
            pass
    if not all_s_s:
        raise RuntimeError(f"No s_s embeddings found in {input_dir}")
    residues = torch.cat(all_s_s, dim=0)  # [N_total, 1024]
    chan_min = residues.min(dim=0).values
    chan_max = residues.max(dim=0).values
    log.info("  Stats: %d residues, dim=%d", residues.shape[0], residues.shape[1])
    return chan_min, chan_max


def scale(s_s: torch.Tensor, chan_min: torch.Tensor, chan_max: torch.Tensor) -> torch.Tensor:
    chan_min = chan_min.to(s_s.device)
    chan_max = chan_max.to(s_s.device)
    return 2.0 * (s_s - chan_min) / (chan_max - chan_min).clamp_min(1e-6) - 1.0


def unscale(z_norm: torch.Tensor, chan_min: torch.Tensor, chan_max: torch.Tensor) -> torch.Tensor:
    chan_min = chan_min.to(z_norm.device)
    chan_max = chan_max.to(z_norm.device)
    return (z_norm + 1.0) / 2.0 * (chan_max - chan_min).clamp_min(1e-6) + chan_min


# ---------------------------------------------------------------------------
# CHEAP model
# ---------------------------------------------------------------------------

def load_cheap_model(device: torch.device, s_s_dim: int = 32):
    from cheap.pretrained import load_model_from_id
    from cheap.constants import CATH_COMPRESS_LEVEL_TO_ID
    model_id = CATH_COMPRESS_LEVEL_TO_ID[1][s_s_dim]
    log.info("Loading CHEAP shorten_1_dim_%d (id=%s) on %s...", s_s_dim, model_id, device)
    return load_model_from_id(model_id, infer_mode=True).to(device)


# ---------------------------------------------------------------------------
# I/O worker (runs in background thread)
# ---------------------------------------------------------------------------

def load_file_batch(
    files: list[Path],
    min_ptm: float,
    max_len: int,
) -> list[tuple[Path, dict]]:
    """Load and filter a batch of .pt files. Called from a background thread."""
    items = []
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception as e:
            log.warning("Could not load %s: %s", f.name, e)
            continue
        if "s_s" not in d:
            continue
        L = d["s_s"].shape[0]
        if L > max_len:
            continue
        if min_ptm > 0.0 and float(d.get("ptm", 1.0)) < min_ptm:
            continue
        items.append((f, d))
    return items


# ---------------------------------------------------------------------------
# Batched GPU encoder
# ---------------------------------------------------------------------------

def encode_and_save_batch(
    items: list[tuple[Path, dict]],
    model,
    chan_min: torch.Tensor,
    chan_max: torch.Tensor,
    device: torch.device,
    output_dir: Path,
) -> int:
    """Encode a batch of proteins with a single GPU forward pass and save results."""
    if not items:
        return 0

    lens = [d["s_s"].shape[0] for _, d in items]
    L_max = max(lens)
    B = len(items)

    s_s_batch = torch.zeros(B, L_max, items[0][1]["s_s"].shape[-1])
    mask_batch = torch.zeros(B, L_max, dtype=torch.bool)
    for i, (_, d) in enumerate(items):
        L = lens[i]
        s_s_batch[i, :L] = d["s_s"]
        mask_batch[i, :L] = True

    s_s_batch = s_s_batch.to(device)
    mask_batch = mask_batch.to(device)
    s_s_scaled = scale(s_s_batch, chan_min, chan_max)

    with torch.no_grad():
        z_batch, _ = model.enc(s_s_scaled, mask_batch)  # [B, L_max, z_dim]

    z_batch = z_batch.cpu()

    for i, (f, d) in enumerate(items):
        L = lens[i]
        out = {
            "z": z_batch[i, :L],  # [L, z_dim]
            "seq_len": d.get("seq_len", L),
            "sequence": d.get("sequence"),
            "ptm": d.get("ptm"),
            "plddt": d.get("plddt"),
            "aa_indices": d.get("aa_indices"),
            "mask": d.get("mask", torch.ones(L, dtype=torch.bool)),
        }
        torch.save({k: v for k, v in out.items() if v is not None}, output_dir / f.name)

    return len(items)


# ---------------------------------------------------------------------------
# Main encoding loop with prefetch
# ---------------------------------------------------------------------------

def run_encoding(
    my_files: list[Path],
    model,
    chan_min: torch.Tensor,
    chan_max: torch.Tensor,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    num_workers: int,
    min_ptm: float,
    max_len: int,
    is_main: bool,
) -> tuple[int, int]:
    """Encode all files assigned to this rank, with threaded prefetch."""
    # Skip already-encoded files
    todo = [f for f in my_files if not (output_dir / f.name).exists()]
    n_already_done = len(my_files) - len(todo)

    if not todo:
        return n_already_done, 0

    # Split into batches
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    n_ok = n_already_done
    n_skip = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Prime the pipeline: start loading the first batch in a background thread
        pending = pool.submit(load_file_batch, batches[0], min_ptm, max_len)

        for batch_i in range(len(batches)):
            # While GPU is about to process current batch, start loading the next
            if batch_i + 1 < len(batches):
                next_pending = pool.submit(load_file_batch, batches[batch_i + 1], min_ptm, max_len)

            items = pending.result()
            n_skip += len(batches[batch_i]) - len(items)
            n_ok += encode_and_save_batch(items, model, chan_min, chan_max, device, output_dir)

            if batch_i + 1 < len(batches):
                pending = next_pending

            if is_main and (batch_i + 1) % 10 == 0:
                pct = 100 * (batch_i + 1) / len(batches)
                log.info("  %.1f%%  %d encoded  %d skipped  (batch %d/%d)",
                         pct, n_ok, n_skip, batch_i + 1, len(batches))

    return n_ok, n_skip


# ---------------------------------------------------------------------------
# Calibration / test utilities
# ---------------------------------------------------------------------------

def print_calibration(input_dir: Path, model, chan_min, chan_max, device, s_s_dim: int):
    """Encode a sample of proteins and print suggested training calibration values."""
    import math
    files = sorted(input_dir.glob("*.pt"))[:100]
    norms, prot_l2s = [], []
    n = 0
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if "s_s" not in d:
            continue
        s_s = d["s_s"].unsqueeze(0).to(device)
        L = s_s.shape[1]
        mask = torch.ones(1, L, dtype=torch.bool, device=device)
        with torch.no_grad():
            z, _ = model.enc(scale(s_s, chan_min, chan_max), mask)
        z = z.squeeze(0)
        norms.extend(z.norm(dim=-1).tolist())
        prot_l2s.append(z.mean(dim=0).norm().item())
        n += 1

    real_norm = sum(norms) / len(norms)
    real_prot_l2 = sum(prot_l2s) / len(prot_l2s)
    # Empirically: gen_norm at output_scale=10 scales as sqrt(dim/32) from 32D baseline (~57)
    gen_norm_est = 57.0 * math.sqrt(s_s_dim / 32)
    output_scale_init = round(10.0 * real_norm / gen_norm_est, 4)
    protein_cond_std = round(0.4 * real_prot_l2 / 65.0, 5)

    log.info("=== CALIBRATION (from %d files) ===", n)
    log.info("  real z residue norm:  mean=%.4f", real_norm)
    log.info("  real prot_L2:         mean=%.4f", real_prot_l2)
    log.info("  Suggested output_scale_init: %.4f", output_scale_init)
    log.info("  Suggested protein_cond_std:  %.5f", protein_cond_std)


def run_test(input_dir: Path, model, chan_min, chan_max, device):
    """Encode + decode a few proteins and print pLDDT comparison."""
    from esm_drift.utils import StructureDecoder
    log.info("Running reconstruction quality test...")
    decoder = StructureDecoder(device=str(device))
    decoder._load_model()

    files = sorted(input_dir.glob("*.pt"))[10:16]
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if "s_s" not in d:
            continue
        s_s = d["s_s"].unsqueeze(0).to(device)
        L = s_s.shape[1]
        mask_t = torch.ones(1, L, dtype=torch.bool, device=device)
        seq = d.get("sequence")

        plddt_orig = decoder.decode(s_s.squeeze(0), sequence=seq)["plddt"].mean().item()

        s_s_scaled = scale(s_s, chan_min, chan_max)
        with torch.no_grad():
            z, _ = model.enc(s_s_scaled, mask_t)
            s_s_recon = unscale(model.dec(z, mask_t), chan_min, chan_max)

        plddt_recon = decoder.decode(s_s_recon.squeeze(0), sequence=seq)["plddt"].mean().item()
        print(f"  {f.name}: L={L}  orig={plddt_orig:.3f}  recon={plddt_recon:.3f}  Δ={plddt_recon - plddt_orig:+.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    local_rank, world_size, is_main = setup_ddp()

    parser = argparse.ArgumentParser(description="Encode ESMFold s_s → CHEAP z (multi-GPU + batched)")
    parser.add_argument("--input_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="data/cheap/")
    parser.add_argument("--s_s_dim", type=int, default=32, choices=[4, 8, 16, 32, 64, 128, 256, 512],
                        help="CHEAP latent dimension (shorten_1_dim_<s_s_dim>)")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Skip proteins longer than this")
    parser.add_argument("--min_ptm", type=float, default=0.0,
                        help="Skip proteins with pTM below this threshold")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Proteins per GPU forward pass (increase for throughput)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Background threads for file I/O prefetching")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test", action="store_true",
                        help="Run reconstruction quality test only")
    parser.add_argument("--calibrate", action="store_true",
                        help="Print suggested output_scale_init / protein_cond_std for training config")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Each rank uses its own GPU
    if world_size > 1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Stats computation: rank 0 only, then barrier so all ranks can load ---
    stats_path = output_dir / "scaler_stats.pt"

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        chan_min, chan_max = compute_channel_stats(input_dir)
        torch.save({"chan_min": chan_min, "chan_max": chan_max}, stats_path)
        log.info("Saved normalisation stats to %s", stats_path)

    if world_size > 1:
        dist.barrier()  # wait for rank 0 to write stats

    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    chan_min, chan_max = stats["chan_min"], stats["chan_max"]

    model = load_cheap_model(device, s_s_dim=args.s_s_dim)

    if args.calibrate:
        if is_main:
            print_calibration(input_dir, model, chan_min, chan_max, device, s_s_dim=args.s_s_dim)
        if world_size > 1:
            dist.destroy_process_group()
        return

    if args.test:
        if is_main:
            run_test(input_dir, model, chan_min, chan_max, device)
        if world_size > 1:
            dist.destroy_process_group()
        return

    # --- Shard files across GPUs ---
    all_files = sorted(f for f in input_dir.glob("*.pt") if f.name != "scaler_stats.pt")
    my_files = all_files[local_rank::world_size]

    if is_main:
        log.info("Encoding: %d total files → %d per rank (world_size=%d)",
                 len(all_files), len(my_files), world_size)
        log.info("  s_s_dim=%d  min_ptm=%.2f  max_len=%d  batch_size=%d  num_workers=%d",
                 args.s_s_dim, args.min_ptm, args.max_len, args.batch_size, args.num_workers)
    else:
        log.info("[rank %d] %d files assigned", local_rank, len(my_files))

    n_ok, n_skip = run_encoding(
        my_files, model, chan_min, chan_max, device, output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_ptm=args.min_ptm,
        max_len=args.max_len,
        is_main=is_main,
    )

    log.info("[rank %d] Done: %d encoded, %d skipped", local_rank, n_ok, n_skip)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

    if is_main:
        total = sum(1 for f in output_dir.glob("*.pt") if f.name != "scaler_stats.pt")
        log.info("Output dir total: %d files in %s", total, output_dir)


if __name__ == "__main__":
    main()

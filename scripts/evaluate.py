#!/usr/bin/env python3
"""Evaluate a trained ESM-Drift generator.

Generates s_s embeddings from noise, decodes through ESMFold's structure module,
decodes the nearest real embeddings as ground truth, and computes TM-scores.

Usage:
    uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --n_samples 8 --skip_decode
    uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --n_samples 4 --save_pdbs output/
"""

import argparse
import logging
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from esm_drift.data.dataset import EmbeddingDataset, pad_collate
from esm_drift.model import DriftingGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def mean_pool(s_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (s_s * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).float().clamp(min=1)


def load_checkpoint(path: str, device: torch.device) -> tuple[DriftingGenerator, dict, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    generator = DriftingGenerator(
        d_noise=256, d_model=256, nhead=8, num_layers=4,
        s_s_dim=1024, max_len=config["max_len"],
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    norm_stats = {
        "embed_mean": ckpt.get("embed_mean"),
        "embed_std": ckpt.get("embed_std"),
    }
    return generator, config, norm_stats


@torch.no_grad()
def generate_embeddings(generator, n_samples, seq_len, device):
    noise = generator.sample_noise(n_samples, seq_len, device)
    mask = torch.ones(n_samples, seq_len, dtype=torch.bool, device=device)
    return generator(noise, mask)


def extract_ca_coords(structure: dict) -> np.ndarray:
    """Extract Cα coordinates from decoded structure.

    Args:
        structure: output from StructureDecoder.decode()

    Returns:
        [L, 3] numpy array of Cα positions
    """
    # positions shape: [n_recycles, B, L, 14, 3] — atom14 format
    # Cα is atom index 1 in atom14
    pos = structure["positions"]
    if pos.dim() == 5:
        pos = pos[-1, 0]  # last recycle, first (only) batch element → [L, 14, 3]
    elif pos.dim() == 4:
        pos = pos[0]  # [L, 14, 3]
    ca = pos[:, 1, :].cpu().numpy().astype(np.float64)  # atom index 1 = Cα
    return ca


def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray, seq1: str, seq2: str) -> float:
    """Compute TM-score between two structures.

    Returns TM-score normalized by the length of the second structure (target).
    """
    from tmtools import tm_align
    result = tm_align(coords1, coords2, seq1, seq2)
    return result.tm_norm_chain2


def decode_and_evaluate(
    gen_s_s: torch.Tensor,
    real_data: list[dict],
    nearest_indices: list[int],
    device: torch.device,
    save_dir: str | None = None,
) -> dict:
    """Decode generated + real embeddings through ESMFold and compute TM-scores.

    For each generated sample, we decode its nearest real embedding as ground
    truth and compute TM-score between them.
    """
    from esm_drift.utils import StructureDecoder
    decoder = StructureDecoder(device=str(device))

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        Path(save_dir, "real").mkdir(exist_ok=True)

    gen_plddts, gen_ptms = [], []
    real_plddts, real_ptms = [], []
    tm_scores = []

    n = gen_s_s.shape[0]
    seq_len = gen_s_s.shape[1]

    for i in range(n):
        # --- Decode generated sample ---
        try:
            gen_struct = decoder.decode(gen_s_s[i], s_z=None, sequence=None)
            gen_plddt = gen_struct["plddt"].mean().item()
            gen_ptm = gen_struct["ptm"].item()
            gen_plddts.append(gen_plddt)
            gen_ptms.append(gen_ptm)
            gen_ca = extract_ca_coords(gen_struct)
            gen_seq = "G" * seq_len  # poly-glycine (no sequence info)

            if save_dir:
                pdb_str = decoder.to_pdb(gen_struct)
                Path(save_dir, f"gen_{i:03d}.pdb").write_text(pdb_str)
        except Exception as e:
            log.warning("  Gen sample %d: decode failed: %s", i, e)
            gen_plddts.append(0.0)
            gen_ptms.append(0.0)
            tm_scores.append(0.0)
            continue

        # --- Decode nearest real sample ---
        ri = nearest_indices[i]
        real = real_data[ri]
        real_seq_len = real["seq_len"]
        # Use poly-glycine for sequences that are all X (DNA structures etc.)
        real_sequence = real.get("sequence")
        if real_sequence and set(real_sequence) == {"X"}:
            real_sequence = None
        try:
            real_struct = decoder.decode(
                real["s_s"], s_z=real.get("s_z"), sequence=real_sequence,
            )
            real_plddt = real_struct["plddt"].mean().item()
            real_ptm = real_struct["ptm"].item()
            real_plddts.append(real_plddt)
            real_ptms.append(real_ptm)
            real_ca = extract_ca_coords(real_struct)[:real_seq_len]
            real_seq = (real_sequence or "G" * real_seq_len)[:real_seq_len]

            if save_dir:
                pdb_str = decoder.to_pdb(real_struct)
                Path(save_dir, "real", f"real_{i:03d}_nearest{ri}.pdb").write_text(pdb_str)
        except Exception as e:
            log.warning("  Real sample %d (idx=%d): decode failed: %s", i, ri, e)
            real_plddts.append(0.0)
            real_ptms.append(0.0)
            tm_scores.append(0.0)
            continue

        # --- TM-score ---
        # Truncate generated coords to real length for fair comparison
        gen_ca_trunc = gen_ca[:real_seq_len]
        gen_seq_trunc = gen_seq[:real_seq_len]
        try:
            tm = compute_tm_score(gen_ca_trunc, real_ca, gen_seq_trunc, real_seq)
            tm_scores.append(tm)
        except Exception as e:
            log.warning("  TM-score failed for sample %d: %s", i, e)
            tm_scores.append(0.0)

        log.info(
            "  Sample %d: gen(pLDDT=%.2f, pTM=%.4f) vs real_%d(pLDDT=%.2f, pTM=%.4f) → TM=%.4f",
            i, gen_plddt, gen_ptm, ri, real_plddt, real_ptm, tm_scores[-1],
        )

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "gen_plddt_mean": safe_mean(gen_plddts),
        "gen_plddt_max": max(gen_plddts) if gen_plddts else 0,
        "gen_ptm_mean": safe_mean(gen_ptms),
        "real_plddt_mean": safe_mean(real_plddts),
        "real_ptm_mean": safe_mean(real_ptms),
        "tm_score_mean": safe_mean(tm_scores),
        "tm_score_max": max(tm_scores) if tm_scores else 0,
        "tm_score_min": min(tm_scores) if tm_scores else 0,
        "n_samples": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ESM-Drift generator")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_pdbs", type=str, default=None)
    parser.add_argument("--skip_decode", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator, config, norm_stats = load_checkpoint(args.checkpoint, device)
    max_len = config["max_len"]
    log.info("Loaded checkpoint: epoch=%s, max_len=%d", config.get("epoch"), max_len)

    # Generate (in standardized space)
    gen_s_s = generate_embeddings(generator, args.n_samples, max_len, device)
    log.info("Generated %d samples, shape=%s (standardized)", args.n_samples, list(gen_s_s.shape))

    # Un-standardize if normalization stats are available
    if norm_stats["embed_mean"] is not None:
        embed_mean = norm_stats["embed_mean"].to(device)
        embed_std = norm_stats["embed_std"].to(device)
        gen_s_s = gen_s_s * embed_std + embed_mean
        log.info("Un-standardized generated embeddings to raw ESMFold space")

    # Load real data
    dataset = EmbeddingDataset(args.data_dir, max_seq_len=max_len)
    if len(dataset) == 0:
        log.error("No real data found")
        return

    # Feature-space metrics (per-residue)
    collate = partial(pad_collate, max_len=max_len)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    real_s_s = batch["s_s"].to(device)
    real_mask = batch["mask"].to(device)
    gen_mask = torch.ones(args.n_samples, max_len, dtype=torch.bool, device=device)

    with torch.no_grad():
        # Per-residue metrics
        gen_residues = gen_s_s.reshape(-1, 1024)  # [n_samples * L, 1024]
        real_residues = real_s_s[real_mask]  # [n_real_valid, 1024]

        gen_normed = torch.nn.functional.normalize(gen_residues, dim=-1)
        real_normed = torch.nn.functional.normalize(real_residues, dim=-1)

        # Subsample for memory (cosine sim matrix)
        n_sub = min(500, gen_residues.shape[0], real_residues.shape[0])
        gi = torch.randperm(gen_residues.shape[0], device=device)[:n_sub]
        ri = torch.randperm(real_residues.shape[0], device=device)[:n_sub]
        cos_cross = gen_normed[gi] @ real_normed[ri].T  # [n_sub, n_sub]

        log.info("Per-residue feature metrics:")
        log.info("  cos_sim (gen↔real): mean=%.4f, max=%.4f", cos_cross.mean().item(), cos_cross.max().item())
        log.info("  gen residue norm: mean=%.2f, std=%.2f",
                 gen_residues.norm(dim=-1).mean().item(), gen_residues.norm(dim=-1).std().item())
        log.info("  real residue norm: mean=%.2f, std=%.2f",
                 real_residues.norm(dim=-1).mean().item(), real_residues.norm(dim=-1).std().item())

        # Per-residue diversity (std across feature dims)
        gen_std = gen_residues.std(dim=0).mean().item()
        real_std = real_residues.std(dim=0).mean().item()
        log.info("  gen diversity (mean dim std): %.2f", gen_std)
        log.info("  real diversity (mean dim std): %.2f", real_std)

        # Nearest real for each generated protein (mean-pool for matching to decode)
        real_feat = mean_pool(real_s_s, real_mask)
        gen_feat = mean_pool(gen_s_s, gen_mask)
        cross = torch.cdist(gen_feat, real_feat)

    nearest_indices = cross.argmin(dim=1).cpu().tolist()
    log.info("  nearest_real_indices: %s", nearest_indices)
    log.info("  mean_nearest_dist (pooled): %.2f", cross.min(dim=1).values.mean().item())

    # Load raw real data (unpadded, with sequences) for decoding
    real_data = [dataset[i] for i in range(len(dataset))]

    # Decode with ESMFold + TM-score
    if not args.skip_decode:
        log.info("Decoding with ESMFold and computing TM-scores...")
        metrics = decode_and_evaluate(
            gen_s_s, real_data, nearest_indices, device, save_dir=args.save_pdbs,
        )
        log.info("=" * 60)
        log.info("Results:")
        log.info("  Generated: pLDDT=%.4f, pTM=%.4f", metrics["gen_plddt_mean"], metrics["gen_ptm_mean"])
        log.info("  Real (nearest): pLDDT=%.4f, pTM=%.4f", metrics["real_plddt_mean"], metrics["real_ptm_mean"])
        log.info("  TM-score (gen vs nearest real): mean=%.4f, max=%.4f, min=%.4f",
                 metrics["tm_score_mean"], metrics["tm_score_max"], metrics["tm_score_min"])
        log.info("=" * 60)
    else:
        log.info("Skipping ESMFold decode (--skip_decode)")


if __name__ == "__main__":
    main()

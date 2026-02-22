#!/usr/bin/env python3
"""Evaluate a trained ESM-Drift generator.

Metrics computed:

  Embedding-space (fast, no ESMFold needed):
    - per-residue norm vs real
    - cosine similarity gen↔real
    - gen diversity (per-dim std, pairwise L2 between proteins)
    - nearest-neighbour uniqueness (mode collapse indicator)

  Structure (needs ESMFold, skipped with --skip_decode):
    - pLDDT: min / median / mean / max / fraction > 0.7
    - pTM
    - Pairwise TM-score between generated structures (diversity)
    - Novelty: TM-score to nearest training protein

Usage:
    uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --skip_decode
    uv run python scripts/evaluate.py --checkpoint checkpoints/best.pt --n_samples 8 --save_pdbs output/
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

from esm_drift.data.dataset import EmbeddingDataset, pad_collate
from esm_drift.data.dataset import _RESTYPES as RESTYPES
from esm_drift.model import DriftingGeneratorUNet, SeqHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str, device: torch.device
) -> tuple[DriftingGeneratorUNet, SeqHead | None, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt["config"]
    _nl = config.get("num_layers", 6)
    _default_half = max(1, _nl // 2)
    generator = DriftingGeneratorUNet(
        d_noise=config.get("d_noise", 256),
        d_model=config.get("d_model", 512),
        d_bottleneck=config.get("d_bottleneck", 128),
        nhead=config.get("nhead", 8),
        enc_layers=config.get("enc_layers", _default_half),
        dec_layers=config.get("dec_layers", _default_half),
        s_s_dim=1024,
        max_len=config["max_len"],
    ).to(device)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    log.info("Loaded DriftingGeneratorUNet from checkpoint (epoch=%s)", config.get("epoch"))

    seq_head = None
    if "seq_head" in ckpt:
        seq_head = SeqHead(s_s_dim=1024).to(device)
        seq_head.load_state_dict(ckpt["seq_head"])
        seq_head.eval()
        log.info("Loaded seq_head from checkpoint.")
    else:
        log.info("No seq_head in checkpoint — will use poly-alanine for decoding.")

    return generator, seq_head, config


# ---------------------------------------------------------------------------
# Generation with variable lengths
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(
    generator: DriftingGeneratorUNet,
    lengths: list[int],
    device: torch.device,
    seq_head: SeqHead | None = None,
) -> tuple[list[torch.Tensor], list[str | None]]:
    """Generate one embedding per length in `lengths`.

    Each sample is generated at max_len and then sliced to its target length.

    Returns:
        gen_samples: list of [L_i, 1024] tensors
        gen_seqs:    list of predicted amino acid strings (or None if no seq_head)
    """
    max_len = generator.pos_enc.pe.shape[1]
    n = len(lengths)
    noise = generator.sample_noise(n, max_len, device)
    z_protein = torch.randn(n, 1, generator.d_noise, device=device)
    noise = noise + z_protein
    mask = torch.ones(n, max_len, dtype=torch.bool, device=device)
    gen_s_s = generator(noise, mask)  # [n, max_len, 1024]

    gen_samples, gen_seqs = [], []
    for i in range(n):
        L = lengths[i]
        s_s_i = gen_s_s[i, :L].cpu()
        gen_samples.append(s_s_i)

        if seq_head is not None:
            logits = seq_head(gen_s_s[i, :L])          # [L, 20]
            aa_idx = logits.argmax(dim=-1).cpu().tolist()
            seq = "".join(RESTYPES[j] if j < 20 else "G" for j in aa_idx)
            gen_seqs.append(seq)
        else:
            gen_seqs.append(None)

    return gen_samples, gen_seqs


def sample_lengths_from_data(dataset: EmbeddingDataset, n: int) -> list[int]:
    """Sample n lengths with replacement from the training set length distribution."""
    real_lengths = [dataset[i]["seq_len"] for i in range(len(dataset))]
    return random.choices(real_lengths, k=n)


# ---------------------------------------------------------------------------
# Embedding-space metrics
# ---------------------------------------------------------------------------

def embedding_metrics(
    gen_samples: list[torch.Tensor],
    real_residues: torch.Tensor,
    real_pooled: torch.Tensor,
    real_mask: torch.Tensor,
    real_s_s: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute embedding-space metrics without ESMFold."""
    gen_residues = torch.cat(gen_samples, dim=0).to(device)  # [n_gen_residues, 1024]
    n_sub = min(500, gen_residues.shape[0], real_residues.shape[0])
    gi = torch.randperm(gen_residues.shape[0])[:n_sub]
    ri = torch.randperm(real_residues.shape[0])[:n_sub]
    gen_sub = gen_residues[gi].to(device)
    real_sub = real_residues[ri].to(device)

    # Norms
    gen_norms = gen_residues.norm(dim=-1)
    real_norms = real_residues.norm(dim=-1)

    # Cosine similarity (per-residue cross)
    cos_cross = F.normalize(gen_sub, dim=-1) @ F.normalize(real_sub, dim=-1).T
    cos_sim_mean = cos_cross.mean().item()
    cos_sim_max = cos_cross.max().item()

    # Diversity: per-dim std of generated residues
    gen_std = gen_residues.std(dim=0).mean().item()
    real_std = real_residues.std(dim=0).mean().item()

    # Protein-level diversity: pairwise L2 between mean-pooled generated proteins
    gen_pooled = torch.stack([s.mean(dim=0) for s in gen_samples]).to(device)  # [n, 1024]
    if gen_pooled.shape[0] > 1:
        pdist = torch.cdist(gen_pooled, gen_pooled)  # [n, n]
        mask_off = ~torch.eye(pdist.shape[0], dtype=torch.bool, device=device)
        gen_pairwise_l2 = pdist[mask_off].mean().item()
        real_pairwise_l2 = torch.cdist(real_pooled[:50], real_pooled[:50])
        real_mask_off = ~torch.eye(min(50, real_pooled.shape[0]), dtype=torch.bool, device=device)
        real_pairwise_l2 = real_pairwise_l2[real_mask_off].mean().item()
    else:
        gen_pairwise_l2 = 0.0
        real_pairwise_l2 = 0.0

    # Nearest training protein for each generated protein (mode collapse indicator)
    cross = torch.cdist(gen_pooled, real_pooled.to(device))  # [n_gen, n_real]
    nearest_idx = cross.argmin(dim=1).cpu().tolist()
    nearest_dists = cross.min(dim=1).values.cpu().tolist()
    n_unique = len(set(nearest_idx))

    log.info("── Embedding-space metrics ──────────────────────────────")
    log.info("  gen  residue norm:  mean=%.1f  std=%.1f", gen_norms.mean().item(), gen_norms.std().item())
    log.info("  real residue norm:  mean=%.1f  std=%.1f", real_norms.mean().item(), real_norms.std().item())
    log.info("  cos_sim gen↔real:   mean=%.4f  max=%.4f", cos_sim_mean, cos_sim_max)
    log.info("  gen  residue std:   %.3f  (real=%.3f)", gen_std, real_std)
    log.info("  gen  protein pairwise L2: %.1f  (real=%.1f)", gen_pairwise_l2, real_pairwise_l2)
    log.info("  nearest real indices:     %s", nearest_idx)
    log.info("  unique nearest:  %d / %d  (1=collapsed, %d=diverse)",
             n_unique, len(gen_samples), len(gen_samples))
    log.info("  mean nearest dist: %.1f", sum(nearest_dists) / len(nearest_dists))

    return {
        "gen_norm_mean": gen_norms.mean().item(),
        "real_norm_mean": real_norms.mean().item(),
        "cos_sim_mean": cos_sim_mean,
        "cos_sim_max": cos_sim_max,
        "gen_residue_std": gen_std,
        "real_residue_std": real_std,
        "gen_pairwise_l2": gen_pairwise_l2,
        "real_pairwise_l2": real_pairwise_l2,
        "nearest_unique": n_unique,
        "nearest_dist_mean": sum(nearest_dists) / len(nearest_dists),
        "nearest_indices": nearest_idx,
    }


# ---------------------------------------------------------------------------
# Structure decoding and metrics
# ---------------------------------------------------------------------------

def extract_ca_coords(structure: dict) -> np.ndarray:
    """Extract Cα coordinates from decoded structure → [L, 3]."""
    pos = structure["positions"]
    if pos.dim() == 5:
        pos = pos[-1, 0]  # last recycle, first batch → [L, 14, 3]
    elif pos.dim() == 4:
        pos = pos[0]
    return pos[:, 1, :].cpu().numpy().astype(np.float64)  # Cα = atom index 1


def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray,
                     seq1: str, seq2: str) -> float:
    from tmtools import tm_align
    result = tm_align(coords1, coords2, seq1, seq2)
    return result.tm_norm_chain2


def decode_and_evaluate(
    gen_samples: list[torch.Tensor],
    gen_seqs: list[str | None],
    lengths: list[int],
    real_data: list[dict],
    nearest_indices: list[int],
    device: torch.device,
    save_dir: str | None = None,
) -> dict:
    """Decode generated embeddings and compute structure metrics.

    Metrics:
      - pLDDT / pTM per sample
      - Pairwise TM-score between all generated samples (diversity)
      - TM-score of each generated sample to its nearest training protein (novelty)
    """
    from esm_drift.utils import StructureDecoder
    decoder = StructureDecoder(device=str(device))

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    gen_plddts, gen_ptms = [], []
    gen_ca_list, gen_seq_list = [], []

    # --- Decode all generated samples ---
    for i, (s_s, L) in enumerate(zip(gen_samples, lengths)):
        # Use seq_head prediction if available; fall back to poly-alanine
        seq = gen_seqs[i] if gen_seqs[i] is not None else "A" * L
        try:
            struct = decoder.decode(s_s[:L], s_z=None, sequence=seq)
            plddt = struct["plddt"].mean().item()
            ptm = struct["ptm"].item()
            gen_plddts.append(plddt)
            gen_ptms.append(ptm)
            gen_ca_list.append(extract_ca_coords(struct)[:L])
            gen_seq_list.append(seq)

            if save_dir:
                pdb_str = decoder.to_pdb(struct)
                Path(save_dir, f"gen_{i:03d}_L{L}_plddt{plddt:.2f}.pdb").write_text(pdb_str)

            log.info("  Gen %d (L=%d): pLDDT=%.3f  pTM=%.3f", i, L, plddt, ptm)
        except Exception as e:
            log.warning("  Gen %d: decode failed: %s", i, e)
            gen_plddts.append(0.0)
            gen_ptms.append(0.0)
            gen_ca_list.append(None)
            gen_seq_list.append(seq)

    # --- pLDDT distribution ---
    valid_plddts = [p for p in gen_plddts if p > 0]
    frac_designable = sum(1 for p in valid_plddts if p > 0.7) / max(len(valid_plddts), 1)
    sorted_plddts = sorted(valid_plddts)
    plddt_median = sorted_plddts[len(sorted_plddts) // 2] if sorted_plddts else 0.0

    log.info("── Structure quality ────────────────────────────────────")
    log.info("  pLDDT:  min=%.3f  median=%.3f  mean=%.3f  max=%.3f",
             min(valid_plddts, default=0), plddt_median,
             sum(valid_plddts) / max(len(valid_plddts), 1), max(valid_plddts, default=0))
    log.info("  pLDDT > 0.7 (designable): %d / %d (%.0f%%)",
             int(frac_designable * len(valid_plddts)), len(valid_plddts),
             100 * frac_designable)
    log.info("  pTM:    mean=%.3f", sum(gen_ptms) / max(len(gen_ptms), 1))

    # --- Pairwise TM-score between generated (diversity) ---
    valid_pairs = [(i, j) for i in range(len(gen_ca_list))
                   for j in range(i + 1, len(gen_ca_list))
                   if gen_ca_list[i] is not None and gen_ca_list[j] is not None]
    pairwise_tms = []
    for i, j in valid_pairs:
        try:
            tm = compute_tm_score(gen_ca_list[i], gen_ca_list[j],
                                  gen_seq_list[i], gen_seq_list[j])
            pairwise_tms.append(tm)
        except Exception:
            pass

    if pairwise_tms:
        log.info("── Structural diversity (pairwise TM among generated) ───")
        log.info("  mean=%.3f  min=%.3f  max=%.3f  (lower = more diverse)",
                 sum(pairwise_tms) / len(pairwise_tms),
                 min(pairwise_tms), max(pairwise_tms))
    else:
        log.info("  (no valid pairs for pairwise TM-score)")

    # --- TM-score to nearest training protein (novelty) ---
    novelty_tms = []
    for i in range(len(gen_ca_list)):
        if gen_ca_list[i] is None:
            continue
        ri = nearest_indices[i]
        real = real_data[ri]
        real_seq = real.get("sequence") or "A" * real["seq_len"]
        if set(real_seq) == {"X"}:
            real_seq = "A" * real["seq_len"]
        try:
            real_struct = decoder.decode(real["s_s"], s_z=real.get("s_z"), sequence=real_seq)
            real_ca = extract_ca_coords(real_struct)[:real["seq_len"]]
            real_seq_trunc = real_seq[:real["seq_len"]]

            # Compare at the shorter length
            L_cmp = min(len(gen_ca_list[i]), len(real_ca))
            tm = compute_tm_score(gen_ca_list[i][:L_cmp], real_ca[:L_cmp],
                                  gen_seq_list[i][:L_cmp], real_seq_trunc[:L_cmp])
            novelty_tms.append(tm)
            log.info("  Novelty TM (gen_%d vs real_%d): %.3f", i, ri, tm)
        except Exception as e:
            log.warning("  Novelty TM failed for sample %d: %s", i, e)

    def safe_mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    if novelty_tms:
        log.info("── Novelty (TM to nearest training protein) ────────────")
        log.info("  mean=%.3f  max=%.3f  (lower = more novel)",
                 safe_mean(novelty_tms), max(novelty_tms))

    return {
        "gen_plddt_min": min(valid_plddts, default=0),
        "gen_plddt_median": plddt_median,
        "gen_plddt_mean": safe_mean(valid_plddts),
        "gen_plddt_max": max(valid_plddts, default=0),
        "gen_plddt_frac_designable": frac_designable,
        "gen_ptm_mean": safe_mean(gen_ptms),
        "pairwise_tm_mean": safe_mean(pairwise_tms),
        "pairwise_tm_min": min(pairwise_tms, default=0),
        "novelty_tm_mean": safe_mean(novelty_tms),
        "novelty_tm_max": max(novelty_tms, default=0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate ESM-Drift generator")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_pdbs", type=str, default=None)
    parser.add_argument("--skip_decode", action="store_true",
                        help="Skip ESMFold decoding; report embedding-space metrics only")
    parser.add_argument("--fixed_length", type=int, default=None,
                        help="Generate all samples at this fixed length instead of "
                             "sampling from training distribution")
    args = parser.parse_args()

    device = torch.device(args.device)
    generator, seq_head, config = load_checkpoint(args.checkpoint, device)
    max_len = config["max_len"]
    log.info("Loaded checkpoint: epoch=%s  max_len=%d", config.get("epoch"), max_len)

    # Load dataset
    dataset = EmbeddingDataset(args.data_dir, max_seq_len=max_len)
    if len(dataset) == 0:
        log.error("No data found in %s", args.data_dir)
        return
    log.info("Dataset: %d proteins", len(dataset))

    # Determine protein lengths to generate
    if args.fixed_length is not None:
        lengths = [args.fixed_length] * args.n_samples
        log.info("Generating %d samples at fixed length %d", args.n_samples, args.fixed_length)
    else:
        lengths = sample_lengths_from_data(dataset, args.n_samples)
        log.info("Sampled lengths from training distribution: %s", sorted(lengths))

    # Generate embeddings (and predict sequences if seq_head is available)
    gen_samples, gen_seqs = generate_samples(generator, lengths, device, seq_head=seq_head)
    log.info("Generated %d samples  lengths=%s", len(gen_samples), [s.shape[0] for s in gen_samples])
    if seq_head is not None:
        log.info("Predicted sequences: %s", gen_seqs)

    # Load real data into memory for metrics
    collate = partial(pad_collate, max_len=max_len)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate)
    batch = next(iter(loader))
    real_s_s = batch["s_s"].to(device)
    real_mask = batch["mask"].to(device)
    real_residues = real_s_s[real_mask]  # [n_valid_residues, 1024]

    # Mean-pooled protein-level embeddings
    real_pooled = (real_s_s * real_mask.unsqueeze(-1).float()).sum(1) \
                  / real_mask.sum(1, keepdim=True).float().clamp(min=1)  # [n_proteins, 1024]

    # Embedding-space metrics
    emb_metrics = embedding_metrics(
        gen_samples, real_residues, real_pooled, real_mask, real_s_s, device,
    )
    nearest_indices = emb_metrics["nearest_indices"]

    if args.skip_decode:
        log.info("Skipping ESMFold decode (--skip_decode)")
        return

    # Structure metrics
    real_data = [dataset[i] for i in range(len(dataset))]
    log.info("Decoding with ESMFold...")
    struct_metrics = decode_and_evaluate(
        gen_samples, gen_seqs, lengths, real_data, nearest_indices,
        device, save_dir=args.save_pdbs,
    )

    log.info("═" * 60)
    log.info("Summary:")
    log.info("  pLDDT  mean=%.3f  median=%.3f  designable(>0.7)=%.0f%%",
             struct_metrics["gen_plddt_mean"],
             struct_metrics["gen_plddt_median"],
             100 * struct_metrics["gen_plddt_frac_designable"])
    log.info("  pTM    mean=%.3f", struct_metrics["gen_ptm_mean"])
    log.info("  Diversity (pairwise TM): %.3f  (0=maximally diverse, 1=identical)",
             struct_metrics["pairwise_tm_mean"])
    log.info("  Novelty (TM to nearest training): %.3f  (lower=more novel)",
             struct_metrics["novelty_tm_mean"])
    log.info("  Unique nearest-neighbor proteins: %d / %d",
             emb_metrics["nearest_unique"], args.n_samples)
    log.info("═" * 60)


if __name__ == "__main__":
    main()

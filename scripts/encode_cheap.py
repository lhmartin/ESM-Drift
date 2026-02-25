#!/usr/bin/env python3
"""One-time script: encode ESMFold s_s [L, 1024] → CHEAP z [L, 32].

Reads existing .pt files from data/ (which contain s_s [L, 1024] extracted
from ESMFold), compresses each to a 32-dimensional CHEAP latent, and saves
to data/cheap/<stem>.pt.

The CHEAP hourglass model (shorten_1_dim_32) is a tanh-bounded continuous
autoencoder: encoder compresses [L, 1024] → [L, 32], decoder restores [L, 1024].

Normalization: per-channel min-max computed from OUR data (not CHEAP's CATH
stats) to keep inputs inside the [-1, 1] range the hourglass was trained on.
The normalization tensors are saved to data/cheap/scaler_stats.pt and loaded
at decode time by CheapDecoder.

Usage:
    uv run python scripts/encode_cheap.py [--input_dir data/] [--output_dir data/cheap/]
    uv run python scripts/encode_cheap.py --test  # encode + decode 5 proteins, print pLDDT
"""

import argparse
import logging
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def compute_channel_stats(input_dir: Path, max_files: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel min/max across all s_s embeddings for normalisation.

    Returns:
        chan_min: [1024] per-channel minimum values
        chan_max: [1024] per-channel maximum values
    """
    files = sorted(input_dir.glob("*.pt"))[:max_files]
    log.info("Computing per-channel stats from %d files...", len(files))
    all_s_s = []
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        all_s_s.append(d["s_s"])  # [L, 1024]
    residues = torch.cat(all_s_s, dim=0)  # [N_total, 1024]
    chan_min = residues.min(dim=0).values  # [1024]
    chan_max = residues.max(dim=0).values  # [1024]
    log.info("  Stats from %d total residues, embedding dim=%d", residues.shape[0], residues.shape[1])
    return chan_min, chan_max


def scale(s_s: torch.Tensor, chan_min: torch.Tensor, chan_max: torch.Tensor) -> torch.Tensor:
    """Per-channel min-max normalise to [-1, 1]."""
    chan_min = chan_min.to(s_s.device)
    chan_max = chan_max.to(s_s.device)
    denom = (chan_max - chan_min).clamp_min(1e-6)
    return 2.0 * (s_s - chan_min) / denom - 1.0


def unscale(z_norm: torch.Tensor, chan_min: torch.Tensor, chan_max: torch.Tensor) -> torch.Tensor:
    """Invert per-channel min-max normalisation."""
    chan_min = chan_min.to(z_norm.device)
    chan_max = chan_max.to(z_norm.device)
    denom = (chan_max - chan_min).clamp_min(1e-6)
    return (z_norm + 1.0) / 2.0 * denom + chan_min


def load_cheap_model(device: torch.device):
    from cheap.pretrained import load_model_from_id
    from cheap.constants import CATH_COMPRESS_LEVEL_TO_ID
    model_id = CATH_COMPRESS_LEVEL_TO_ID[1][32]
    log.info("Loading CHEAP shorten_1_dim_32 (model_id=%s)...", model_id)
    model = load_model_from_id(model_id, infer_mode=True).to(device)
    return model


def encode_file(
    path: Path,
    model,
    chan_min: torch.Tensor,
    chan_max: torch.Tensor,
    device: torch.device,
    max_len: int = 512,
) -> dict | None:
    """Encode a single .pt file's s_s to CHEAP z [L, 32].

    Returns a dict ready to save, or None if the file can't be processed.
    """
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "s_s" not in d:
        log.warning("  Skipping %s: no 's_s' key", path.name)
        return None

    s_s = d["s_s"]  # [L, 1024]
    L = s_s.shape[0]
    if L > max_len:
        log.debug("  Skipping %s: L=%d > max_len=%d", path.name, L, max_len)
        return None

    s_s = s_s.unsqueeze(0).to(device)           # [1, L, 1024]
    mask = torch.ones(1, L, dtype=torch.bool, device=device)

    s_s_scaled = scale(s_s, chan_min, chan_max)  # [1, L, 1024] in [-1, 1]

    with torch.no_grad():
        z, _ = model.enc(s_s_scaled, mask)      # [1, L, 32]

    z = z.squeeze(0).cpu()                       # [L, 32]

    out = {
        "z": z,
        "seq_len": d.get("seq_len", torch.tensor(L)),
        "mask": d.get("mask", torch.ones(L, dtype=torch.bool)),
        "aa_indices": d.get("aa_indices"),
        "sequence": d.get("sequence"),
    }
    # Drop None values
    return {k: v for k, v in out.items() if v is not None}


def run_test(input_dir: Path, model, chan_min, chan_max, device):
    """Encode + decode a few proteins and print pLDDT comparison."""
    from esm_drift.utils import StructureDecoder

    log.info("Running reconstruction quality test...")
    decoder = StructureDecoder(device=str(device))
    decoder._load_model()

    files = sorted(input_dir.glob("*.pt"))[10:16]
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        if "s_s" not in d:
            continue
        s_s = d["s_s"].unsqueeze(0).to(device)
        L = s_s.shape[1]
        mask_t = torch.ones(1, L, dtype=torch.bool, device=device)
        seq = d.get("sequence")

        struct_orig = decoder.decode(s_s.squeeze(0), sequence=seq)
        plddt_orig = struct_orig["plddt"].mean().item()

        s_s_scaled = scale(s_s, chan_min, chan_max)
        with torch.no_grad():
            z, _ = model.enc(s_s_scaled, mask_t)
            s_s_norm_recon = model.dec(z, mask_t)
        s_s_recon = unscale(s_s_norm_recon, chan_min, chan_max)

        struct_recon = decoder.decode(s_s_recon.squeeze(0), sequence=seq)
        plddt_recon = struct_recon["plddt"].mean().item()

        print(f"  {f.name}: L={L}  pLDDT orig={plddt_orig:.3f}  recon={plddt_recon:.3f}  delta={plddt_recon - plddt_orig:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Encode ESMFold s_s → CHEAP z [L, 32]")
    parser.add_argument("--input_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="data/cheap/")
    parser.add_argument("--max_len", type=int, default=512, help="Skip proteins longer than this")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test", action="store_true", help="Run reconstruction quality test only")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_cheap_model(device)
    chan_min, chan_max = compute_channel_stats(input_dir)

    if args.test:
        run_test(input_dir, model, chan_min, chan_max, device)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save normalization stats so CheapDecoder can load them
    stats_path = output_dir / "scaler_stats.pt"
    torch.save({"chan_min": chan_min, "chan_max": chan_max}, stats_path)
    log.info("Saved normalisation stats to %s", stats_path)

    files = sorted(input_dir.glob("*.pt"))
    log.info("Encoding %d files from %s → %s", len(files), input_dir, output_dir)

    n_ok, n_skip = 0, 0
    for i, f in enumerate(files):
        out_path = output_dir / f.name
        if out_path.exists():
            n_ok += 1
            continue

        result = encode_file(f, model, chan_min, chan_max, device, max_len=args.max_len)
        if result is None:
            n_skip += 1
            continue

        torch.save(result, out_path)
        n_ok += 1

        if (i + 1) % 200 == 0:
            log.info("  %d/%d encoded (%d skipped)", n_ok, len(files), n_skip)

    log.info("Done: %d encoded, %d skipped.", n_ok, n_skip)


if __name__ == "__main__":
    main()

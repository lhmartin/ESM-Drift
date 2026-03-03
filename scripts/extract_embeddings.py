#!/usr/bin/env python
"""Extract ESMFold embeddings (s_s) from PDB/mmCIF structure files.

Parsing (CPU) and extraction (GPU) are pipelined: background processes parse
files while the GPU processes completed batches, so the GPU starts within
seconds and both resources stay busy throughout.

Usage:
    # Process a directory of PDB files
    python scripts/extract_embeddings.py --input_dir /path/to/pdbs --output_dir data/embeddings

    # Filter by quality and length, tune batch size and parse workers
    python scripts/extract_embeddings.py --input_dir pdbs/ --output_dir data/embeddings \\
        --max_seq_len 512 --min_plddt 70.0 --batch_size 16 --parse_workers 8
"""

import argparse
import functools
import logging
import random
from multiprocessing import Pool
from pathlib import Path

import torch
from tqdm import tqdm

from esm_drift.data.extract import (
    COMPRESSED_EXTENSIONS,
    EmbeddingExtractor,
    collect_sequences_from_file,
)

STRUCTURE_EXTENSIONS = {".pdb", ".ent", ".cif", ".mmcif"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def find_structure_files(input_dir: Path) -> list[Path]:
    files = []
    for ext in STRUCTURE_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
        files.extend(input_dir.rglob(f"*{ext}.gz"))
    return sorted(set(files))


def output_stem(filepath: Path) -> str:
    stem = filepath.stem
    if filepath.suffix.lower() in COMPRESSED_EXTENSIONS:
        stem = Path(stem).stem
    return stem


def parse_one_file(
    filepath: Path,
    output_dir: Path,
    chain_id: str | None,
    max_seq_len: int,
) -> list[tuple[Path, str, str, Path]]:
    """Parse one structure file; return tasks not yet extracted."""
    try:
        chains = collect_sequences_from_file(filepath, chain_id=chain_id, max_seq_len=max_seq_len)
    except Exception as e:
        log.error("Failed to parse %s: %s", filepath, e)
        return []
    stem = output_stem(filepath)
    tasks = []
    for cid, seq in chains:
        out_path = output_dir / f"{stem}_{cid}.pt"
        if not out_path.exists():
            tasks.append((filepath, cid, seq, out_path))
    return tasks


def run_batch(
    extractor: EmbeddingExtractor,
    batch: list[tuple],
    min_plddt: float,
) -> tuple[int, int]:
    """GPU-extract one batch; save results. Returns (saved, skipped)."""
    seqs = [t[2] for t in batch]

    try:
        results = extractor.extract_batch(seqs)
    except (RuntimeError, IndexError) as e:
        if len(batch) == 1:
            log.error("Skipping %s chain %s (L=%d): %s",
                      batch[0][0].name, batch[0][1], len(seqs[0]), e)
            return 0, 1
        log.warning("Batch of %d failed, retrying one-by-one: %s", len(batch), e)
        results = []
        for i, seq in enumerate(seqs):
            try:
                results.append(extractor.extract_batch([seq])[0])
            except (RuntimeError, IndexError) as e2:
                log.error("Skipping %s chain %s (L=%d): %s",
                          batch[i][0].name, batch[i][1], len(seq), e2)
                results.append(None)

    saved = skipped = 0
    for (filepath, cid, seq, out_path), result in zip(batch, results):
        if result is None:
            skipped += 1
            continue
        mean_plddt = result["plddt"].mean().item()
        if mean_plddt < min_plddt:
            log.info("Skipping %s chain %s (pLDDT %.1f < %.1f)",
                     filepath.name, cid, mean_plddt, min_plddt)
            skipped += 1
            continue
        data = {
            "s_s": result["s_s"].half(),
            "sequence": seq,
            "source_file": str(filepath),
            "chain_id": cid,
            "plddt": result["plddt"].half(),
            "ptm": result["ptm"],
            "seq_len": len(seq),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out_path)
        log.info("Saved %s (L=%d, pLDDT=%.1f, pTM=%.3f)",
                 out_path.name, len(seq), mean_plddt, result["ptm"])
        saved += 1
    return saved, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESMFold embeddings from protein structure files."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=Path)
    input_group.add_argument("--input_file", type=Path)

    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--min_plddt", type=float, default=0.0)
    parser.add_argument("--chain_id", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--parse_workers", type=int, default=8,
                        help="Background CPU threads for decompressing/parsing CIF files (default: 8)")
    parser.add_argument("--sort_buffer", type=int, default=512,
                        help="Buffer size for approximate length-sorting before batching (default: 512)")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    # ── Gather files ──────────────────────────────────────────────────────────
    files = [args.input_file] if args.input_file else find_structure_files(args.input_dir)
    if not files:
        log.error("No structure files found")
        return
    log.info("Found %d structure files", len(files))

    if args.world_size > 1:
        files = files[args.rank::args.world_size]
        log.info("Rank %d/%d: %d files", args.rank, args.world_size, len(files))

    # Shuffle so protein structures are distributed throughout (avoids long runs of
    # DNA/RNA-only files stalling the sort buffer at the start)
    random.seed(args.rank)
    random.shuffle(files)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Lazy-load extractor (model loads on first extract_batch call) ─────────
    extractor = EmbeddingExtractor(
        device=args.device,
        max_seq_len=args.max_seq_len,
        compile_model=args.compile,
    )

    # ── Pipeline: parse on CPU threads, extract on GPU ────────────────────────
    # Background threads decompress + parse CIFs while GPU processes batches.
    # A sort buffer of ~512 tasks gives good length-homogeneous batches without
    # waiting for all files to be parsed first.

    total_saved = total_skipped = 0
    buffer: list[tuple] = []

    def drain_buffer(min_size: int) -> None:
        nonlocal total_saved, total_skipped, buffer
        while len(buffer) >= min_size:
            buffer.sort(key=lambda t: len(t[2]))
            batch = buffer[:args.batch_size]
            buffer = buffer[args.batch_size:]
            s, sk = run_batch(extractor, batch, args.min_plddt)
            total_saved += s
            total_skipped += sk

    _parse = functools.partial(
        parse_one_file,
        output_dir=args.output_dir,
        chain_id=args.chain_id,
        max_seq_len=args.max_seq_len,
    )

    with Pool(processes=args.parse_workers) as pool:
        for tasks in tqdm(
            pool.imap_unordered(_parse, files, chunksize=4),
            total=len(files),
            desc="Parsing+extracting",
        ):
            buffer.extend(tasks)
            drain_buffer(args.sort_buffer)

    # Flush anything left in the buffer
    drain_buffer(1)

    log.info("Done. Saved %d, skipped %d.", total_saved, total_skipped)


if __name__ == "__main__":
    main()

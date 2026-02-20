#!/usr/bin/env python
"""Extract ESMFold embeddings (s_s, s_z) from PDB/mmCIF structure files.

Usage:
    # Process a directory of PDB files
    python scripts/extract_embeddings.py --input_dir /path/to/pdbs --output_dir data/embeddings

    # Process a single file
    python scripts/extract_embeddings.py --input_file protein.pdb --output_dir data/embeddings

    # Filter by quality and length, use batching and torch.compile
    python scripts/extract_embeddings.py --input_dir pdbs/ --output_dir data/embeddings \
        --max_seq_len 512 --min_plddt 70.0 --batch_size 4 --compile

    # Use CPU (slower but no GPU needed)
    python scripts/extract_embeddings.py --input_file protein.pdb --output_dir data/embeddings \
        --device cpu
"""

import argparse
import logging
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
    """Recursively find all PDB/mmCIF files in a directory (including .gz compressed)."""
    files = []
    for ext in STRUCTURE_EXTENSIONS:
        files.extend(input_dir.rglob(f"*{ext}"))
        files.extend(input_dir.rglob(f"*{ext}.gz"))

    # Deduplicate (rglob already searches recursively, including subdirs)
    return sorted(set(files))


def output_stem(filepath: Path) -> str:
    """Get the base name for output files, stripping compression extension."""
    stem = filepath.stem
    if filepath.suffix.lower() in COMPRESSED_EXTENSIONS:
        stem = Path(stem).stem  # "protein.pdb.gz" -> "protein"
    return stem


def collect_tasks(
    files: list[Path],
    output_dir: Path,
    chain_id: str | None,
    max_seq_len: int,
) -> list[tuple[Path, str, str, Path]]:
    """Parse all structure files and collect (filepath, chain_id, seq, output_path) tasks.

    Skips entries whose output .pt file already exists.
    Sorts tasks by sequence length (shortest first) to improve batching efficiency.
    """
    tasks = []
    for filepath in tqdm(files, desc="Parsing structures"):
        try:
            chains = collect_sequences_from_file(filepath, chain_id=chain_id, max_seq_len=max_seq_len)
        except Exception as e:
            log.error("Failed to parse %s: %s", filepath, e)
            continue

        if not chains:
            log.warning("No valid chains found in %s", filepath)
            continue

        stem = output_stem(filepath)
        for cid, seq in chains:
            out_path = output_dir / f"{stem}_{cid}.pt"
            if out_path.exists():
                log.info("Skipping %s (already exists)", out_path)
                continue
            tasks.append((filepath, cid, seq, out_path))

    # Sort by length so batches contain sequences of similar length (minimises padding waste)
    tasks.sort(key=lambda t: len(t[2]))
    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Extract ESMFold embeddings from protein structure files."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", type=Path, help="Directory of PDB/mmCIF files")
    input_group.add_argument("--input_file", type=Path, help="Single PDB/mmCIF file")

    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for .pt files")
    parser.add_argument("--device", type=str, default="cuda", help="Device for ESMFold (default: cuda)")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max sequence length (default: 1024)")
    parser.add_argument("--min_plddt", type=float, default=0.0, help="Min mean pLDDT to save (default: 0.0)")
    parser.add_argument("--chain_id", type=str, default=None, help="Only process this chain ID")
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Number of sequences per GPU batch (default: 1). "
             "Increase for shorter sequences; memory scales as batch * L^2.",
    )
    parser.add_argument(
        "--compile", action="store_true",
        help="Compile ESMFold with torch.compile for faster repeated inference.",
    )
    args = parser.parse_args()

    # Gather input files
    if args.input_file:
        files = [args.input_file]
    else:
        files = find_structure_files(args.input_dir)

    if not files:
        log.error("No structure files found")
        return

    log.info("Found %d structure files", len(files))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: parse all structures (CPU, fast) and collect extraction tasks
    tasks = collect_tasks(files, args.output_dir, args.chain_id, args.max_seq_len)
    log.info("%d sequences to extract", len(tasks))

    if not tasks:
        log.info("Nothing to do.")
        return

    # Phase 2: batch extraction (GPU)
    extractor = EmbeddingExtractor(
        device=args.device,
        max_seq_len=args.max_seq_len,
        compile_model=args.compile,
    )

    total_saved = 0
    total_skipped = 0

    batch_iter = range(0, len(tasks), args.batch_size)
    for batch_start in tqdm(batch_iter, desc="Extracting embeddings"):
        batch = tasks[batch_start : batch_start + args.batch_size]
        seqs = [t[2] for t in batch]

        try:
            results = extractor.extract_batch(seqs)
        except RuntimeError as e:
            log.error("Batch extraction failed (batch_size=%d, max_L=%d): %s",
                      len(seqs), max(len(s) for s in seqs), e)
            log.error("Consider reducing --batch_size")
            total_skipped += len(batch)
            continue

        for (filepath, cid, seq, out_path), result in zip(batch, results):
            mean_plddt = result["plddt"].mean().item()
            if mean_plddt < args.min_plddt:
                log.info("Skipping %s chain %s (pLDDT %.1f < %.1f)",
                         filepath.name, cid, mean_plddt, args.min_plddt)
                total_skipped += 1
                continue

            data = {
                "s_s": result["s_s"],
                "s_z": result["s_z"],
                "sequence": seq,
                "source_file": str(filepath),
                "chain_id": cid,
                "plddt": result["plddt"],
                "ptm": result["ptm"],
                "seq_len": len(seq),
            }
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, out_path)
            log.info("Saved %s (L=%d, pLDDT=%.1f, pTM=%.3f)",
                     out_path.name, len(seq), mean_plddt, result["ptm"])
            total_saved += 1

    log.info("Done. Saved %d embeddings, skipped %d.", total_saved, total_skipped)


if __name__ == "__main__":
    main()

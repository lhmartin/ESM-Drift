#!/usr/bin/env python
"""Extract ESMFold embeddings (s_s, s_z) from PDB/mmCIF structure files.

Usage:
    # Process a directory of PDB files
    python scripts/extract_embeddings.py --input_dir /path/to/pdbs --output_dir data/embeddings

    # Process a single file
    python scripts/extract_embeddings.py --input_file protein.pdb --output_dir data/embeddings

    # Filter by quality and length
    python scripts/extract_embeddings.py --input_dir pdbs/ --output_dir data/embeddings \
        --max_seq_len 512 --min_plddt 70.0

    # Use CPU (slower but no GPU needed)
    python scripts/extract_embeddings.py --input_file protein.pdb --output_dir data/embeddings \
        --device cpu
"""

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from esm_drift.data.extract import EmbeddingExtractor, process_structure_file

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
    args = parser.parse_args()

    # Gather input files
    if args.input_file:
        files = [args.input_file]
    else:
        files = find_structure_files(args.input_dir)

    if not files:
        log.error("No structure files found")
        return

    log.info("Found %d structure files to process", len(files))

    # Create extractor (loads model lazily on first use)
    extractor = EmbeddingExtractor(device=args.device, max_seq_len=args.max_seq_len)

    total_saved = 0
    total_skipped = 0

    for filepath in tqdm(files, desc="Processing structures"):
        saved = process_structure_file(
            filepath=filepath,
            output_dir=args.output_dir,
            extractor=extractor,
            chain_id=args.chain_id,
            min_plddt=args.min_plddt,
        )
        total_saved += len(saved)
        if not saved:
            total_skipped += 1

    log.info("Done. Saved %d embeddings, skipped %d files.", total_saved, total_skipped)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download and process structures from the ESM Metagenomic Atlas.

Downloads high-confidence predicted structures from the ESMFold Atlas,
filters by sequence length, and extracts ESMFold embeddings (s_s, s_z)
into the project's .pt format.

The Atlas organizes structures in tarballs binned by pTM and pLDDT quality.
Each tarball contains gzipped PDB files (~25GB per tarball).

Usage:
    # Download highest quality bin, filter to len<=128, extract embeddings
    uv run python scripts/download_atlas.py \
        --output_dir data/ \
        --download_dir /home/luke/data/atlas/ \
        --max_seq_len 128 \
        --max_proteins 1000 \
        --device cuda

    # Download and filter PDBs only (no ESMFold extraction)
    uv run python scripts/download_atlas.py \
        --output_dir data/ \
        --download_dir /home/luke/data/atlas/ \
        --max_seq_len 128 \
        --skip_extraction

    # Use a specific quality bin
    uv run python scripts/download_atlas.py \
        --output_dir data/ \
        --download_dir /home/luke/data/atlas/ \
        --bin_name tm_.80_.90_plddt_.90_1
"""

import argparse
import gzip
import logging
import re
import subprocess
import tarfile
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Three-letter to one-letter amino acid code mapping
THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

BASE_URL = "https://dl.fbaipublicfiles.com/esmatlas/v2023_02/full/tarballs"

# Number of tarball chunks per bin (from the Atlas tarballs.txt)
BIN_CHUNKS = {
    "tm_.90_1_plddt_.90_1": 17,   # 00-16
    "tm_.80_.90_plddt_.90_1": 4,   # 00-03
    "tm_.90_1_plddt_.80_.90": 4,   # 00-03
    "tm_.80_.90_plddt_.80_.90": 4, # 00-03
    "tm_.70_.80_plddt_.90_1": 3,   # 00-02
    "tm_.90_1_plddt_.70_.80": 2,   # 00-01
}


def get_tarball_urls(bin_name: str) -> list[str]:
    """Get all tarball URLs for a given quality bin."""
    n_chunks = BIN_CHUNKS.get(bin_name)
    if n_chunks is None:
        raise ValueError(f"Unknown bin: {bin_name}. Known bins: {list(BIN_CHUNKS.keys())}")
    return [f"{BASE_URL}/{bin_name}_{i:02d}.tar.gz" for i in range(n_chunks)]


def download_tarball(url: str, download_dir: Path) -> Path:
    """Download a tarball using aria2c or wget."""
    filename = url.split("/")[-1]
    out_path = download_dir / filename
    if out_path.exists():
        log.info("Already downloaded: %s", out_path)
        return out_path

    download_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s ...", url)

    # Try aria2c first (faster, supports resume), fall back to wget
    try:
        subprocess.run(
            ["aria2c", "--dir", str(download_dir), "--out", filename,
             "--continue=true", "--max-connection-per-server=4", url],
            check=True,
        )
    except FileNotFoundError:
        log.info("aria2c not found, using wget")
        subprocess.run(
            ["wget", "-c", "-O", str(out_path), url],
            check=True,
        )

    return out_path


def sequence_from_pdb_bytes(pdb_bytes: bytes) -> str | None:
    """Extract amino acid sequence from PDB file bytes (fast text parsing).

    Reads ATOM lines with CA atoms to reconstruct sequence. Much faster than
    BioPython PDB parser since we only need the sequence, not coordinates.
    """
    try:
        text = pdb_bytes.decode("utf-8")
    except Exception:
        return None

    residues = []
    seen_res_ids = set()

    for line in text.split("\n"):
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        # PDB format: columns 13-16 = atom name, 18-20 = residue name,
        # 22 = chain, 23-26 = residue sequence number
        if len(line) < 27:
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        res_name = line[17:20].strip()
        chain_id = line[21]
        res_seq = line[22:27].strip()
        res_id = (chain_id, res_seq)
        if res_id in seen_res_ids:
            continue
        seen_res_ids.add(res_id)
        aa = THREE_TO_ONE.get(res_name)
        residues.append(aa if aa else "X")

    seq = "".join(residues)
    return seq if len(seq) >= 10 else None


def fast_residue_count(pdb_bytes: bytes) -> int:
    """Quickly count residues by counting unique CA ATOM lines.

    Even faster than sequence_from_pdb_bytes when we only need the length.
    """
    try:
        text = pdb_bytes.decode("utf-8")
    except Exception:
        return 0
    seen = set()
    for line in text.split("\n"):
        if line.startswith("ATOM") and len(line) > 26 and line[12:16].strip() == "CA":
            seen.add((line[21], line[22:27]))
    return len(seen)


def _manifest_path(tarball_path: Path, pdb_dir: Path) -> Path:
    """Path to the manifest file for a processed tarball."""
    return pdb_dir / f".{tarball_path.stem.replace('.tar', '')}.manifest.tsv"


def _load_manifest(manifest: Path, pdb_dir: Path) -> list[tuple[str, str, Path]]:
    """Load a previously-saved tarball manifest."""
    proteins = []
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        protein_id, seq, pdb_name = line.split("\t")
        proteins.append((protein_id, seq, pdb_dir / pdb_name))
    return proteins


def _save_manifest(
    manifest: Path, proteins: list[tuple[str, str, Path]]
) -> None:
    """Save a tarball manifest (protein_id, sequence, pdb filename)."""
    with open(manifest, "w") as f:
        for protein_id, seq, pdb_path in proteins:
            f.write(f"{protein_id}\t{seq}\t{pdb_path.name}\n")


def extract_pdbs_from_tarball(
    tarball_path: Path,
    pdb_dir: Path,
    min_seq_len: int = 10,
    max_seq_len: int = 128,
    max_proteins: int | None = None,
    existing_count: int = 0,
) -> list[tuple[str, str, Path]]:
    """Extract PDB files from a tarball, filtering by sequence length.

    If a manifest exists from a previous complete run, loads from that instead
    of re-reading the tarball. Manifests are only written when the full tarball
    is processed (not when stopped early by max_proteins).

    Args:
        tarball_path: Path to downloaded .tar.gz file.
        pdb_dir: Directory to save filtered PDB files.
        min_seq_len: Minimum sequence length.
        max_seq_len: Maximum sequence length.
        max_proteins: Stop after this many proteins total.
        existing_count: Number of proteins already extracted (for max_proteins tracking).

    Returns:
        List of (protein_id, sequence, pdb_path) tuples for extracted proteins.
    """
    pdb_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing manifest from a previous complete run
    manifest = _manifest_path(tarball_path, pdb_dir)
    if manifest.exists():
        all_proteins = _load_manifest(manifest, pdb_dir)
        if max_proteins:
            remaining = max_proteins - existing_count
            all_proteins = all_proteins[:remaining]
        log.info(
            "Loaded %d proteins from manifest for %s (skipping tarball)",
            len(all_proteins), tarball_path.name,
        )
        return all_proteins

    extracted = []
    skipped = 0
    count = existing_count
    hit_limit = False

    log.info("Processing tarball: %s", tarball_path.name)

    with tarfile.open(tarball_path, "r:gz") as tar:
        for member in tar:
            if max_proteins and count >= max_proteins:
                hit_limit = True
                break

            if not member.name.endswith(".pdb.gz"):
                continue

            # Extract protein ID from path like ./657/MGYP004442083657.pdb.gz
            protein_id = Path(member.name).stem.replace(".pdb", "")
            out_pdb = pdb_dir / f"{protein_id}.pdb"

            if out_pdb.exists():
                # Already extracted — use fast count to check length
                n_res = fast_residue_count(out_pdb.read_bytes())
                if min_seq_len <= n_res <= max_seq_len:
                    seq = sequence_from_pdb_bytes(out_pdb.read_bytes())
                    if seq:
                        extracted.append((protein_id, seq, out_pdb))
                        count += 1
                continue

            # Read and decompress the gzipped PDB
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                pdb_bytes = gzip.decompress(f.read())
            except Exception:
                continue

            # Fast length check first (avoids full sequence parse for most proteins)
            n_res = fast_residue_count(pdb_bytes)
            if not (min_seq_len <= n_res <= max_seq_len):
                skipped += 1
                continue

            # Full sequence parse only for proteins that pass length filter
            seq = sequence_from_pdb_bytes(pdb_bytes)
            if seq is None:
                skipped += 1
                continue

            # Save filtered PDB
            out_pdb.write_bytes(pdb_bytes)
            extracted.append((protein_id, seq, out_pdb))
            count += 1

            if count % 500 == 0:
                log.info("  Extracted %d proteins so far (skipped %d)", count, skipped)

    log.info(
        "Tarball %s: extracted %d proteins, skipped %d",
        tarball_path.name, len(extracted), skipped,
    )

    # Only save manifest if we processed the entire tarball (not truncated by max_proteins)
    if not hit_limit:
        _save_manifest(manifest, extracted)
        log.info("Saved manifest for %s (%d proteins)", tarball_path.name, len(extracted))

    return extracted


def run_embedding_extraction(
    proteins: list[tuple[str, str, Path]],
    output_dir: Path,
    device: str = "cuda",
    max_seq_len: int = 1024,
    batch_size: int = 1,
    compile_model: bool = False,
) -> int:
    """Run ESMFold embedding extraction on filtered PDB files.

    Uses batched extraction for throughput. Proteins are sorted by sequence
    length to minimize padding waste within each batch.

    Returns:
        Number of successfully extracted embeddings.
    """
    from esm_drift.data.extract import EmbeddingExtractor

    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = EmbeddingExtractor(
        device=device, max_seq_len=max_seq_len, compile_model=compile_model,
    )

    # Filter out already-extracted, then sort by length for batching efficiency
    todo = []
    skipped = 0
    for protein_id, sequence, pdb_path in proteins:
        out_path = output_dir / f"{protein_id}.pt"
        if out_path.exists():
            skipped += 1
            continue
        todo.append((protein_id, sequence, pdb_path, out_path))

    if skipped:
        log.info("Skipping %d already-extracted embeddings", skipped)

    todo.sort(key=lambda t: len(t[1]))
    success = 0

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start : batch_start + batch_size]
        seqs = [t[1] for t in batch]

        try:
            results = extractor.extract_batch(seqs)
        except RuntimeError as e:
            log.error("Batch failed (size=%d, max_L=%d): %s. Try reducing --batch_size",
                      len(seqs), max(len(s) for s in seqs), e)
            continue

        for (protein_id, sequence, pdb_path, out_path), result in zip(batch, results):
            data = {
                "s_s": result["s_s"],
                "s_z": result["s_z"],
                "sequence": sequence,
                "source_file": str(pdb_path),
                "chain_id": "A",
                "plddt": result["plddt"],
                "ptm": result["ptm"],
                "seq_len": len(sequence),
            }
            torch.save(data, out_path)
            success += 1

        done = batch_start + len(batch)
        if done % 100 < batch_size or done == len(todo):
            log.info("Extraction progress: %d/%d (success=%d)", done, len(todo), success)

    return success + skipped


def main():
    parser = argparse.ArgumentParser(description="Download and process ESM Metagenomic Atlas")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for .pt embedding files")
    parser.add_argument("--download_dir", type=str, required=True,
                        help="Directory for downloaded tarballs and extracted PDBs")
    parser.add_argument("--bin_name", type=str, default="tm_.90_1_plddt_.90_1",
                        help="Quality bin to download (default: highest quality)")
    parser.add_argument("--max_seq_len", type=int, default=128,
                        help="Maximum sequence length filter")
    parser.add_argument("--min_seq_len", type=int, default=30,
                        help="Minimum sequence length filter")
    parser.add_argument("--max_proteins", type=int, default=None,
                        help="Maximum number of proteins to extract")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for ESMFold extraction")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Sequences per GPU batch (default: 1). Memory scales as batch * L^2.")
    parser.add_argument("--compile", action="store_true",
                        help="Compile ESMFold with torch.compile for faster repeated inference")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Only download and filter PDBs, skip ESMFold extraction")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download, process existing tarballs only")

    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    pdb_dir = download_dir / "pdbs"
    output_dir = Path(args.output_dir)

    urls = get_tarball_urls(args.bin_name)
    log.info("Bin %s: %d tarballs to process", args.bin_name, len(urls))

    # Phase 1: Download and extract PDBs
    all_proteins = []
    for url in urls:
        if args.max_proteins and len(all_proteins) >= args.max_proteins:
            break

        if not args.skip_download:
            tarball_path = download_tarball(url, download_dir)
        else:
            tarball_path = download_dir / url.split("/")[-1]
            if not tarball_path.exists():
                log.warning("Tarball not found: %s", tarball_path)
                continue

        proteins = extract_pdbs_from_tarball(
            tarball_path, pdb_dir,
            min_seq_len=args.min_seq_len,
            max_seq_len=args.max_seq_len,
            max_proteins=args.max_proteins,
            existing_count=len(all_proteins),
        )
        all_proteins.extend(proteins)

    log.info("Total proteins after filtering: %d", len(all_proteins))

    if args.max_proteins:
        all_proteins = all_proteins[:args.max_proteins]
        log.info("Capped to %d proteins", len(all_proteins))

    # Phase 2: Run ESMFold embedding extraction
    if not args.skip_extraction:
        log.info("Starting ESMFold embedding extraction...")
        n_success = run_embedding_extraction(
            all_proteins, output_dir,
            device=args.device,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            compile_model=args.compile,
        )
        log.info("Extraction complete: %d/%d successful", n_success, len(all_proteins))
    else:
        log.info("Skipping ESMFold extraction (--skip_extraction)")
        # Save a manifest of filtered proteins for later use
        manifest_path = download_dir / "filtered_proteins.txt"
        with open(manifest_path, "w") as f:
            for pid, seq, pdb_path in all_proteins:
                f.write(f"{pid}\t{len(seq)}\t{seq}\t{pdb_path}\n")
        log.info("Saved manifest to %s", manifest_path)


if __name__ == "__main__":
    main()

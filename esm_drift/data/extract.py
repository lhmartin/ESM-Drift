"""Extract ESMFold intermediate embeddings (s_s, s_z) from protein structures."""

import gzip
import logging
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from Bio.PDB import MMCIFParser, PDBParser



log = logging.getLogger(__name__)

# Standard amino acids that ESMFold understands
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Three-letter to one-letter amino acid code mapping
THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# ESMFold dimension constants
S_S_DIM = 1024  # sequence_state_dim (per-residue)
S_Z_DIM = 128   # pairwise_state_dim (per-pair)

COMPRESSED_EXTENSIONS = {".gz"}


@contextmanager
def _maybe_decompress(filepath: Path):
    """If the file is compressed (.gz), decompress to a temp file and yield its path.

    Otherwise, yield the original path unchanged.
    """
    if filepath.suffix.lower() in COMPRESSED_EXTENSIONS:
        # e.g. "protein.pdb.gz" -> suffix=".gz", stem="protein.pdb"
        decompressed_name = Path(filepath.stem)  # strips the .gz
        suffix = decompressed_name.suffix  # gets .pdb / .cif / etc.

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with gzip.open(filepath, "rb") as f_in:
                shutil.copyfileobj(f_in, tmp)

        try:
            log.info("Decompressed %s -> %s", filepath.name, tmp_path.name)
            yield tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        yield filepath


@dataclass
class ProteinEmbedding:
    """Container for extracted ESMFold embeddings."""
    s_s: torch.Tensor          # [L, 1024] single representation
    s_z: torch.Tensor          # [L, L, 128] pair representation
    sequence: str              # amino acid sequence
    source_file: str           # original file path
    chain_id: str              # chain identifier
    plddt: torch.Tensor        # [L] per-residue confidence
    ptm: float                 # predicted TM-score


def sequence_from_structure(filepath: Path, chain_id: str | None = None) -> list[tuple[str, str]]:
    """Extract amino acid sequences from a PDB/mmCIF file.

    Returns list of (chain_id, sequence) tuples.
    """
    suffix = filepath.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    elif suffix in (".pdb", ".ent"):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    structure = parser.get_structure(filepath.stem, str(filepath))
    model = structure[0]

    results = []
    for chain in model.get_chains():
        if chain_id is not None and chain.id != chain_id:
            continue

        residues = []
        for residue in chain.get_residues():
            # Skip heteroatoms (water, ligands, etc.)
            if residue.id[0] != " ":
                continue
            aa = THREE_TO_ONE.get(residue.resname)
            if aa is not None:
                residues.append(aa)
            else:
                # Non-standard residue, replace with X (will be masked by ESMFold)
                residues.append("X")

        seq = "".join(residues)
        if len(seq) >= 10:  # skip very short fragments
            results.append((chain.id, seq))

    return results


def collect_sequences_from_file(
    filepath: Path,
    chain_id: str | None = None,
    max_seq_len: int = 1024,
) -> list[tuple[str, str]]:
    """Parse a PDB/mmCIF file and return (chain_id, sequence) pairs.

    Handles .gz compression automatically. Sequences exceeding max_seq_len are truncated.
    """
    with _maybe_decompress(filepath) as actual_path:
        chains = sequence_from_structure(actual_path, chain_id=chain_id)
    return [(cid, seq[:max_seq_len]) for cid, seq in chains]


class EmbeddingExtractor:
    """Extracts ESMFold intermediate representations from protein sequences.

    Loads ESMFold once and reuses it across many proteins.
    """

    def __init__(self, device: str = "cuda", max_seq_len: int = 1024, compile_model: bool = False):
        self.device = torch.device(device)
        self.max_seq_len = max_seq_len
        self._compile_model = compile_model
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy-load ESMFold model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoTokenizer, EsmForProteinFolding

        log.info("Loading ESMFold model (this takes a moment)...")
        self._tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self._model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self._model = self._model.to(self.device)
        self._model.eval()
        log.info("ESMFold loaded on %s", self.device)

        if self._compile_model:
            log.info("Compiling ESMFold with torch.compile (first inference call will be slow)...")
            try:
                self._model = torch.compile(self._model, mode="default", dynamic=True)
                log.info("torch.compile succeeded")
            except Exception as e:
                log.warning("torch.compile failed (%s), continuing without compilation", e)

    @torch.no_grad()
    def extract_batch(self, sequences: list[str]) -> list[dict]:
        """Run ESMFold on a batch of sequences.

        Sequences are padded to the longest in the batch. Returns one result dict
        per sequence with keys: s_s [L, 1024], s_z [L, L, 128], plddt [L], ptm.
        """
        self._load_model()

        truncated = []
        for seq in sequences:
            if len(seq) > self.max_seq_len:
                log.warning("Sequence length %d exceeds max %d, truncating", len(seq), self.max_seq_len)
                truncated.append(seq[:self.max_seq_len])
            else:
                truncated.append(seq)
        sequences = truncated

        inputs = self._tokenizer(
            sequences, return_tensors="pt", add_special_tokens=False, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            outputs = self._model(**inputs)

        results = []
        for i, seq in enumerate(sequences):
            L = len(seq)
            s_s = outputs.s_s[i, :L].cpu()        # [L, 1024]
            s_z = outputs.s_z[i, :L, :L].cpu()    # [L, L, 128]
            plddt = outputs.plddt[i, :L, 0].cpu() # [L]
            # ptm shape varies: scalar when batch=1, [B] otherwise
            ptm_t = outputs.ptm
            ptm = ptm_t[i].item() if ptm_t.dim() > 0 and ptm_t.shape[0] > 1 else ptm_t.item()
            results.append({"s_s": s_s, "s_z": s_z, "plddt": plddt, "ptm": ptm})

        return results

    @torch.no_grad()
    def extract(self, sequence: str) -> dict:
        """Run ESMFold on a single sequence and return intermediate embeddings.

        Args:
            sequence: Amino acid sequence (single-letter codes).

        Returns:
            Dict with keys: s_s [L, 1024], s_z [L, L, 128], plddt [L], ptm
        """
        return self.extract_batch([sequence])[0]

    def extract_and_save(
        self,
        sequence: str,
        output_path: Path,
        source_file: str = "",
        chain_id: str = "",
    ) -> Path:
        """Extract embeddings and save to a .pt file.

        The saved file contains:
            - s_s: [L, 1024] single representation
            - s_z: [L, L, 128] pair representation
            - sequence: amino acid string
            - source_file: origin file path
            - chain_id: chain identifier
            - plddt: [L] per-residue confidence
            - ptm: scalar confidence score
            - seq_len: sequence length
        """
        result = self.extract(sequence)

        data = {
            "s_s": result["s_s"],
            "s_z": result["s_z"],
            "sequence": sequence,
            "source_file": source_file,
            "chain_id": chain_id,
            "plddt": result["plddt"],
            "ptm": result["ptm"],
            "seq_len": len(sequence),
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, output_path)
        log.info("Saved embeddings to %s (L=%d, ptm=%.3f)", output_path, len(sequence), result["ptm"])
        return output_path


def process_structure_file(
    filepath: Path,
    output_dir: Path,
    extractor: EmbeddingExtractor,
    chain_id: str | None = None,
    min_plddt: float = 0.0,
) -> list[Path]:
    """Process a single PDB/mmCIF file: extract sequences, compute embeddings, save.

    Args:
        filepath: Path to PDB or mmCIF file.
        output_dir: Directory to save .pt embedding files.
        extractor: Reusable EmbeddingExtractor instance.
        chain_id: If set, only process this chain.
        min_plddt: Skip proteins with mean pLDDT below this threshold.

    Returns:
        List of paths to saved embedding files.
    """
    # Decompress if needed, then parse the structure
    with _maybe_decompress(filepath) as actual_path:
        chains = sequence_from_structure(actual_path, chain_id=chain_id)

    if not chains:
        log.warning("No valid chains found in %s", filepath)
        return []

    # Use the original filename (minus .gz) for the output stem
    stem = filepath.stem
    if filepath.suffix.lower() in COMPRESSED_EXTENSIONS:
        stem = Path(stem).stem  # "protein.pdb.gz" -> "protein"

    saved = []
    for cid, seq in chains:
        out_path = output_dir / f"{stem}_{cid}.pt"

        if out_path.exists():
            log.info("Skipping %s (already exists)", out_path)
            saved.append(out_path)
            continue

        try:
            result = extractor.extract(seq)
        except Exception as e:
            log.error("Failed to extract embeddings for %s chain %s: %s", filepath, cid, e)
            continue

        mean_plddt = result["plddt"].mean().item()
        if mean_plddt < min_plddt:
            log.info("Skipping %s chain %s (mean pLDDT %.1f < %.1f)", stem, cid, mean_plddt, min_plddt)
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
        log.info("Saved %s (L=%d, pLDDT=%.1f, pTM=%.3f)", out_path.name, len(seq), mean_plddt, result["ptm"])
        saved.append(out_path)

    return saved

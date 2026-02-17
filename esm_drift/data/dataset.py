"""PyTorch Dataset for loading pre-extracted ESMFold embeddings."""

import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """Dataset that loads pre-extracted ESMFold embeddings from .pt files.

    Each .pt file contains:
        - s_s: [L, 1024] single representation
        - s_z: [L, L, 128] pair representation
        - sequence: amino acid string
        - plddt: [L] per-residue confidence
        - ptm: scalar
        - seq_len: int

    Since proteins have variable lengths, this dataset returns individual
    samples without batching. Use a custom collate_fn for batched training
    (e.g., pad or crop to fixed length).
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_seq_len: int | None = None,
        min_ptm: float = 0.0,
        min_plddt: float = 0.0,
    ):
        """
        Args:
            data_dir: Directory containing .pt embedding files.
            max_seq_len: If set, skip proteins longer than this.
            min_ptm: Skip proteins with pTM below this threshold.
            min_plddt: Skip proteins with mean pLDDT below this threshold.
        """
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.min_ptm = min_ptm
        self.min_plddt = min_plddt

        all_files = sorted(self.data_dir.glob("*.pt"))
        self.files = self._filter_files(all_files)
        log.info("Loaded %d / %d embedding files from %s", len(self.files), len(all_files), data_dir)

    def _filter_files(self, files: list[Path]) -> list[Path]:
        """Apply filtering criteria by loading metadata from each file."""
        valid = []
        for f in files:
            try:
                data = torch.load(f, map_location="cpu", weights_only=False)
            except Exception as e:
                log.warning("Could not load %s: %s", f, e)
                continue

            seq_len = data.get("seq_len", len(data.get("sequence", "")))
            ptm = data.get("ptm", 1.0)
            plddt = data.get("plddt", None)

            if self.max_seq_len and seq_len > self.max_seq_len:
                continue
            if ptm < self.min_ptm:
                continue
            if plddt is not None and plddt.mean().item() < self.min_plddt:
                continue

            valid.append(f)
        return valid

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return {
            "s_s": data["s_s"],             # [L, 1024]
            "s_z": data["s_z"],             # [L, L, 128]
            "sequence": data["sequence"],
            "seq_len": data["seq_len"],
            "plddt": data["plddt"],         # [L]
            "ptm": data["ptm"],
        }


def pad_collate(batch: list[dict], max_len: int | None = None) -> dict:
    """Collate variable-length embeddings into a padded batch.

    Args:
        batch: List of dicts from EmbeddingDataset.__getitem__.
        max_len: If set, crop sequences to this length. Otherwise use the
                 longest sequence in the batch.

    Returns:
        Dict with padded tensors and a mask:
            - s_s: [B, L_max, 1024]
            - s_z: [B, L_max, L_max, 128]
            - mask: [B, L_max] boolean (True = valid residue)
            - seq_lens: [B] original lengths
    """
    seq_lens = [item["seq_len"] for item in batch]
    L_max = max(seq_lens)
    if max_len is not None:
        L_max = min(L_max, max_len)

    B = len(batch)
    s_s_dim = batch[0]["s_s"].shape[-1]
    s_z_dim = batch[0]["s_z"].shape[-1]

    s_s_padded = torch.zeros(B, L_max, s_s_dim)
    s_z_padded = torch.zeros(B, L_max, L_max, s_z_dim)
    mask = torch.zeros(B, L_max, dtype=torch.bool)

    for i, item in enumerate(batch):
        L = min(item["seq_len"], L_max)
        s_s_padded[i, :L] = item["s_s"][:L]
        s_z_padded[i, :L, :L] = item["s_z"][:L, :L]
        mask[i, :L] = True

    return {
        "s_s": s_s_padded,
        "s_z": s_z_padded,
        "mask": mask,
        "seq_lens": torch.tensor(seq_lens),
    }

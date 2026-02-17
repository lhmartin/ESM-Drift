"""Utilities for decoding ESMFold embeddings back to 3D structures."""

import logging
from pathlib import Path

import torch

log = logging.getLogger(__name__)

# AlphaFold2/OpenFold amino acid ordering (used by ESMFold internally)
RESTYPES = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_ORDER = {aa: i for i, aa in enumerate(RESTYPES)}
UNK_RESTYPE_INDEX = 20  # 'X'


def sequence_to_aatype(sequence: str) -> torch.Tensor:
    """Convert amino acid sequence string to aatype index tensor.

    Uses the AlphaFold2/OpenFold residue ordering that ESMFold expects internally.
    Unknown residues map to index 20 ('X').

    Returns:
        LongTensor of shape [L]
    """
    return torch.tensor(
        [RESTYPE_ORDER.get(aa, UNK_RESTYPE_INDEX) for aa in sequence],
        dtype=torch.long,
    )


class StructureDecoder:
    """Decode ESMFold embeddings (s_s, s_z) back to 3D protein structures.

    Loads the ESMFold model and runs the folding trunk + structure module
    on provided embeddings to produce atom coordinates and PDB files.

    Usage:
        decoder = StructureDecoder(device="cuda")

        # Share model with an existing EmbeddingExtractor to avoid loading twice:
        decoder = StructureDecoder(model=extractor._model)

        # From a saved embedding file
        pdb_string = decoder.decode_embedding_file("data/embeddings/1abc_A.pt")

        # From tensors (e.g., model-generated)
        pdb_string = decoder.decode(s_s, sequence="MLKN...")

        # Save directly to file
        decoder.save_pdb(s_s, "output.pdb", sequence="MLKN...")
    """

    def __init__(self, device: str = "cuda", model=None):
        self.device = torch.device(device)
        self._model = model

    def _load_model(self):
        if self._model is not None:
            return

        from transformers import EsmForProteinFolding

        log.info("Loading ESMFold model for structure decoding...")
        self._model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self._model = self._model.to(self.device)
        self._model.eval()
        log.info("ESMFold loaded on %s", self.device)

    @torch.no_grad()
    def decode(
        self,
        s_s: torch.Tensor,
        s_z: torch.Tensor | None = None,
        sequence: str | None = None,
        num_recycles: int | None = None,
    ) -> dict:
        """Decode embeddings through ESMFold's folding trunk to get structure.

        Args:
            s_s: Single representation [L, 1024] or [1, L, 1024].
            s_z: Pair representation [L, L, 128] or [1, L, L, 128].
                 If None, initialized to zeros.
            sequence: Amino acid sequence for sidechain prediction.
                      If None, uses poly-glycine (backbone only).
            num_recycles: Number of recycling passes. None = model default.

        Returns:
            Dict with model outputs including 'positions', 'plddt', 'ptm',
            and all fields needed for PDB conversion.
        """
        self._load_model()
        cfg = self._model.config.esmfold_config

        # Handle dimensions
        if s_s.dim() == 2:
            s_s = s_s.unsqueeze(0)  # [1, L, 1024]
        L = s_s.shape[1]

        if s_z is None:
            s_z = torch.zeros(1, L, L, cfg.trunk.pairwise_state_dim)
        elif s_z.dim() == 3:
            s_z = s_z.unsqueeze(0)  # [1, L, L, 128]

        s_s = s_s.to(self.device)
        s_z = s_z.to(self.device)

        # Build aatype from sequence or default to poly-glycine
        if sequence is not None:
            aa = sequence_to_aatype(sequence[:L]).unsqueeze(0).to(self.device)
        else:
            # Glycine (index 7) - simplest AA, gives backbone-only
            aa = torch.full((1, L), 7, dtype=torch.long, device=self.device)

        mask = torch.ones(1, L, device=self.device)
        residx = torch.arange(L, device=self.device).unsqueeze(0)

        # Run the folding trunk (Evoformer + structure module)
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            structure = self._model.trunk(s_s, s_z, aa, residx, mask, no_recycles=num_recycles)

        # Filter to expected keys
        structure = {
            k: v for k, v in structure.items()
            if k in ["s_z", "s_s", "frames", "sidechain_frames",
                      "unnormalized_angles", "angles", "positions", "states"]
        }

        # Post-processing (same as EsmForProteinFolding.forward)
        from transformers.models.esm.openfold_utils.data_transforms import make_atom14_masks

        structure["aatype"] = aa
        make_atom14_masks(structure)

        for k in ["atom14_atom_exists", "atom37_atom_exists"]:
            structure[k] *= mask.unsqueeze(-1)
        structure["residue_index"] = residx

        # Confidence scores (in float32 to avoid NaN in loss/pTM computation)
        B = 1
        lddt_head = self._model.lddt_head(structure["states"].float())
        lddt_head = lddt_head.reshape(
            structure["states"].shape[0], B, L, -1, self._model.lddt_bins
        )
        structure["lddt_head"] = lddt_head

        from transformers.models.esm.modeling_esmfold import categorical_lddt

        plddt = categorical_lddt(lddt_head[-1], bins=self._model.lddt_bins)
        structure["plddt"] = plddt

        # pTM score
        from transformers.models.esm.openfold_utils.loss import (
            compute_predicted_aligned_error,
            compute_tm,
        )

        ptm_logits = self._model.ptm_head(structure["s_z"].float())
        structure["ptm_logits"] = ptm_logits
        structure["ptm"] = compute_tm(
            ptm_logits, max_bin=31, no_bins=self._model.distogram_bins
        )
        structure.update(
            compute_predicted_aligned_error(
                ptm_logits, max_bin=31, no_bins=self._model.distogram_bins
            )
        )

        return structure

    def to_pdb(self, structure: dict) -> str:
        """Convert decode() output to a PDB format string.

        Args:
            structure: Output dict from decode().

        Returns:
            PDB file content as a string.
        """
        from transformers import EsmForProteinFolding

        pdbs = EsmForProteinFolding.output_to_pdb(structure)
        return pdbs[0]

    def decode_embedding_file(self, path: str | Path) -> str:
        """Load a .pt embedding file and decode to PDB string.

        Args:
            path: Path to .pt file saved by EmbeddingExtractor.

        Returns:
            PDB file content as a string.
        """
        data = torch.load(path, map_location="cpu", weights_only=False)
        structure = self.decode(
            s_s=data["s_s"],
            s_z=data.get("s_z"),
            sequence=data.get("sequence"),
        )
        return self.to_pdb(structure)

    def save_pdb(
        self,
        s_s: torch.Tensor,
        output_path: str | Path,
        s_z: torch.Tensor | None = None,
        sequence: str | None = None,
        num_recycles: int | None = None,
    ) -> Path:
        """Decode embeddings and save as a PDB file.

        Args:
            s_s: Single representation [L, 1024].
            output_path: Where to save the PDB file.
            s_z: Optional pair representation [L, L, 128].
            sequence: Optional amino acid sequence.
            num_recycles: Optional number of recycling passes.

        Returns:
            Path to the saved PDB file.
        """
        structure = self.decode(s_s, s_z=s_z, sequence=sequence, num_recycles=num_recycles)
        pdb_string = self.to_pdb(structure)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(pdb_string)
        log.info("Saved PDB to %s", output_path)
        return output_path

    def save_pdb_from_file(self, embedding_path: str | Path, output_path: str | Path) -> Path:
        """Load an embedding .pt file and save the decoded structure as PDB.

        Args:
            embedding_path: Path to .pt embedding file.
            output_path: Where to save the PDB file.

        Returns:
            Path to the saved PDB file.
        """
        pdb_string = self.decode_embedding_file(embedding_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(pdb_string)
        log.info("Saved PDB to %s", output_path)
        return output_path

"""Training and model configuration.

Single source of truth for all hyperparameters. Load from YAML, override via
CLI, save with every checkpoint.

Usage:
    # Load a config file
    cfg = Config.from_yaml("configs/v14.yaml")

    # Start from defaults and override specific fields
    cfg = Config(lr=1e-3, epochs=20000, save_dir="checkpoints/v14-hi-lr")

    # Save / load
    cfg.save_yaml("checkpoints/v14-full/config.yaml")
    cfg = Config.from_yaml("checkpoints/v14-full/config.yaml")
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """All settings for a training run, grouped by concern."""

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir: str = "data/"

    # ── Model architecture ────────────────────────────────────────────────────
    d_noise: int = 256
    d_model: int = 512
    d_bottleneck: int = 128
    nhead: int = 8
    num_layers: int = 4          # enc_layers = dec_layers = num_layers
    max_len: int = 128
    use_length_cond: bool = True

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 5000
    batch_size: int = 32
    lr: float = 3e-4
    max_grad_norm: float = 2.0
    eta_min: float = 1e-5
    eval_every: int = 200

    # ── LR schedule ───────────────────────────────────────────────────────────
    warmup_T0: int = 2000        # 0 = plain cosine, no restarts
    warmup_T_mult: int = 2       # 1 = fixed period, 2 = doubling cycles

    # ── Kernel ────────────────────────────────────────────────────────────────
    taus: list[float] | None = None   # None = auto-calibrate from data
    tau_recal_every: int = 1000

    # ── Loss weights ──────────────────────────────────────────────────────────
    seq_ce_weight: float = 0.05
    prot_drift_weight: float = 0.5   # 0.0 = disabled

    # ── Ablation flags ────────────────────────────────────────────────────────
    use_strict_antisymmetry: bool = True
    use_dynamic_len: bool = True

    # ── Infrastructure ────────────────────────────────────────────────────────
    device: str = "cuda"
    save_dir: str = "checkpoints"
    wandb_project: str = "esm-drift"
    wandb_name: str | None = None
    no_wandb: bool = False

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        """Create a Config from a dict, ignoring unknown keys for backwards compat."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    def save_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

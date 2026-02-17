"""Generator and feature encoder for the drifting method."""

import math

import torch
import torch.nn as nn


class FeatureEncoder(nn.Module):
    """Projects per-residue s_s embeddings to a fixed-size feature vector.

    Used to compute the kernel in the drifting field. Should be pretrained
    (e.g., reconstruction objective) and frozen during drifting training.

    Architecture: per-residue MLP → masked mean pool → projection.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.per_residue = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.project = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),  # keeps feature norms ~sqrt(output_dim)
        )

    def forward(self, s_s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s_s: [B, L, input_dim] single representations
            mask: [B, L] boolean, True = valid residue

        Returns:
            [B, output_dim] pooled feature vectors
        """
        h = self.per_residue(s_s)  # [B, L, hidden_dim]
        # Masked mean pool
        h = h * mask.unsqueeze(-1).float()
        h = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return self.project(h)  # [B, output_dim]


class DriftingGenerator(nn.Module):
    """Transformer that maps noise to s_s embeddings.

    Input: Gaussian noise [B, L, d_noise]
    Output: s_s [B, L, s_s_dim]

    Uses a standard Transformer encoder with sinusoidal positional encoding.
    The mask ensures padded positions output zeros.
    """

    def __init__(
        self,
        d_noise: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        s_s_dim: int = 1024,
        dropout: float = 0.0,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_noise = d_noise
        self.d_model = d_model

        self.input_proj = nn.Linear(d_noise, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.s_s_head = nn.Linear(d_model, s_s_dim)

        # Skip connection: noise → output, ensuring per-position diversity
        # even if the Transformer averages positions via self-attention
        self.noise_skip = nn.Linear(d_noise, s_s_dim, bias=False)

        # Learnable output scale — default Linear init gives output norm ~2,
        # but ESMFold s_s has norms ~4000. This scalar closes the gap.
        self.output_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: [B, L, d_noise] sampled from N(0, I)
            mask: [B, L] boolean, True = valid residue

        Returns:
            s_s: [B, L, s_s_dim] generated single representations
        """
        h = self.input_proj(noise)  # [B, L, d_model]
        h = h + self.pos_enc(h)

        # Transformer expects mask where True = IGNORE
        h = self.transformer(h, src_key_padding_mask=~mask)
        h = self.output_norm(h)
        s_s = self.s_s_head(h)  # [B, L, s_s_dim]

        # Add noise skip connection for per-position diversity
        s_s = s_s + self.noise_skip(noise)

        # Learnable scale
        s_s = s_s * self.output_scale

        # Zero out padded positions
        s_s = s_s * mask.unsqueeze(-1).float()
        return s_s

    def sample_noise(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """Sample input noise from N(0, I)."""
        return torch.randn(batch_size, seq_len, self.d_noise, device=device)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]

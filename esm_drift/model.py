"""Generator and sequence head for the drifting method."""

import math

import torch
import torch.nn as nn


class DriftingGeneratorUNet(nn.Module):
    """U-Net generator: noise [B, L, d_noise] → s_s [B, L, 1024].

    The U-Net principle is applied to the feature dimension (not spatial
    resolution): per-residue features are compressed stage-by-stage in the
    encoder and expanded back with skip connections in the decoder.
    Sequence length L is unchanged throughout.

        noise [B, L, d_noise]
          → input_proj + pos_enc  → [B, L, d_model]
          ┌──────────────────────────────────────────────────────────────┐
          │  Enc-1: Transformer (enc1_layers)   [B, L, d_model]         │── skip1
          │  down1: Linear + LN                 [B, L, d_mid]           │
          │  Enc-2: Transformer (enc2_layers)   [B, L, d_mid]           │── skip2
          │  down2: Linear + LN                 [B, L, d_bottleneck]    │
          │  Bottleneck: Transformer (1 layer)  [B, L, d_bottleneck]    │
          │  up2:   Linear                      [B, L, d_mid]           │
          │  merge2: cat(up2, skip2) → Linear   [B, L, d_mid]           │
          │  Dec-2: Transformer (dec2_layers)   [B, L, d_mid]           │
          │  up1:   Linear                      [B, L, d_model]         │
          │  merge1: cat(up1, skip1) → Linear   [B, L, d_model]         │
          │  Dec-1: Transformer (dec1_layers)   [B, L, d_model]         │
          └──────────────────────────────────────────────────────────────┘
          → output_norm → s_s_head → [B, L, s_s_dim]
          → + noise_skip (per-position diversity)
          → × output_scale (learnable)

    d_mid is automatically computed as the midpoint between d_model and
    d_bottleneck, rounded up to the nearest multiple of nhead.
    """

    def __init__(
        self,
        d_noise: int = 256,
        d_model: int = 512,
        d_bottleneck: int = 128,
        nhead: int = 8,
        enc_layers: int = 3,
        dec_layers: int = 3,
        s_s_dim: int = 1024,
        dropout: float = 0.0,
        max_len: int = 512,
    ):
        super().__init__()
        self.d_noise = d_noise
        self.d_model = d_model
        self.d_bottleneck = d_bottleneck
        self.nhead = nhead
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.num_layers = enc_layers + dec_layers

        # Intermediate dimension: midpoint rounded up to nhead multiple
        d_mid = (d_model + d_bottleneck + 1) // 2
        d_mid = ((d_mid + nhead - 1) // nhead) * nhead
        self.d_mid = d_mid

        # Split enc/dec layers evenly between the two stages
        enc1_layers = max(1, enc_layers - enc_layers // 2)
        enc2_layers = max(1, enc_layers // 2)
        dec2_layers = max(1, dec_layers - dec_layers // 2)
        dec1_layers = max(1, dec_layers // 2)

        def _nhead_for(d: int) -> int:
            """Largest power of 2 ≤ nhead that evenly divides d."""
            n = nhead
            while n > 1 and d % n != 0:
                n //= 2
            return max(1, n)

        def _transformer(d: int, n: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=_nhead_for(d), dim_feedforward=d * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=n)

        self.input_proj = nn.Linear(d_noise, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)

        # Encoder: d_model → d_mid → d_bottleneck
        self.enc1 = _transformer(d_model, enc1_layers)
        self.down1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_mid))
        self.enc2 = _transformer(d_mid, enc2_layers)
        self.down2 = nn.Sequential(nn.LayerNorm(d_mid), nn.Linear(d_mid, d_bottleneck))

        # Bottleneck: single Transformer at the compressed dimension
        self.bottleneck = _transformer(d_bottleneck, 1)

        # Decoder: d_bottleneck → d_mid → d_model (each with skip connection)
        self.up2       = nn.Linear(d_bottleneck, d_mid)
        self.merge2    = nn.Sequential(nn.Linear(d_mid * 2, d_mid), nn.LayerNorm(d_mid))
        self.dec2      = _transformer(d_mid, dec2_layers)

        self.up1       = nn.Linear(d_mid, d_model)
        self.merge1    = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model))
        self.dec1      = _transformer(d_model, dec1_layers)

        self.output_norm = nn.LayerNorm(d_model)
        self.s_s_head    = nn.Linear(d_model, s_s_dim)

        self.noise_skip  = nn.Linear(d_noise, s_s_dim, bias=False)
        self.output_scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, noise: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: [B, L, d_noise]
            mask:  [B, L] boolean, True = valid residue
        Returns:
            s_s: [B, L, s_s_dim]
        """
        pad_mask = ~mask  # Transformer convention: True = ignore

        h = self.input_proj(noise) + self.pos_enc(noise)  # [B, L, d_model]

        # Encoder — compress feature dimension stage by stage
        e1 = self.enc1(h,          src_key_padding_mask=pad_mask)  # [B, L, d_model]  ← skip1
        e2 = self.enc2(self.down1(e1), src_key_padding_mask=pad_mask)  # [B, L, d_mid]  ← skip2
        z  = self.bottleneck(self.down2(e2), src_key_padding_mask=pad_mask)  # [B, L, d_bottleneck]

        # Decoder — expand back with skip connections
        d2 = self.merge2(torch.cat([self.up2(z),  e2], dim=-1))  # [B, L, d_mid]
        d2 = self.dec2(d2, src_key_padding_mask=pad_mask)

        d1 = self.merge1(torch.cat([self.up1(d2), e1], dim=-1))  # [B, L, d_model]
        d1 = self.dec1(d1, src_key_padding_mask=pad_mask)
        d1 = self.output_norm(d1)

        s_s = self.s_s_head(d1)
        s_s = s_s + self.noise_skip(noise)
        s_s = s_s * self.output_scale
        s_s = s_s * mask.unsqueeze(-1).float()
        return s_s

    def sample_noise(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, seq_len, self.d_noise, device=device)


class SeqHead(nn.Module):
    """Linear head: s_s [*, 1024] → amino acid logits [*, 20].

    Trained on real (s_s, sequence) pairs via cross-entropy. At inference,
    argmax gives a sequence consistent with the generated embedding, which
    produces better pLDDT than poly-alanine when passed to ESMFold.
    """

    N_AA = 20  # standard amino acids (ARNDCQEGHILKMFPSTWYV)

    def __init__(self, s_s_dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(s_s_dim, self.N_AA)

    def forward(self, s_s: torch.Tensor) -> torch.Tensor:
        """Args:
            s_s: [..., s_s_dim]
        Returns:
            logits: [..., 20]
        """
        return self.proj(s_s)


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

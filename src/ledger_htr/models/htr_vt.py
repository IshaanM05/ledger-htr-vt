import math

import torch
import torch.nn as nn

from ledger_htr.models.cnn_stem import HTRConvStem


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def sincos_position_embedding(num_positions: int, dim: int) -> torch.Tensor:
    """Fixed (non-learned) 1D sinusoidal position embedding. The CNN stem
    collapses height to a single row, so the sequence fed to the Transformer
    is already 1D (indexed along width/time) -- no 2D grid embedding needed."""
    position = torch.arange(num_positions).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_positions, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class HTRViT(nn.Module):
    """CNN stem + Transformer encoder + CTC head, with span-masking
    regularization at train time -- the general recipe of Li et al. 2024
    ("HTR-VT: Handwritten Text Recognition with Vision Transformer"):
    a ViT-style encoder over CNN features, trained with a CTC loss, span
    masking, and (in the training loop, see train_htrvt.py) a SAM optimizer.

    Written independently from the paper's method description (see
    docs/plan.md's Learning curriculum for the reference) rather than the
    authors' reference implementation, which ships without a LICENSE file
    (see docs/plan.md's License audit) -- architecture and hyperparameter
    choices below are this project's own, not a port of their code."""

    def __init__(
        self,
        nb_classes: int,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = HTRConvStem(embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, nb_classes)
        self._pos_cache: torch.Tensor | None = None

    def _get_pos_embed(self, n: int, device, dtype) -> torch.Tensor:
        if self._pos_cache is None or self._pos_cache.shape[1] != n or self._pos_cache.device != device:
            pe = sincos_position_embedding(n, self.embed_dim).to(device=device, dtype=dtype)
            self._pos_cache = pe.unsqueeze(0)
        return self._pos_cache

    def _span_mask(self, x: torch.Tensor, mask_ratio: float, max_span_length: int) -> torch.Tensor:
        b, n, _c = x.shape
        if mask_ratio <= 0 or max_span_length <= 0:
            return x
        num_masked_total = int(n * mask_ratio)
        num_spans = max(num_masked_total // max_span_length, 0)
        mask = torch.ones(b, n, 1, device=x.device, dtype=x.dtype)
        span = min(max_span_length, n)
        for _ in range(num_spans):
            start = int(torch.randint(0, max(n - span, 1), (1,)).item())
            mask[:, start : start + span, :] = 0.0
        return x * mask + (1 - mask) * self.mask_token

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.0,
        max_span_length: int = 1,
        use_masking: bool = False,
    ) -> torch.Tensor:
        feat = self.stem(x)  # (B, C, H', W')
        b, c, h, w = feat.shape
        seq = feat.reshape(b, c, h * w).permute(0, 2, 1)  # (B, N, C)
        if use_masking:
            seq = self._span_mask(seq, mask_ratio, max_span_length)
        seq = seq + self._get_pos_embed(seq.shape[1], seq.device, seq.dtype)
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)
        return self.head(seq)


def create_model(nb_classes: int) -> HTRViT:
    return HTRViT(nb_classes=nb_classes, embed_dim=768, depth=4, num_heads=6, mlp_ratio=4.0)


class HTRMaskedAutoencoder(nn.Module):
    """Masked-image-modeling pretraining model: the same CNN stem +
    Transformer trunk as HTRViT (matching submodule names -- `stem`,
    `blocks`, `norm` -- so `load_pretrained_encoder` below can transfer
    weights by simple name match), plus a linear reconstruction head instead
    of a CTC head. Masking happens at the pixel level (see
    ledger_htr.data.pretrain_dataset.mask_image_strips) rather than the
    feature-level span-masking HTRViT uses for supervised regularization,
    since pixel-level masking gives an unambiguous reconstruction target
    without needing exact patch-to-pixel alignment through the CNN stem's
    overlapping receptive fields.

    Deliberately NOT sharing a common base class with HTRViT: doing so would
    nest these submodules under a shared `encoder.*` prefix and change
    HTRViT's state_dict key names, breaking compatibility with checkpoints
    already trained against the current key layout (see
    docs/HTRVT_FOLD0_RUN_REPORT.md). A little duplication here is the
    deliberate trade for not invalidating that checkpoint."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        patch_width: int = 4,
        target_height: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_width = patch_width
        self.target_height = target_height
        self.stem = HTRConvStem(embed_dim)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.recon_head = nn.Linear(embed_dim, target_height * patch_width)
        self._pos_cache: torch.Tensor | None = None

    def _get_pos_embed(self, n: int, device, dtype) -> torch.Tensor:
        if self._pos_cache is None or self._pos_cache.shape[1] != n or self._pos_cache.device != device:
            pe = sincos_position_embedding(n, self.embed_dim).to(device=device, dtype=dtype)
            self._pos_cache = pe.unsqueeze(0)
        return self._pos_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)  # (B, C, H', W'), H'=1, W'=W/4
        b, c, h, w = feat.shape
        seq = feat.reshape(b, c, h * w).permute(0, 2, 1)  # (B, N, C)
        seq = seq + self._get_pos_embed(seq.shape[1], seq.device, seq.dtype)
        for block in self.blocks:
            seq = block(seq)
        seq = self.norm(seq)
        recon = self.recon_head(seq)  # (B, N, target_height*patch_width)
        n = recon.shape[1]
        recon = recon.reshape(b, n, self.target_height, self.patch_width).permute(0, 2, 1, 3)
        recon = recon.reshape(b, self.target_height, n * self.patch_width)
        return recon.unsqueeze(1)  # (B, 1, target_height, W) -- same shape as the input image


def load_pretrained_encoder(model: HTRViT, checkpoint_path: str, device: str = "cpu") -> int:
    """Copy the CNN stem + Transformer blocks + final norm from a
    HTRMaskedAutoencoder pretraining checkpoint into a fresh HTRViT, leaving
    `mask_token` and `head` at their random initialization (mask_token is
    specific to supervised span-masking, and head is the task-specific
    output layer -- neither has a pretraining counterpart to transfer).
    Returns the number of tensors actually transferred, so the caller can
    sanity-check it's nonzero rather than silently no-op'ing on a key
    mismatch."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model"] if "model" in checkpoint else checkpoint
    own_state = model.state_dict()
    transferred = 0
    for key, tensor in source_state.items():
        if key in own_state and (key.startswith("stem.") or key.startswith("blocks.") or key.startswith("norm.")):
            own_state[key] = tensor
            transferred += 1
    model.load_state_dict(own_state)
    return transferred

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 43
    seq_len: int = 78
    n_moves: int = 1968
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
    # LLaMA-ish hidden sizing: ~ 8/3 * d_model is common for SwiGLU
    mlp_hidden_mul: float = 8.0 / 3.0

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = (x / rms).type_as(x) * self.weight
        return x

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.fc12 = nn.Linear(d_model, hidden_dim * 2, bias=False)
        self.fc3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.fc12(x).chunk(2, dim=-1)
        x = F.silu(a) * b
        x = self.fc3(x)
        return self.dropout(x)

class SelfAttention(nn.Module):
    """
    Not causual
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.attn_drop_p = float(dropout)

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return self.resid_drop(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_hidden: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, mlp_hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DumboBC(nn.Module):
    def __init__(self, cfg: ModelConfig, pad_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        mlp_hidden = int(cfg.mlp_hidden_mul * cfg.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, mlp_hidden, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm_f = RMSNorm(cfg.d_model)

        self.head = nn.Linear(cfg.d_model, cfg.n_moves, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T]
        B, T = tokens.shape
        assert T == self.cfg.seq_len, f"Expected seq_len={self.cfg.seq_len}, got {T}"

        pos = torch.arange(T, device=tokens.device)
        x = self.tok_emb(tokens) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm_f(x)

        cls = x[:, 0, :]          # CLS pooling (you already inserted CLS into tokens)
        logits = self.head(cls)   # [B, 1968]
        return logits
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import constants
import dumbo_bc
import parquet_dataset


DATA_DIRS = ["parquet_out/2025-10-MidGame", "parquet_out/Lichess-Puzzles"]
OUT_PATH = "checkpoints/dumbo_bc.pt"


class FixedMicrobatchSlicer:
    """
    Pulls big CPU batches from a DataLoader iterator and returns *exactly*
    micro rows each time by carrying leftovers across big-batch boundaries.
    """
    def __init__(self, dl, micro: int):
        self.dl = dl
        self.it = iter(dl)

        self.micro = micro
        self.tokens_big = None
        self.moves_big = None
        self.i = 0  # cursor into current big batch

    def _refill(self):
        try:
            self.tokens_big, self.moves_big = next(self.it)
        except StopIteration:
            self.it = iter(self.dl)
            self.tokens_big, self.moves_big = next(self.it)
        self.i = 0

    def next(self):
        if self.tokens_big is None:
            self._refill()

        seq_len = self.tokens_big.size(1)
        tokens_out = torch.empty((self.micro, seq_len), dtype=self.tokens_big.dtype, pin_memory=True)
        moves_out  = torch.empty((self.micro,), dtype=self.moves_big.dtype, pin_memory=True)

        filled = 0
        while filled < self.micro:
            if self.tokens_big is None:
                self._refill()

            n = self.tokens_big.size(0)
            if self.i >= n:
                self.tokens_big = None
                self.moves_big = None
                continue

            take = min(self.micro - filled, n - self.i)

            tokens_out[filled:filled+take].copy_(self.tokens_big[self.i:self.i+take])
            moves_out[filled:filled+take].copy_(self.moves_big[self.i:self.i+take])

            self.i += take
            filled += take

        return tokens_out, moves_out


@dataclass
class TrainConfig:
    steps: int = 200_000
    global_batch: int = 16_384
    microbatch_max: int = 8_192  # kept for compatibility; we still pick microbatch_size explicitly below

    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2_000

    log_every: int = 1
    save_every: int = 500

    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1

    num_workers: int = 4
    prefetch_factor: int = 2

    grad_clip: float = 1.0

    parquet_batch_rows: int = 16_384
    shuffle_buffer_batches: int = 64


def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    """
    Linear warmup then cosine decay to 10% of base lr.
    step is the *global* step index (0-based).
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))


def _atomic_save(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _filter_cfg_keys(cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    If your dumbo_bc.ModelConfig changes over time, this avoids crashing on extra keys.
    """
    ann = getattr(dumbo_bc.ModelConfig, "__annotations__", {})
    if not ann:
        return dict(cfg_dict)
    return {k: v for k, v in cfg_dict.items() if k in ann}


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    train_cfg = TrainConfig()

    if not os.path.exists(OUT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {OUT_PATH}")

    # -------------------------
    # Load checkpoint
    # -------------------------
    ckpt = torch.load(OUT_PATH, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "cfg" not in ckpt:
        raise ValueError(f"{OUT_PATH} is not a valid checkpoint (expected keys: cfg, state_dict, step).")

    start_step = int(ckpt.get("step", 0))  # number of steps already completed
    if start_step >= train_cfg.steps:
        print(f"Checkpoint already at step={start_step} >= train_cfg.steps={train_cfg.steps}. Nothing to do.")
        return

    cfg_dict = ckpt["cfg"]
    if not isinstance(cfg_dict, dict):
        raise ValueError("Checkpoint cfg is not a dict (expected asdict(cfg) format).")

    cfg_dict = _filter_cfg_keys(cfg_dict)
    cfg = dumbo_bc.ModelConfig(**cfg_dict)

    model = dumbo_bc.DumboBC(cfg).to(device)
    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
    if missing or unexpected:
        print("[warn] load_state_dict(strict=False) mismatches:")
        if missing:
            print("  missing:", missing[:25], "..." if len(missing) > 25 else "")
        if unexpected:
            print("  unexpected:", unexpected[:25], "..." if len(unexpected) > 25 else "")

    print(f"Loaded checkpoint: {OUT_PATH}")
    print(f"Resuming from step {start_step} -> {train_cfg.steps}")
    print(f"Model Config: {cfg_dict}")

    # -------------------------
    # Microbatch / accumulation
    # -------------------------
    microbatch_size = 1024  # keep same behavior you hardcoded before
    grad_accum = max(1, int(math.ceil(train_cfg.global_batch / microbatch_size)))
    effective_batch_size = microbatch_size * grad_accum
    print(f"Effective batch size: {effective_batch_size} (micro={microbatch_size}, grad_accum={grad_accum})")

    # -------------------------
    # Data loader
    # -------------------------
    stream_cfg = parquet_dataset.ParquetStreamConfig(
        seq_len=constants.L,
        batch_rows=train_cfg.parquet_batch_rows,
        shuffle_files=True,
        shuffle_buffer_batches=train_cfg.shuffle_buffer_batches,
    )

    ds = parquet_dataset.TokenMoveParquetDataset(DATA_DIRS, stream_cfg)
    dl = DataLoader(
        ds,
        batch_size=None,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        persistent_workers=(train_cfg.num_workers > 0),
        prefetch_factor=train_cfg.prefetch_factor if train_cfg.num_workers > 0 else None,
    )
    slicer = FixedMicrobatchSlicer(dl, micro=microbatch_size)
    print("Data Loaded!")

    # -------------------------
    # Optimizer (resume if possible)
    # -------------------------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=train_cfg.weight_decay,
    )

    if "opt_state_dict" in ckpt:
        try:
            opt.load_state_dict(ckpt["opt_state_dict"])
            print("[resume] loaded optimizer state")
        except Exception as e:
            print(f"[warn] failed to load optimizer state; continuing fresh: {e}")
    else:
        print("[resume] no optimizer state in checkpoint (first resume will be weights-only)")

    ema_loss: Optional[float] = ckpt.get("ema_loss", None)

    # -------------------------
    # Train loop
    # -------------------------
    model.train()
    t_log0 = time.time()
    steps_in_window = 0

    print("Started Model Training (RESUME)!")
    for step in range(start_step, train_cfg.steps):
        lr = lr_at(step, train_cfg.lr, train_cfg.warmup_steps, train_cfg.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(grad_accum):
            tokens, moves = slicer.next()

            tokens = tokens.to(device, non_blocking=True, dtype=torch.long)
            moves  = moves.to(device, non_blocking=True, dtype=torch.long)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(tokens)
                loss = F.cross_entropy(logits, moves) / grad_accum

            loss.backward()
            total_loss += float(loss.item())

        if train_cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        opt.step()

        if ema_loss is None:
            ema_loss = total_loss
        else:
            ema_loss = ema_loss * 0.98 + total_loss * 0.02

        steps_in_window += 1

        if (step + 1) % train_cfg.log_every == 0 or step == start_step:
            torch.cuda.synchronize()
            dt = time.time() - t_log0
            steps_per_s = steps_in_window / max(dt, 1e-9)
            examples_per_s = steps_per_s * effective_batch_size
            mem = torch.cuda.max_memory_allocated() / (1024**3)

            print(
                f"step {step+1:>7}/{train_cfg.steps} "
                f"loss={total_loss:.4f} ema={ema_loss:.4f} "
                f"lr={lr:.2e} "
                f"ex/s={examples_per_s:,.0f} "
                f"max_mem={mem:.1f}GB"
            )

            t_log0 = time.time()
            steps_in_window = 0

        if (step + 1) % train_cfg.save_every == 0:
            out = {
                "cfg": cfg_dict,  # keep same stored cfg format (dict)
                "state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),  # NEW: makes future resumes exact
                "step": step + 1,
                "ema_loss": ema_loss,
            }
            _atomic_save(out, OUT_PATH)
            print(f"[saved] {OUT_PATH}")

    # final save
    out = {
        "cfg": cfg_dict,
        "state_dict": model.state_dict(),
        "opt_state_dict": opt.state_dict(),
        "step": train_cfg.steps,
        "ema_loss": ema_loss,
    }
    _atomic_save(out, OUT_PATH)
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()

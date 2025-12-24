import math
import os 
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import constants
import dumbo_bc
import parquet_dataset

DATA_DIRS = ["parquet_out/2025-09"]
OUT_PATH = "checkpoints/dumbo_bc.pt"

class FixedMicrobatchSlicer:
    """
    Pulls big CPU batches from a DataLoader iterator and returns *exactly*
    micro rows each time by carrying leftovers across big-batch boundaries.

    This prevents wasting (batch_rows - micro) rows and avoids ragged microbatches.
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
            # restart the dataloader iteration (new epoch / pass)
            self.it = iter(self.dl)
            self.tokens_big, self.moves_big = next(self.it)
        self.i = 0

    def next(self):
        # Lazily allocate pinned output buffers (fast H2D when using non_blocking=True)
        # We allocate *fresh* buffers each call to avoid any chance of overwriting
        # memory while an async H2D copy is still in flight.
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
                # current big batch exhausted, grab next
                self.tokens_big = None
                self.moves_big = None
                continue

            take = min(self.micro - filled, n - self.i)

            # copy slice into output buffers
            tokens_out[filled:filled+take].copy_(self.tokens_big[self.i:self.i+take])
            moves_out[filled:filled+take].copy_(self.moves_big[self.i:self.i+take])

            self.i += take
            filled += take

        return tokens_out, moves_out



@dataclass
class TrainConfig:
    steps: int = 200_000
    global_batch: int = 16384
    microbatch_max: int = 8_192
    
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2_000
    
    log_every: int = 50
    save_every: int = 5_000
    
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.1
    
    num_workers: int = 4
    prefetch_factor: int = 2
    
    grad_clip: float = 1.0
    # Parquet reader batch size (CPU). Independent of microbatch.
    # Keep fairly big for throughput.
    parquet_batch_rows: int = 16384 
    shuffle_buffer_batches: int = 64

def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    """
    Linear warmup then cosine decay to 10% of base lr.
    """
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * t)))


def main():
    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    vocab_size = constants.VOCAB_SIZE
    train_cfg = TrainConfig()

    cfg = dumbo_bc.ModelConfig(
        vocab_size=vocab_size,
        seq_len=constants.L,
        d_model=train_cfg.d_model,
        n_layers=train_cfg.n_layers,
        n_heads=train_cfg.n_heads,
        dropout=train_cfg.dropout,
    )

    model = dumbo_bc.DumboBC(cfg).to(device)
    print(f"Model Config: {asdict(cfg)}")

    microbatch_size = 1024
    grad_accum = max(1, int(math.ceil(train_cfg.global_batch / microbatch_size)))
    effective_batch_size = microbatch_size * grad_accum

    print(f"Effective batch size: {effective_batch_size}")
    
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

    print("Data Loaded!")

    slicer = FixedMicrobatchSlicer(dl, micro=microbatch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, betas=(0.9, 0.95), weight_decay=train_cfg.weight_decay)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    model.train()
    t_log0 = time.time()
    steps_in_window = 0
    ema_loss = None

    print("Started Model Training!")
    for step in range(train_cfg.steps):
        lr = lr_at(step, train_cfg.lr, train_cfg.warmup_steps, train_cfg.steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        
        opt.zero_grad(set_to_none=True)

        total_loss = 0.0
        for _ in range(grad_accum):
            tokens, moves = slicer.next()
            
            tokens = tokens.to(device, non_blocking=True, dtype=torch.long)
            moves = moves.to(device, non_blocking=True, dtype=torch.long)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(tokens)
                loss = F.cross_entropy(logits, moves) / grad_accum
            
            loss.backward()
            total_loss += loss.item()
        
        if train_cfg.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        
        opt.step()

        if ema_loss is None:
            ema_loss = total_loss
        else:
            ema_loss = ema_loss * 0.98 + total_loss * 0.02
            
        steps_in_window += 1
        
        if True:
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

            # reset window
            t_log0 = time.time()
            steps_in_window = 0
        
        if (step + 1) % train_cfg.save_every == 0:
            tmp = OUT_PATH + ".tmp"
            torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict(), "step": step + 1}, tmp)
            os.replace(tmp, OUT_PATH)
            print(f"[saved] {OUT_PATH}")

    # final save
    tmp = OUT_PATH + ".tmp"
    torch.save({"cfg": asdict(cfg), "state_dict": model.state_dict(), "step": train_cfg.steps}, tmp)
    os.replace(tmp, OUT_PATH)
    print(f"[saved] {OUT_PATH}")


if __name__ == "__main__":
    main()
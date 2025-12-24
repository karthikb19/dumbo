from dataclasses import dataclass
from typing import Tuple, List
from pathlib import Path
import random

import numpy as np 
import torch
from torch.utils.data import IterableDataset, get_worker_info

import pyarrow as pa
import pyarrow.parquet as pq

@dataclass
class ParquetStreamConfig:
    seq_len: int = 78
    columns: Tuple[str, ...] = ("tokens", "move_id")
    batch_rows: int = 16384 
    shuffle_files: bool = True
    shuffle_buffer_batches: int = 64 # buffers in flight so thats a total of 16384 * 64 bufffers in flight!


def _list_parquet_files(dirs) -> List[str]:
    files = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            raise FileNotFoundError(f"{d} does not exist")
        files.extend(str(x) for x in p.rglob("*.parquet"))
    
    if not files:
        raise FileNotFoundError("No parquet files found")
    return sorted(files)


def _split_by_worker(files) -> List[str]:
    wi = get_worker_info()
    if wi is None:
        return files
    return files[wi.id::wi.num_workers]


class TokenMoveParquetDataset(IterableDataset):
    def __init__(self, data_dirs, cfg: ParquetStreamConfig):
        super().__init__()
        self.data_dirs = data_dirs
        self.cfg = cfg
        self.files = _list_parquet_files(data_dirs)
    
    def __iter__(self):
        files = _split_by_worker(self.files)
        if self.cfg.shuffle_files:
            random.shuffle(files)
        
        buf_tokens = []
        buf_move_ids = []

        def maybe_yield_from_buffer():
            nonlocal buf_tokens, buf_move_ids
            if len(buf_tokens) >= self.cfg.shuffle_buffer_batches:
                idx = random.randrange(len(buf_tokens))
                t = buf_tokens.pop(idx)
                m = buf_move_ids.pop(idx)
                return (t, m)
            return None

        for path in files:
            pf = pq.ParquetFile(path)
            schema = pf.schema_arrow
            tok_name, move_name = self.cfg.columns
            
            if tok_name not in schema.names or move_name not in schema.names:
                raise ValueError(f"{path}: expected columns {self.cfg.columns}, got {schema.names}")
            
            tok_type = schema.field(tok_name).type
            move_type = schema.field(move_name).type
            # STRICT: require fixed-size binary(seq_len)
            if not pa.types.is_fixed_size_binary(tok_type) or tok_type.byte_width != self.cfg.seq_len:
                raise TypeError(
                    f"{path}: tokens must be fixed_size_binary[{self.cfg.seq_len}] "
                    f"(you wrote pa.binary(constants.L)). Got: {tok_type}"
                )

            if not pa.types.is_uint16(move_type):
                raise TypeError(f"{path}: move_id must be uint16. Got: {move_type}")


            for rb in pf.iter_batches(batch_size=self.cfg.batch_rows, columns=[tok_name, move_name], use_threads=True):
                tok_arr = rb.column(0)
                mv_arr = rb.column(1)
                
                if tok_arr.null_count != 0 or mv_arr.null_count != 0:
                    raise ValueError(f"{path}: nulls found; expected none.")
                
                # ---- tokens: fixed_size_binary ----
                # Handle potential offset safely.
                t_off = tok_arr.offset
                t_len = len(tok_arr)
                t_buf = tok_arr.buffers()[1]
                if t_buf is None:
                    raise RuntimeError(f"{path}: tokens buffer missing")

                # slice bytes for [t_off : t_off+t_len]
                byte_start = t_off * self.cfg.seq_len
                byte_size = t_len * self.cfg.seq_len
                t_view = t_buf.slice(byte_start, byte_size)
                t_np = np.frombuffer(t_view, dtype=np.uint8).reshape(t_len, self.cfg.seq_len).copy()
                tokens = torch.from_numpy(t_np)  # uint8

                # ---- moves: uint16 ----
                m_off = mv_arr.offset
                m_len = len(mv_arr)
                m_buf = mv_arr.buffers()[1]
                if m_buf is None:
                    raise RuntimeError(f"{path}: move buffer missing")

                mv_start = m_off * 2
                mv_size = m_len * 2
                m_view = m_buf.slice(mv_start, mv_size)
                m_np = np.frombuffer(m_view, dtype=np.uint16).copy()
                moves = torch.from_numpy(m_np.astype(np.int64, copy=False))  # CE expects int64

                # chunk into smaller training batches (GPU-friendly)
                # Here we yield exactly cfg.batch_rows at a time; you can further split outside if wanted.
                buf_tokens.append(tokens)
                buf_move_ids.append(moves)

                out = maybe_yield_from_buffer()
                if out is not None:
                    yield out

        # flush buffer
        while buf_tokens:
            idx = random.randrange(len(buf_tokens))
            yield buf_tokens.pop(idx), buf_move_ids.pop(idx)
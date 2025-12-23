import chess.pgn
from pathlib import Path
import time
from multiprocessing import get_context
import pyarrow as pa
import pyarrow.parquet as pq

from typing import List, Dict, Tuple

import constants
import fen_helpers
import uci_helpers
import shard


def _wrap(t):
    return shard.process_shard(*t)

def main():
    shard_paths = sorted(Path("datasets/shards-2025-11").glob("shard_*.pgn"))
    args = [(str(path), "parquet_out/2025-11", 100_000) for path in shard_paths]

    total_games = 0
    total_rows = 0

    # Explicit spawn context (what mac uses anyway)
    with get_context("spawn").Pool(processes=8) as pool:
        results = pool.imap_unordered(_wrap, args, chunksize=1)
        
        num_shards = len(args)
        for i, (g, r) in enumerate(results, 1):
            total_games += g
            total_rows += r
            print(f"[{i}/{num_shards}] Processed shard: {g} games, {r} rows | Total: {total_games} games, {total_rows} rows")

    print(f"Total games: {total_games}")
    print(f"Total rows: {total_rows}")

if __name__ == "__main__":
    main()
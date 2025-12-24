from pathlib import Path
from multiprocessing import get_context

import puzzle_shard


def _wrap(t):
    return puzzle_shard.process_puzzle_shard(*t)


def main():
    # Change these to match your setup
    in_dir = Path("datasets/")
    out_dir = "parquet_out/Lichess-Puzzles"
    batch_rows = 200_000

    # Supports both .csv and .csv.zst
    shard_paths = sorted(list(in_dir.glob("lichess_db_puzzle.csv.zst")))
    if not shard_paths:
        raise FileNotFoundError(f"No puzzle shards found in {in_dir} (expected lichess_db_puzzle.csv.zst)")

    args = [(str(path), out_dir, batch_rows) for path in shard_paths]

    total_puzzles = 0
    total_rows = 0

    with get_context("spawn").Pool(processes=8) as pool:
        results = pool.imap_unordered(_wrap, args, chunksize=1)

        num_shards = len(args)
        for i, (p, r) in enumerate(results, 1):
            total_puzzles += p
            total_rows += r
            print(f"[{i}/{num_shards}] Processed puzzle shard: {p} puzzles, {r} rows | Total: {total_puzzles} puzzles, {total_rows} rows")

    print(f"Total puzzles: {total_puzzles}")
    print(f"Total rows: {total_rows}")


if __name__ == "__main__":
    main()

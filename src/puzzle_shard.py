from __future__ import annotations

from pathlib import Path
from typing import Tuple, List, Optional
import io
import csv

import chess
import pyarrow as pa
import pyarrow.parquet as pq

import constants
import fen_helpers
import uci_helpers


SCHEMA = pa.schema([
    ("tokens", pa.binary(constants.L)),
    ("move_id", pa.uint16()),
])


def _flush(writer: pq.ParquetWriter, tokens_buffer: List[bytes], move_buffer: List[int]) -> int:
    if not tokens_buffer:
        return 0
    table = pa.Table.from_arrays(
        [
            pa.array(tokens_buffer, type=pa.binary(constants.L)),
            pa.array(move_buffer, type=pa.uint16()),
        ],
        schema=SCHEMA,
    )
    writer.write_table(table)
    n = len(tokens_buffer)
    tokens_buffer.clear()
    move_buffer.clear()
    return n


def _open_csv_stream(path: str):
    """
    Returns a text file-like object for CSV reading.
    Supports .csv and .csv.zst (zstandard streaming).
    """
    p = Path(path)
    if p.suffix == ".csv":
        return open(p, "r", encoding="utf-8", errors="replace", newline="")

    # handle .csv.zst
    if p.suffix == ".zst" and p.name.endswith(".csv.zst"):
        import zstandard as zstd  # pip install zstandard
        fh = open(p, "rb")
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace", newline="")
        # attach closers so caller can close everything
        text._zst_reader = reader  # type: ignore[attr-defined]
        text._zst_fh = fh          # type: ignore[attr-defined]
        return text

    raise ValueError(f"Unsupported puzzle file suffix: {p.name} (expected .csv or .csv.zst)")


def _close_csv_stream(f):
    # Close in reverse order if we created a zstd reader stack
    try:
        reader = getattr(f, "_zst_reader", None)
        fh = getattr(f, "_zst_fh", None)
        f.close()
        if reader is not None:
            reader.close()
        if fh is not None:
            fh.close()
    except Exception:
        try:
            f.close()
        except Exception:
            pass


def process_puzzle_shard(
    path: str,
    out_dir: str = "parquet_out/lichess_puzzles",
    batch_rows: int = 200_000,
    solver_moves_only: bool = True,
    strict_legal: bool = False,
    skip_missing_move_ids: bool = True,
) -> Tuple[int, int]:
    """
    Reads a lichess puzzles CSV (optionally zstd-compressed) and writes Parquet with (tokens, move_id).

    Interpretation:
      - row[1] = FEN (position BEFORE the first move in Moves)
      - row[2] = Moves (space-separated UCI list)
      - We push Moves[0] (opponent blunder) first to reach the position presented to the solver.
      - Then:
          * solver_moves_only=True => emit moves at indices 1,3,5,... (solver turns)
          * solver_moves_only=False => emit indices 1..end (both sides continuation)
      - Repetition flag: puzzle dump doesn’t provide history, so rep=0.
    """
    out_path = Path(out_dir) / (Path(path).stem.replace(".csv", "") + ".parquet")
    out_path.parent.mkdir(exist_ok=True, parents=True)

    puzzles = 0
    rows = 0
    tokens_buffer: List[bytes] = []
    move_buffer: List[int] = []

    with pq.ParquetWriter(out_path, SCHEMA, compression="zstd") as writer:
        f = _open_csv_stream(path)
        try:
            reader = csv.reader(f)

            header = next(reader, None)
            if header is None:
                return 0, 0

            for row in reader:
                if len(row) < 3:
                    continue

                puzzles += 1
                fen = row[1]
                moves = row[2].split()
                if len(moves) < 2:
                    continue

                try:
                    board = chess.Board(fen)
                except Exception:
                    continue

                # Apply the initial (opponent) move to reach the puzzle start position for the solver.
                try:
                    if strict_legal:
                        m0 = chess.Move.from_uci(moves[0])
                        if m0 not in board.legal_moves:
                            continue
                        board.push(m0)
                    else:
                        board.push_uci(moves[0])
                except Exception:
                    continue

                for i in range(1, len(moves)):
                    is_solver_move = ((i - 1) % 2 == 0)  # i=1,3,5... are solver moves

                    # If we only want solver moves, still push opponent replies to advance the line.
                    if solver_moves_only and not is_solver_move:
                        try:
                            if strict_legal:
                                mi = chess.Move.from_uci(moves[i])
                                if mi not in board.legal_moves:
                                    break
                                board.push(mi)
                            else:
                                board.push_uci(moves[i])
                        except Exception:
                            break
                        continue

                    move_uci = moves[i]
                    move_id = uci_helpers.UCI_TO_ID.get(move_uci)
                    if move_id is None:
                        if skip_missing_move_ids:
                            break
                        raise KeyError(f"Move {move_uci} not found in UCI_TO_ID")

                    tokens = fen_helpers.encode_state(board, rep=0)
                    tokens_buffer.append(tokens)
                    move_buffer.append(move_id)

                    # advance
                    try:
                        if strict_legal:
                            mi = chess.Move.from_uci(move_uci)
                            if mi not in board.legal_moves:
                                break
                            board.push(mi)
                        else:
                            board.push_uci(move_uci)
                    except Exception:
                        break

                    if len(tokens_buffer) >= batch_rows:
                        rows += _flush(writer, tokens_buffer, move_buffer)

            rows += _flush(writer, tokens_buffer, move_buffer)
        finally:
            _close_csv_stream(f)

    return puzzles, rows

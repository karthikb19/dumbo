from pathlib import Path
import time
import chess.pgn
from typing import Tuple, List
import pyarrow as pa
import pyarrow.parquet as pq

import constants
import fen_helpers
import uci_helpers

SCHEMA = pa.schema([
    ("tokens", pa.binary(constants.L)),
    ("move_id", pa.uint16()),
])

def _repetition_key(board: chess.Board) -> Tuple:
    return (board.board_fen(), board.turn, board.castling_rights, board.ep_square)

def _flush(writer: pq.ParquetWriter, tokens_buffer: List[bytes], move_buffer: List[int]) -> int:
    if not tokens_buffer or not move_buffer:
        return 0 

    table = pa.Table.from_arrays(
        [pa.array(tokens_buffer, type=pa.binary(constants.L)), pa.array(move_buffer, type=pa.uint16())],
        schema=SCHEMA,
    )
    writer.write_table(table)
    N = len(tokens_buffer)
    tokens_buffer.clear()
    move_buffer.clear()
    return N

def shard_pgn(in_path: str, out_dir: str, games_per_shard: int = 10000):
    out = Path(out_dir)
    out.mkdir(exist_ok=True, parents=True)

    shard_idx = 0
    games = 0
    total_games = 0
    fout = None # the current output file

    def open_new_shard():
        nonlocal shard_idx, fout, games
        if fout:
            fout.close()
        shard_path = out / f"shard_{shard_idx}.pgn" # new shard path
        fout = open(shard_path, "w", encoding="utf-8")
        games = 0
        shard_idx += 1
        print(f"Opened new shard: {shard_path}")
    

    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        open_new_shard()
        overall_start_time, current_start_time, shard_start_time = time.time(), time.time(), time.time()
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            fout.write(game.accept(exporter))
            fout.write("\n\n")
            games += 1
            total_games += 1
            if games % games_per_shard == 0:
                print("----- NEW SHARD -----")
                print(f"Wrote {games} games to {shard_idx}")
                print(f"Total games: {total_games}")
                print(f"Time taken: {time.time() - shard_start_time}")
                print("---------------------")
                open_new_shard()
                shard_start_time = time.time()
            if time.time() - current_start_time >= 5:
                print(f"Processed: {total_games} games in {time.time() - overall_start_time} seconds") 
                current_start_time = time.time()
    print("DONE")
            
    
    if fout:
        fout.close()

def process_shard(path: str, out_dir: str = "parquet_out/2025-10", batch_rows: int = 100_000) -> Tuple[int, int]:
    out_path = Path(out_dir) / (Path(path).stem + ".parquet")
    out_path.parent.mkdir(exist_ok=True, parents=True)

    games = 0
    rows = 0

    tokens_buffer = []
    move_buffer = []

    with pq.ParquetWriter(out_path, SCHEMA, compression="zstd") as writer:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while (game := chess.pgn.read_game(f)) is not None:
                games += 1
                board = game.board()
                
                rep_counts = {}
                rep_counts[_repetition_key(board)] = 0
                
                for move in game.mainline_moves():
                    key = _repetition_key(board)
                    seen = rep_counts.get(key, 1)
                    rep = min(seen - 1, 2)

                    tokens = fen_helpers.encode_state(board, rep)
                    move_id = uci_helpers.UCI_TO_ID.get(move.uci())

                    if move_id is None:
                        raise KeyError(f"Move {move.uci()} not found in uci_to_id")
                    
                    tokens_buffer.append(tokens)
                    move_buffer.append(move_id)
                    board.push(move)

                    rep_k = _repetition_key(board)
                    rep_counts[rep_k] = rep_counts.get(rep_k, 0) + 1

                    if len(tokens_buffer) >= batch_rows:
                        rows += _flush(writer, tokens_buffer, move_buffer)
        rows += _flush(writer, tokens_buffer, move_buffer)

    return games, rows

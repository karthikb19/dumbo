# Parser for datasets/ 
import sys
import os
import chess.pgn
from shard import shard_pgn

def parse_pgn(pgn_path: str):
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        games_count = 0
        while True:
            game = chess.pgn.read_game(f) 
            if game is None:
                break  
            
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            games_count += 1
            if games_count % 1000 == 0:
                print(f"Processed {games_count} games")
    print(f"Total games: {games_count}")

def main():
    # shard_pgn("datasets/lichess_elite_2025-11.pgn", "datasets/shards-2025-11", 10000)
    # shard_pgn("datasets/lichess_elite_2025-10.pgn", "datasets/shards-2025-10", 10000)
    shard_pgn("datasets/lichess_elite_2025-09.pgn", "datasets/shards-2025-09", 10000)

if __name__ == "__main__":
    main()
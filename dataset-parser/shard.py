import chess.pgn
from pathlib import Path
import time

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
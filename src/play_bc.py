from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional
from collections import Counter

import torch
import chess

# Your files
import constants
import fen_helpers
from infer_bc import load_checkpoint, load_uci_maps, choose_legal_move


def parse_user_move(board: chess.Board, s: str) -> Optional[chess.Move]:
    s = s.strip()
    if not s:
        return None

    # Try SAN first (e.g. Nf3, O-O, exd5, Qh5+)
    try:
        mv = board.parse_san(s)
        return mv
    except Exception:
        pass

    # Try UCI (e.g. e2e4, e7e8q)
    try:
        mv = chess.Move.from_uci(s)
        if mv in board.legal_moves:
            return mv
        return None
    except Exception:
        return None

def print_board(board: chess.Board) -> None:
    print()
    print(board)  # <-- default: letters like R N B Q K P
    turn = "White" if board.turn == chess.WHITE else "Black"
    
    # Get repetition count for current position
    rep_stats = get_repetition_stats(board)
    current_pos = get_position_key(board)
    rep_count = rep_stats[current_pos]
    
    print(f"\nTurn: {turn} | Fullmove: {board.fullmove_number} | Halfmove: {board.halfmove_clock} | Repetitions: {rep_count}")
    if board.is_check():
        print("CHECK!")
    if rep_count >= 3:
        print("⚠️  THREEFOLD REPETITION - Draw can be claimed!")
    print("FEN:", board.fen())
    print()

def print_help() -> None:
    print(
        "Commands:\n"
        "  help            Show this help\n"
        "  quit / exit      Quit the game\n"
        "  resign           Resign\n"
        "  fen              Print current FEN\n"
        "  moves            Print legal moves (SAN)\n"
        "  reps             Show position repetition statistics\n"
        "  undo             Undo last full turn (you + engine), if possible\n"
        "\nMove input formats:\n"
        "  SAN: Nf3, exd5, O-O, Qh5+, a8=Q, etc.\n"
        "  UCI: e2e4, g1f3, e7e8q, etc.\n"
    )


def legal_moves_san(board: chess.Board, limit: int = 200) -> List[str]:
    out = []
    for i, mv in enumerate(board.legal_moves):
        if i >= limit:
            out.append("...")
            break
        out.append(board.san(mv))
    return out


def get_position_key(board: chess.Board) -> str:
    """
    Returns a unique key for the current position.
    Uses board_fen() which includes piece placement, turn, castling, and en passant.
    This is what chess uses for threefold repetition.
    """
    return board.board_fen() + " " + ("w" if board.turn else "b") + " " + board.castling_xfen() + " " + (board.ep_square and chess.square_name(board.ep_square) or "-")


def get_repetition_stats(board: chess.Board) -> Dict[str, int]:
    """
    Returns a Counter of how many times each position has occurred in the game.
    """
    position_counter = Counter()
    temp_board = chess.Board(chess.STARTING_FEN)
    
    # Replay all moves to track positions
    for move in board.move_stack:
        position_counter[get_position_key(temp_board)] += 1
        temp_board.push(move)
    
    # Count current position
    position_counter[get_position_key(temp_board)] += 1
    
    return position_counter


def get_current_rep_count(board: chess.Board) -> int:
    """
    Returns the repetition count for the current position (0-indexed for transformer).
    0 = first time seeing this position
    1 = seen once before (2nd occurrence)
    2+ = seen 2+ times before (clamped to 2 in the model)
    """
    rep_stats = get_repetition_stats(board)
    current_pos = get_position_key(board)
    # Subtract 1 because rep_stats counts current occurrence, but we want "times seen before"
    return max(0, rep_stats[current_pos] - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="checkpoints/dumbo_bc.pt", help="Path to dumbo_bc.pt checkpoint")
    ap.add_argument("--uci", type=str, default="datasets/uci_1968.txt", help="Path to 1968-UCI mapping file")
    ap.add_argument("--side", type=str, choices=["white", "black"], default="white", help="Your side")
    ap.add_argument("--device", type=str, default=None, help='cuda / cpu (default: auto)')
    ap.add_argument("--fen", type=str, default=chess.STARTING_FEN, help="Starting FEN")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load maps + model once
    uci_to_id, id_to_uci = load_uci_maps(args.uci)
    model = load_checkpoint(args.ckpt).to(device)
    model.eval()

    board = chess.Board(args.fen)

    user_is_white = (args.side == "white")
    engine_color = chess.BLACK if user_is_white else chess.WHITE

    print_help()
    print_board(board)

    # If engine goes first (user chose black), play one engine move immediately
    if board.turn == engine_color and not board.is_game_over():
        rep = get_current_rep_count(board)
        mv = choose_legal_move(board, model, uci_to_id, id_to_uci, device=device, rep=rep)
        print(f"Engine plays: {board.san(mv)}  ({mv.uci()})")
        board.push(mv)
        print_board(board)

    while True:
        if board.is_game_over():
            print("Game over:", board.result(), "-", board.outcome())
            
            # Show repetition statistics
            rep_stats = get_repetition_stats(board)
            repeated_positions = {k: v for k, v in rep_stats.items() if v > 1}
            print("\n=== Game Statistics ===")
            print(f"Total unique positions: {len(rep_stats)}")
            print(f"Repeated positions: {len(repeated_positions)}")
            if repeated_positions:
                max_reps = max(repeated_positions.values())
                print(f"Maximum repetitions: {max_reps}x")
            print()
            
            break

        # User turn?
        if board.turn == (chess.WHITE if user_is_white else chess.BLACK):
            s = input("You> ").strip()

            if s.lower() in ["quit", "exit", "q"]:
                print("Bye.")
                break
            if s.lower() == "help":
                print_help()
                continue
            if s.lower() == "resign":
                print("You resigned.")
                break
            if s.lower() == "fen":
                print(board.fen())
                continue
            if s.lower() == "moves":
                ms = legal_moves_san(board)
                print("Legal moves (SAN):", " ".join(ms))
                continue
            if s.lower() == "reps":
                rep_stats = get_repetition_stats(board)
                repeated_positions = {k: v for k, v in rep_stats.items() if v > 1}
                if repeated_positions:
                    print(f"\nPosition repetitions (showing {len(repeated_positions)} repeated positions):")
                    for pos, count in sorted(repeated_positions.items(), key=lambda x: x[1], reverse=True):
                        print(f"  {count}x: {pos[:50]}...")  # Show first 50 chars of position key
                else:
                    print("No positions have been repeated yet.")
                continue
            if s.lower() == "undo":
                # Undo last full turn (engine + you) if possible
                if len(board.move_stack) >= 2:
                    board.pop()
                    board.pop()
                    print("Undid last full turn.")
                elif len(board.move_stack) == 1:
                    board.pop()
                    print("Undid last move.")
                else:
                    print("Nothing to undo.")
                print_board(board)
                continue

            mv = parse_user_move(board, s)
            if mv is None:
                print("Invalid / illegal move. Type 'moves' to see legal SAN moves.")
                continue

            print(f"You played: {board.san(mv)}  ({mv.uci()})")
            board.push(mv)
            print_board(board)

            if board.is_game_over():
                print("Game over:", board.result(), "-", board.outcome())
                break

        # Engine turn
        if board.turn == engine_color and not board.is_game_over():
            # (Optional) show your encoded tokens quickly for debugging:
            # tokens_bytes = fen_helpers.encode_state(board, rep=0)
            # print("tokens:", list(tokens_bytes))

            rep = get_current_rep_count(board)
            mv = choose_legal_move(board, model, uci_to_id, id_to_uci, device=device, rep=rep)
            san = board.san(mv)
            print(f"Engine> {san}  ({mv.uci()})")
            board.push(mv)
            print_board(board)


if __name__ == "__main__":
    main()

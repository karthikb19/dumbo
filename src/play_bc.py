from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional

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
    print(f"\nTurn: {turn} | Fullmove: {board.fullmove_number} | Halfmove: {board.halfmove_clock}")
    if board.is_check():
        print("CHECK!")
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
        mv = choose_legal_move(board, model, uci_to_id, id_to_uci, device=device)
        print(f"Engine plays: {board.san(mv)}  ({mv.uci()})")
        board.push(mv)
        print_board(board)

    while True:
        if board.is_game_over():
            print("Game over:", board.result(), "-", board.outcome())
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

            mv = choose_legal_move(board, model, uci_to_id, id_to_uci, device=device)
            san = board.san(mv)
            print(f"Engine> {san}  ({mv.uci()})")
            board.push(mv)
            print_board(board)


if __name__ == "__main__":
    main()

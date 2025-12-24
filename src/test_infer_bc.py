import torch
import chess

import constants
import fen_helpers

# from your infer_bc.py
from infer_bc import load_checkpoint, load_uci_maps, choose_legal_move


def topk_legal_moves(board, model, uci_to_id, id_to_uci, device="cuda", k=10):
    tokens_bytes = fen_helpers.encode_state(board, rep=0)
    toks = torch.tensor(list(tokens_bytes), dtype=torch.long, device=device).unsqueeze(0)  # [1, 78]

    logits = model(toks)[0]  # [1968]

    legal_ids = []
    for mv in board.legal_moves:
        mid = uci_to_id.get(mv.uci())
        if mid is not None:
            legal_ids.append(mid)

    if not legal_ids:
        raise RuntimeError("No legal moves mapped into your 1968 UCI set.")

    mask = torch.full_like(logits, float("-inf"))
    mask[torch.tensor(legal_ids, device=device)] = 0.0
    masked_logits = logits + mask

    probs = torch.softmax(masked_logits, dim=-1)
    vals, idxs = torch.topk(probs, k=min(k, len(legal_ids)))

    out = []
    for p, mid in zip(vals.tolist(), idxs.tolist()):
        out.append((id_to_uci[mid], p))
    return out, tokens_bytes


def main():
    CKPT = "checkpoints/dumbo_bc.pt"              # <-- change if yours is elsewhere (e.g. checkpoints/dumbo_bc.pt)
    UCI_FILE = "datasets/uci_1968.txt"  # <-- change if needed

    FEN = "3q1rk1/2p2pb1/R1B1p1pp/5b2/2QPN3/2P1P3/1r3PPP/4R1K1 b - - 0 24"  # try others above
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    uci_to_id, id_to_uci = load_uci_maps(UCI_FILE)

    model = load_checkpoint(CKPT).to(device)
    model.eval()

    board = chess.Board(FEN)
    print("FEN:", board.fen())

    # 1) show your fixed-length layout + tokens
    layout = fen_helpers.generate_fixed_board_layout(board)
    tokens_bytes = fen_helpers.encode_state(board, rep=0)

    print("\nfixed layout len:", len(layout))
    print("fixed layout:", layout)
    print("\ntokens len:", len(tokens_bytes))
    print("tokens (ints):", list(tokens_bytes))

    # basic sanity checks
    assert len(tokens_bytes) == constants.L == 78
    assert all(0 <= t < constants.VOCAB_SIZE for t in tokens_bytes), "token out of vocab range!"

    # 2) show model choice + top-k distribution over legal moves
    mv = choose_legal_move(board, model, uci_to_id, id_to_uci, device=device)
    print("\nCHOSEN MOVE:", mv.uci())

    topk, _ = topk_legal_moves(board, model, uci_to_id, id_to_uci, device=device,k=30)
    print("\nTop-10 legal moves by model prob:")
    for uci, p in topk:
        print(f"  {uci:6s}  p={p:.4f}")

if __name__ == "__main__":
    main()

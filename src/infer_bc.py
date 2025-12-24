# infer_bc.py
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import chess

import constants
import fen_helpers
from dumbo_bc import DumboBC, ModelConfig


def load_uci_maps(path: str) -> Tuple[Dict[str, int], List[str]]:
    uci_to_id: Dict[str, int] = {}
    id_to_uci: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            u = line.strip()
            if not u:
                continue
            uci_to_id[u] = i
            id_to_uci.append(u)
    assert len(id_to_uci) == 1968
    return uci_to_id, id_to_uci


@torch.inference_mode()
def choose_legal_move(board: chess.Board, model: DumboBC, uci_to_id: Dict[str, int], id_to_uci: List[str], device: str = "cuda", rep: int = 0) -> chess.Move:
    tokens_bytes = fen_helpers.encode_state(board, rep=rep)
    tokens = torch.tensor(list(tokens_bytes), dtype=torch.long, device=device).unsqueeze(0)  # [1,78]

    logits = model(tokens)[0]  # [1968]

    legal_ids = []
    for mv in board.legal_moves:
        u = mv.uci()
        mid = uci_to_id.get(u)
        if mid is not None:
            legal_ids.append(mid)

    if not legal_ids:
        raise RuntimeError("No legal moves mapped into your 1968 UCI set (unexpected).")

    mask = torch.full_like(logits, float("-inf"))
    mask[torch.tensor(legal_ids, device=logits.device)] = 0.0
    masked_logits = logits + mask

    best_id = int(torch.argmax(masked_logits).item())
    best_uci = id_to_uci[best_id]
    return chess.Move.from_uci(best_uci)


def load_checkpoint(ckpt_path: str) -> DumboBC:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ModelConfig(**ckpt["cfg"])
    model = DumboBC(cfg, pad_id=constants.PAD_ID)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model

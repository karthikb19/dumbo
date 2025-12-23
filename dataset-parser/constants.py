from typing import Dict

L = 78

PAD_ID = 0
CLS_ID = 1
REP0_ID = 2
REP1_ID = 3
REP2_ID = 4

CHARS = (
    ["."]
    + list("PNBRQKpnbrqk")
    + list("wb")
    + list("KQkq")          # castling letters
    + ["-"]                 # ep none marker
    + list("abcdefgh")      # ep file
    + list("12345678")      # ep rank
    + list("0123456789")    # clocks
)
# Deduplicate while preserving order:
seen = set()
CHARS = [c for c in CHARS if not (c in seen or seen.add(c))]

CHAR_BASE = 5
CHAR_TO_ID: Dict[str, int] = {c: CHAR_BASE + i for i, c in enumerate(CHARS)}

def rep_token_id(rep: int) -> int:
    if rep <= 0: 
        return REP0_ID
    if rep == 1: 
        return REP1_ID
    return REP2_ID  # clamp 2+

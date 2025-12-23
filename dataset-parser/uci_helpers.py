from typing import Dict

def load_uci_to_id(path: str) -> Dict[str, int]:
    uci_to_id: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            uci = line.strip()
            if not uci:
                continue
            uci_to_id[uci] = i
    return uci_to_id

UCI_TO_ID = load_uci_to_id("datasets/uci_1968.txt")
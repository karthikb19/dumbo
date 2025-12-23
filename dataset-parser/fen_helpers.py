import constants
import chess

def expand_board_fen_to_64(board_fen: str) -> str:
    out = []
    for c in board_fen:
        if c == "/":
            continue
        if c.isdigit():
            out.extend(["."] * int(c))
        else:
            out.append(c)
    res = "".join(out)
    assert len(res) == 64
    return res

def generate_fixed_board_layout(board: chess.Board) -> str:
    board_fen = board.board_fen()
    expanded_board_fen = expand_board_fen_to_64(board_fen)
    side_to_move = "w" if board.turn == chess.WHITE else "b"
    castling = "".join(c if c in board.castling_xfen() else "." for c in "KQkq")

    if board.ep_square is None:
        ep_square = "-."
    else:
        ep_square = chess.square_name(board.ep_square)
    
    halfmove = str(board.halfmove_clock)
    halfmove = (halfmove + "..")[:2]
    halfmove = "".join(c if c.isdigit() else "." for c in halfmove)

    fullmove = str(board.fullmove_number)
    fullmove = (fullmove + "...")[:3]
    fullmove = "".join(c if c.isdigit() else "." for c in fullmove)

    res = expanded_board_fen + side_to_move + castling + ep_square + halfmove + fullmove
    return res

def encode_state(board: chess.Board, rep: int) -> bytes:
    layout = generate_fixed_board_layout(board)
    toks = bytearray(constants.L)
    toks[0] = constants.CLS_ID
    for i, c in enumerate(layout):
        token_id = constants.CHAR_TO_ID.get(c)
        if token_id is None:
            raise ValueError(f"Invalid character: {c}")
        toks[i + 1] = token_id
    toks[-1] = constants.rep_token_id(rep)
    assert len(toks) == constants.L
    return bytes(toks)
 
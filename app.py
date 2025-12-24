"""
Flask web interface for playing chess against the Dumbo BC engine.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional
from collections import Counter
from flask import Flask, render_template, request, jsonify, session
import torch
import chess
import secrets

# Your files
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from infer_bc import load_checkpoint, load_uci_maps, choose_legal_move

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Global model and maps (loaded once at startup)
MODEL = None
UCI_TO_ID = None
ID_TO_UCI = None
DEVICE = None


def init_model():
    """Initialize the chess engine model."""
    global MODEL, UCI_TO_ID, ID_TO_UCI, DEVICE
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on device: {DEVICE}")
    
    # Load UCI maps and model
    UCI_TO_ID, ID_TO_UCI = load_uci_maps("datasets/uci_1968.txt")
    MODEL = load_checkpoint("checkpoints/dumbo_bc.pt").to(DEVICE)
    MODEL.eval()
    
    print("Model loaded successfully!")


def get_position_key(board: chess.Board) -> str:
    """Returns a unique key for the current position."""
    return (board.board_fen() + " " + 
            ("w" if board.turn else "b") + " " + 
            board.castling_xfen() + " " + 
            (chess.square_name(board.ep_square) if board.ep_square else "-"))


def get_current_rep_count(board: chess.Board, move_stack: List[str]) -> int:
    """
    Calculate the repetition count for the current position.
    Returns how many times this position was seen BEFORE (0-indexed for transformer).
    """
    position_counter = Counter()
    temp_board = chess.Board()
    
    # Replay all moves to track positions
    for move_uci in move_stack:
        position_counter[get_position_key(temp_board)] += 1
        try:
            temp_board.push(chess.Move.from_uci(move_uci))
        except ValueError:
            pass
    
    # Count current position occurrence
    current_key = get_position_key(temp_board)
    position_counter[current_key] += 1
    
    # Return times seen before (subtract 1 for current occurrence)
    return max(0, position_counter[current_key] - 1)


def get_game_state() -> Dict:
    """Get or initialize the current game state from session."""
    if 'fen' not in session:
        session['fen'] = chess.STARTING_FEN
        session['move_history'] = []
        session['move_stack_uci'] = []  # Track UCI moves for repetition
    
    return {
        'fen': session['fen'],
        'move_history': session.get('move_history', []),
        'move_stack_uci': session.get('move_stack_uci', [])
    }


def update_game_state(board: chess.Board, move: Optional[chess.Move] = None):
    """Update the session with the current board state."""
    session['fen'] = board.fen()
    if move:
        history = session.get('move_history', [])
        history.append({
            'san': board.san(move) if move in board.legal_moves else move.uci(),
            'uci': move.uci()
        })
        session['move_history'] = history
        
        # Also track UCI moves for repetition calculation
        move_stack = session.get('move_stack_uci', [])
        move_stack.append(move.uci())
        session['move_stack_uci'] = move_stack


@app.route('/')
def index():
    """Serve the main chess interface."""
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    data = request.json
    player_color = data.get('color', 'white')
    
    # Reset game state
    session['fen'] = chess.STARTING_FEN
    session['move_history'] = []
    session['move_stack_uci'] = []
    session['player_color'] = player_color
    
    board = chess.Board()
    
    # If player chose black, engine makes first move
    if player_color == 'black':
        # Rep is 0 at start of game
        engine_move = choose_legal_move(board, MODEL, UCI_TO_ID, ID_TO_UCI, device=DEVICE, rep=0)
        engine_move_san = board.san(engine_move)
        board.push(engine_move)
        update_game_state(board, engine_move)
        
        return jsonify({
            'success': True,
            'fen': board.fen(),
            'engine_move': engine_move.uci(),
            'engine_move_san': engine_move_san,
            'game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None
        })
    
    return jsonify({
        'success': True,
        'fen': board.fen(),
        'game_over': False
    })


@app.route('/api/make_move', methods=['POST'])
def make_move():
    """Handle a player move and get engine response."""
    data = request.json
    move_uci = data.get('move')
    
    if not move_uci:
        return jsonify({'success': False, 'error': 'No move provided'}), 400
    
    # Get current board state
    game_state = get_game_state()
    board = chess.Board(game_state['fen'])
    
    # Validate and make player move
    try:
        player_move = chess.Move.from_uci(move_uci)
        if player_move not in board.legal_moves:
            return jsonify({'success': False, 'error': 'Illegal move'}), 400
        
        board.push(player_move)
        update_game_state(board, player_move)
        
        # Check if game is over after player move
        if board.is_game_over():
            return jsonify({
                'success': True,
                'fen': board.fen(),
                'game_over': True,
                'result': board.result(),
                'outcome': str(board.outcome())
            })
        
        # Calculate repetition count for engine's turn
        move_stack = session.get('move_stack_uci', [])
        rep = get_current_rep_count(board, move_stack)
        
        # Engine makes a move with rep info
        engine_move = choose_legal_move(board, MODEL, UCI_TO_ID, ID_TO_UCI, device=DEVICE, rep=rep)
        engine_move_san = board.san(engine_move)
        board.push(engine_move)
        update_game_state(board, engine_move)
        
        return jsonify({
            'success': True,
            'fen': board.fen(),
            'engine_move': engine_move.uci(),
            'engine_move_san': engine_move_san,
            'game_over': board.is_game_over(),
            'result': board.result() if board.is_game_over() else None,
            'outcome': str(board.outcome()) if board.is_game_over() else None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/get_state', methods=['GET'])
def get_state():
    """Get the current game state."""
    game_state = get_game_state()
    board = chess.Board(game_state['fen'])
    
    return jsonify({
        'fen': game_state['fen'],
        'move_history': game_state['move_history'],
        'game_over': board.is_game_over(),
        'result': board.result() if board.is_game_over() else None,
        'turn': 'white' if board.turn == chess.WHITE else 'black'
    })


@app.route('/api/legal_moves', methods=['GET'])
def legal_moves():
    """Get all legal moves in UCI format."""
    game_state = get_game_state()
    board = chess.Board(game_state['fen'])
    
    moves = [move.uci() for move in board.legal_moves]
    
    return jsonify({
        'legal_moves': moves
    })


if __name__ == '__main__':
    init_model()
    app.run(debug=True, port=5000)

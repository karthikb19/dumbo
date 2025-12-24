// Game state
let board = null;
let game = new Chess();
let playerColor = 'white';
let gameActive = false;

// Initialize the chess board
function initBoard() {
    const config = {
        draggable: true,
        position: 'start',
        pieceTheme: 'static/img/chesspieces/wikipedia/{piece}.png',
        onDragStart: onDragStart,
        onDrop: onDrop,
        onSnapEnd: onSnapEnd
    };

    board = Chessboard('board', config);
    $(window).resize(board.resize);
}

// Prevent dragging if game is not active or not player's turn
function onDragStart(source, piece, position, orientation) {
    if (!gameActive) return false;
    if (game.game_over()) return false;

    // Only allow player to move their own pieces
    if ((playerColor === 'white' && piece.search(/^b/) !== -1) ||
        (playerColor === 'black' && piece.search(/^w/) !== -1)) {
        return false;
    }

    // Check if it's player's turn
    if ((game.turn() === 'w' && playerColor !== 'white') ||
        (game.turn() === 'b' && playerColor !== 'black')) {
        return false;
    }
}

// Handle piece drop
function onDrop(source, target) {
    // Check if the move is legal
    const move = game.move({
        from: source,
        to: target,
        promotion: 'q' // Always promote to queen for simplicity
    });

    // Illegal move
    if (move === null) return 'snapback';

    // Make the move on the server
    makeMove(move.from + move.to + (move.promotion || ''));
}

// Update board position after move
function onSnapEnd() {
    board.position(game.fen());
}

// Make a move via API
async function makeMove(moveUci) {
    showEngineThinking(true);
    updateStatus('Waiting for engine...');

    try {
        const response = await fetch('/api/make_move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ move: moveUci })
        });

        const data = await response.json();

        if (data.success) {
            // Update game state
            game.load(data.fen);
            board.position(data.fen);

            // Update move history
            updateMoveHistory();

            // Check game status
            if (data.game_over) {
                gameActive = false;
                updateStatus(`Game Over: ${data.result}`);
                showGameOver(data.result);
            } else {
                updateStatus('Your turn');
                updateTurnIndicator();
            }
        } else {
            alert('Error: ' + data.error);
            game.undo();
            board.position(game.fen());
        }
    } catch (error) {
        console.error('Error making move:', error);
        alert('Failed to make move. Please try again.');
        game.undo();
        board.position(game.fen());
    } finally {
        showEngineThinking(false);
    }
}

// Start a new game
async function startNewGame(color) {
    playerColor = color;
    game.reset();

    showEngineThinking(true);
    updateStatus('Starting new game...');

    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ color: color })
        });

        const data = await response.json();

        if (data.success) {
            game.load(data.fen);

            // Set board orientation
            board.orientation(color);
            board.position(data.fen);

            gameActive = true;

            // Clear move history
            $('#moveHistory').html('<p class="empty-state">No moves yet</p>');

            // Update UI
            updateTurnIndicator();
            if (color === 'white') {
                updateStatus('Your turn');
            } else {
                updateStatus('Engine moved: ' + (data.engine_move_san || data.engine_move));
                updateMoveHistory();
            }

            // Hide color selection
            $('#colorSelection').hide();
        } else {
            alert('Error starting game: ' + data.error);
        }
    } catch (error) {
        console.error('Error starting game:', error);
        alert('Failed to start game. Please try again.');
    } finally {
        showEngineThinking(false);
    }
}

// Update turn indicator
function updateTurnIndicator() {
    const turn = game.turn() === 'w' ? 'White' : 'Black';
    $('#turnIndicator').text(turn);

    if ((turn === 'White' && playerColor === 'white') ||
        (turn === 'Black' && playerColor === 'black')) {
        $('#turnIndicator').css('color', '#10b981');
    } else {
        $('#turnIndicator').css('color', '#6366f1');
    }
}

// Update status text
function updateStatus(text) {
    $('#statusText').text(text);
}

// Show/hide engine thinking indicator
function showEngineThinking(show) {
    if (show) {
        $('#engineThinking').fadeIn(200);
    } else {
        $('#engineThinking').fadeOut(200);
    }
}

// Update move history display
function updateMoveHistory() {
    const history = game.history({ verbose: true });

    if (history.length === 0) {
        $('#moveHistory').html('<p class="empty-state">No moves yet</p>');
        return;
    }

    let html = '';
    for (let i = 0; i < history.length; i += 2) {
        const moveNum = Math.floor(i / 2) + 1;
        const whiteMove = history[i].san;
        const blackMove = history[i + 1] ? history[i + 1].san : '';

        html += `
            <div class="move-item">
                <span class="move-number">${moveNum}.</span>
                <span class="move-notation">${whiteMove}</span>
                ${blackMove ? `<span class="move-notation">${blackMove}</span>` : ''}
            </div>
        `;
    }

    $('#moveHistory').html(html);

    // Scroll to bottom
    const moveHistory = document.getElementById('moveHistory');
    moveHistory.scrollTop = moveHistory.scrollHeight;
}

// Show game over message
function showGameOver(result) {
    let message = 'Game Over!\\n\\n';

    if (result === '1-0') {
        message += 'White wins!';
    } else if (result === '0-1') {
        message += 'Black wins!';
    } else if (result === '1/2-1/2') {
        message += "It's a draw!";
    }

    setTimeout(() => {
        alert(message);
    }, 500);
}

// Event handlers
$(document).ready(function () {
    initBoard();

    // New game button
    $('#newGameBtn').click(function () {
        $('#colorSelection').slideToggle(200);
    });

    // Color selection buttons
    $('.color-selection button').click(function () {
        const color = $(this).data('color');
        startNewGame(color);
    });

    // Initialize status
    updateStatus('Click "New Game" to start');
});

"""
TensorFlow Chess: Brain vs Brain
================================

Two players, each with their OWN brain model:
  - Player 1 Brain (White)
  - Player 2 Brain (Black)

Each brain has the SAME architecture:
  - Frozen foundation layers (read-only, shared initialization, then locked)
  - Trainable "learning head" layers on top (each player's own)

They play chess against each other in the terminal (Unicode board).
After every game, each brain trains ONLY its learning head on the
move outcomes it experienced (win = reinforce chosen moves,
loss = discourage them). The frozen foundation never changes.

Run:
    python tf_chess_brain_vs_brain.py
    python tf_chess_brain_vs_brain.py --games 20 --delay 0.15
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import time
import sys
import numpy as np
import chess
import tensorflow as tf
from tensorflow.keras import layers, Model

tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Board encoding: 8x8x12 planes (6 piece types x 2 colors) + side-to-move plane
# ---------------------------------------------------------------------------
PIECE_TO_PLANE = {
    (chess.PAWN,   chess.WHITE): 0,  (chess.PAWN,   chess.BLACK): 6,
    (chess.KNIGHT, chess.WHITE): 1,  (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.WHITE): 2,  (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK,   chess.WHITE): 3,  (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.WHITE): 4,  (chess.QUEEN,  chess.BLACK): 10,
    (chess.KING,   chess.WHITE): 5,  (chess.KING,   chess.BLACK): 11,
}

def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as an (8, 8, 13) tensor."""
    planes = np.zeros((8, 8, 13), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        row = 7 - (sq // 8)
        col = sq % 8
        planes[row, col, PIECE_TO_PLANE[(piece.piece_type, piece.color)]] = 1.0
    planes[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes


# ---------------------------------------------------------------------------
# Brain model: frozen foundation + trainable learning head
# ---------------------------------------------------------------------------
def build_foundation(seed: int = 42) -> Model:
    """Shared frozen foundation. Built once, weights frozen forever."""
    tf.random.set_seed(seed)
    inp = layers.Input(shape=(8, 8, 13), name="board")
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="found_conv1")(inp)
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="found_conv2")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="found_conv3")(x)
    x = layers.Flatten(name="found_flat")(x)
    x = layers.Dense(128, activation="relu", name="found_dense")(x)
    foundation = Model(inp, x, name="foundation")
    for layer in foundation.layers:
        layer.trainable = False
    return foundation


def build_brain(name: str, foundation: Model, head_seed: int) -> Model:
    """A player's brain: frozen foundation + this player's own learning head.

    Output: a scalar in [-1, 1] scoring the position for the side to move
    (higher = better for the mover).
    """
    tf.random.set_seed(head_seed)
    inp = layers.Input(shape=(8, 8, 13), name=f"{name}_in")
    features = foundation(inp, training=False)      # frozen path
    h = layers.Dense(64, activation="relu", name=f"{name}_head1")(features)
    h = layers.Dense(32, activation="relu", name=f"{name}_head2")(h)
    out = layers.Dense(1, activation="tanh", name=f"{name}_score")(h)
    brain = Model(inp, out, name=name)
    brain.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return brain


# ---------------------------------------------------------------------------
# Move selection: brain scores every resulting position; pick the best
# (with a little exploration temperature so games differ)
# ---------------------------------------------------------------------------
def choose_move(brain: Model, board: chess.Board, temperature: float = 0.4):
    legal = list(board.legal_moves)
    if not legal:
        return None, None, None

    # Build a batch of resulting positions, scored from the OPPONENT'S view
    # after we move. Lower opponent-score = better for us.
    batch = np.zeros((len(legal), 8, 8, 13), dtype=np.float32)
    for i, mv in enumerate(legal):
        board.push(mv)
        batch[i] = encode_board(board)
        board.pop()

    opp_scores = brain.predict(batch, verbose=0).flatten()   # opp's POV
    our_scores = -opp_scores                                  # our POV

    if temperature > 1e-6:
        logits = our_scores / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = int(np.random.choice(len(legal), p=probs))
    else:
        idx = int(np.argmax(our_scores))

    chosen = legal[idx]
    # Encode the position from our POV BEFORE the move — that's what we
    # trained the brain to score. We store it so we can reinforce it later.
    pre_move_encoding = encode_board(board)
    return chosen, pre_move_encoding, float(our_scores[idx])


# ---------------------------------------------------------------------------
# Terminal rendering
# ---------------------------------------------------------------------------
UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}
RESET = "\033[0m"
LIGHT_BG = "\033[47m"
DARK_BG  = "\033[100m"
FG_BLACK = "\033[30m"
FG_WHITE = "\033[97m"
BOLD = "\033[1m"

def render_board(board: chess.Board, header: str = ""):
    lines = []
    if header:
        lines.append(BOLD + header + RESET)
    lines.append("   a  b  c  d  e  f  g  h ")
    for rank in range(7, -1, -1):
        row = [f" {rank+1} "]
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            bg = LIGHT_BG if (rank + file) % 2 == 0 else DARK_BG
            if piece:
                sym = UNICODE[piece.symbol()]
                fg = FG_WHITE if piece.color == chess.WHITE else FG_BLACK
                row.append(f"{bg}{fg} {sym} {RESET}")
            else:
                row.append(f"{bg}   {RESET}")
        row.append(f" {rank+1}")
        lines.append("".join(row))
    lines.append("   a  b  c  d  e  f  g  h ")
    print("\n".join(lines))


def clear_screen():
    sys.stdout.write("\033[H\033[J")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Reward shaping: material + game outcome
# ---------------------------------------------------------------------------
PIECE_VALUE = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

def material_balance(board: chess.Board, color: bool) -> float:
    v = 0
    for piece_type, val in PIECE_VALUE.items():
        v += val * len(board.pieces(piece_type, color))
        v -= val * len(board.pieces(piece_type, not color))
    return v / 39.0   # normalize to roughly [-1, 1]


# ---------------------------------------------------------------------------
# Play one game
# ---------------------------------------------------------------------------
def play_game(brain_white: Model, brain_black: Model, game_num: int,
              delay: float, max_plies: int = 200, temperature: float = 0.4):
    board = chess.Board()

    # Per-player memory: list of (encoded_position, immediate_material_after)
    memory = {chess.WHITE: [], chess.BLACK: []}

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        mover = board.turn
        brain = brain_white if mover == chess.WHITE else brain_black
        move, pre_enc, score = choose_move(brain, board, temperature=temperature)
        if move is None:
            break

        board.push(move)
        material_after = material_balance(board, mover)
        memory[mover].append((pre_enc, material_after))

        clear_screen()
        who = "Player 1 Brain (White)" if mover == chess.WHITE else "Player 2 Brain (Black)"
        header = (f"Game {game_num}  |  Ply {ply+1}  |  {who} plays {move.uci()}"
                  f"  |  self-score {score:+.3f}")
        render_board(board, header=header)
        print(f"\nFEN: {board.fen()}")
        ply += 1
        if delay > 0:
            time.sleep(delay)

    # Outcome from each player's POV: +1 win, -1 loss, 0 draw
    result = board.result(claim_draw=True)
    if result == "1-0":
        outcome = {chess.WHITE: 1.0, chess.BLACK: -1.0}
        winner = "Player 1 (White)"
    elif result == "0-1":
        outcome = {chess.WHITE: -1.0, chess.BLACK: 1.0}
        winner = "Player 2 (Black)"
    else:
        outcome = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        winner = "Draw"

    print(f"\nResult: {result}  ({winner})  after {ply} plies")
    return memory, outcome, result


# ---------------------------------------------------------------------------
# Train each player's learning head on the game it just played
# ---------------------------------------------------------------------------
def train_head(brain: Model, memory_list, outcome: float, name: str):
    if not memory_list:
        return None
    X = np.stack([m[0] for m in memory_list])
    materials = np.array([m[1] for m in memory_list], dtype=np.float32)
    # Blend immediate material signal with final outcome (outcome dominates).
    targets = np.clip(0.25 * materials + 0.75 * outcome, -1.0, 1.0)
    hist = brain.fit(X, targets, epochs=1, batch_size=32, verbose=0)
    loss = float(hist.history["loss"][0])
    print(f"  trained {name} head on {len(X)} positions  |  loss = {loss:.4f}")
    return loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5, help="number of games")
    parser.add_argument("--delay", type=float, default=0.20,
                        help="seconds between moves (0 = no delay)")
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="exploration temperature; 0 = greedy")
    parser.add_argument("--load", type=str, default=None,
                        help="Directory with pretrained brain_player1.keras / "
                             "brain_player2.keras / foundation.keras "
                             "(from pretrain_parallel.py)")
    args = parser.parse_args()

    if args.load and os.path.isdir(args.load):
        print(f"Loading pretrained brains from {args.load}...")
        foundation = tf.keras.models.load_model(
            os.path.join(args.load, "foundation.keras"), compile=False)
        for layer in foundation.layers:
            layer.trainable = False
        brain_p1 = tf.keras.models.load_model(
            os.path.join(args.load, "brain_player1.keras"), compile=False)
        brain_p2 = tf.keras.models.load_model(
            os.path.join(args.load, "brain_player2.keras"), compile=False)
        brain_p1.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        brain_p2.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        frozen_params = sum(np.prod(w.shape) for w in foundation.weights)
        print(f"  foundation params (FROZEN): {frozen_params:,}")
    else:
        print("Building shared frozen foundation...")
        foundation = build_foundation(seed=42)
        frozen_params = sum(np.prod(w.shape) for w in foundation.weights)
        print(f"  foundation params (FROZEN, read-only): {frozen_params:,}")

        print("Building Player 1 Brain (White) with its own learning head...")
        brain_p1 = build_brain("player1", foundation, head_seed=101)
        print("Building Player 2 Brain (Black) with its own learning head...")
        brain_p2 = build_brain("player2", foundation, head_seed=202)

    def trainable_count(m):
        return sum(np.prod(w.shape) for w in m.trainable_weights)
    print(f"  P1 trainable head params: {trainable_count(brain_p1):,}")
    print(f"  P2 trainable head params: {trainable_count(brain_p2):,}")
    print()

    scoreboard = {"P1_wins": 0, "P2_wins": 0, "draws": 0}

    for g in range(1, args.games + 1):
        memory, outcome, result = play_game(
            brain_p1, brain_p2, game_num=g,
            delay=args.delay, max_plies=args.max_plies,
            temperature=args.temperature,
        )

        if result == "1-0": scoreboard["P1_wins"] += 1
        elif result == "0-1": scoreboard["P2_wins"] += 1
        else: scoreboard["draws"] += 1

        # Sanity check: foundation weights unchanged before training
        found_before = [w.numpy().copy() for w in foundation.weights]

        print("\nTraining learning heads (foundation stays frozen)...")
        train_head(brain_p1, memory[chess.WHITE], outcome[chess.WHITE], "P1")
        train_head(brain_p2, memory[chess.BLACK], outcome[chess.BLACK], "P2")

        # Verify frozen foundation didn't move
        found_after = [w.numpy() for w in foundation.weights]
        drift = max(np.max(np.abs(a - b)) for a, b in zip(found_before, found_after))
        print(f"  foundation weight drift: {drift:.2e}  (should be 0)")

        print(f"\nScoreboard after game {g}:  "
              f"P1={scoreboard['P1_wins']}  "
              f"P2={scoreboard['P2_wins']}  "
              f"Draws={scoreboard['draws']}")
        print("-" * 60)

    print("\nFinal scoreboard:")
    print(f"  Player 1 Brain wins: {scoreboard['P1_wins']}")
    print(f"  Player 2 Brain wins: {scoreboard['P2_wins']}")
    print(f"  Draws:               {scoreboard['draws']}")


if __name__ == "__main__":
    main()

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
import sys
import subprocess
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def _pip_install(*pkgs):
    print(f"[bootstrap] pip install {' '.join(pkgs)}")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           "--disable-pip-version-check", *pkgs]
    try:
        subprocess.check_call(cmd + ["--break-system-packages"])
    except subprocess.CalledProcessError:
        subprocess.check_call(cmd)


def _detect_gpu() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"],
                                      stderr=subprocess.DEVNULL, timeout=5)
        return b"GPU" in out
    except Exception:
        return False


def _bootstrap():
    try:
        import numpy  # noqa: F401
    except ImportError:
        _pip_install("numpy")
    try:
        import chess  # noqa: F401
    except ImportError:
        _pip_install("chess")
    try:
        import tensorflow  # noqa: F401
    except ImportError:
        if _detect_gpu():
            print("[bootstrap] NVIDIA GPU detected -> installing 'tensorflow'")
            _pip_install("tensorflow")
        else:
            print("[bootstrap] No GPU detected -> installing 'tensorflow-cpu'")
            _pip_install("tensorflow-cpu")


_bootstrap()

import argparse
import time
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
# Move selection: alpha-beta minimax with the brain as the leaf evaluator.
#
# The brain scores a position from the side-to-move's POV in [-1, 1]. Search
# lets it see tactics and avoid the shuffling / repetition draws you get from
# pure one-ply greedy play. Captures and checks get one extra ply of "quiet"
# search so we don't stop the search mid-exchange.
# ---------------------------------------------------------------------------
_EVAL_CACHE = {}   # (fen_key) -> score-from-side-to-move POV
_MATE_SCORE = 1.0  # tanh-normalized brain scores live in [-1, 1]

# Cache one tf.function per brain — the raw Keras __call__ has ~0.5ms of
# Python dispatch overhead which dominates when we call it thousands of times
# inside a search. Compiling to a concrete function drops that to microseconds.
_BRAIN_FN_CACHE = {}
def _brain_fn(brain: Model):
    fn = _BRAIN_FN_CACHE.get(id(brain))
    if fn is None:
        @tf.function(reduce_retracing=True)
        def _fn(x):
            return brain(x, training=False)
        _BRAIN_FN_CACHE[id(brain)] = _fn
        fn = _fn
    return fn

# Simple material fallback so the leaf eval isn't purely the brain — helps a
# lot with tactics even when the brain's take is mushy.
_PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
              chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

def _material_from_pov(board: chess.Board) -> float:
    us = board.turn
    v = 0
    for pt, val in _PIECE_VAL.items():
        v += val * len(board.pieces(pt, us))
        v -= val * len(board.pieces(pt, not us))
    return v / 39.0

def _terminal_score(board: chess.Board):
    """None if not terminal, else score-from-side-to-move POV."""
    if board.is_checkmate():
        return -_MATE_SCORE            # side-to-move is mated -> very bad
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_seventyfive_moves() or board.is_fivefold_repetition()
            or board.can_claim_threefold_repetition()
            or board.can_claim_fifty_moves()):
        return 0.0
    return None

def _leaf_eval(brain: Model, board: chess.Board) -> float:
    """Blend brain score with material. Both are POV of side-to-move."""
    key = board._transposition_key()
    cached = _EVAL_CACHE.get(key)
    if cached is not None:
        return cached
    enc = encode_board(board)[None, ...]
    brain_val = float(_brain_fn(brain)(tf.constant(enc)).numpy().flatten()[0])
    mat = _material_from_pov(board)
    # 70% brain, 30% material — brain still leads, material stops it from
    # hallucinating that being down a rook is "fine".
    score = 0.7 * brain_val + 0.3 * np.tanh(mat * 1.5)
    _EVAL_CACHE[key] = score
    return score

def _order_moves(board: chess.Board):
    """Cheap move ordering: captures & promotions first, then checks, then rest.
    Better ordering -> more alpha-beta cutoffs -> much faster search."""
    def key(mv):
        score = 0
        if board.is_capture(mv):
            victim = board.piece_at(mv.to_square)
            attacker = board.piece_at(mv.from_square)
            v_val = _PIECE_VAL.get(victim.piece_type, 0) if victim else 1  # ep
            a_val = _PIECE_VAL.get(attacker.piece_type, 0) if attacker else 0
            score += 100 + 10 * v_val - a_val   # MVV-LVA
        if mv.promotion:
            score += 90
        if board.gives_check(mv):
            score += 5
        return -score   # sort ascending -> best first
    return sorted(board.legal_moves, key=key)

def _alphabeta(brain: Model, board: chess.Board, depth: int,
               alpha: float, beta: float, q_depth: int = 2) -> float:
    """Negamax alpha-beta. Returns score from side-to-move POV in [-1, 1]."""
    term = _terminal_score(board)
    if term is not None:
        return term

    if depth <= 0:
        # Quiescence: keep searching noisy moves (captures / promotions) so
        # we don't cut off in the middle of an exchange.
        stand_pat = _leaf_eval(brain, board)
        if q_depth <= 0:
            return stand_pat
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
        for mv in _order_moves(board):
            if not (board.is_capture(mv) or mv.promotion):
                continue
            board.push(mv)
            val = -_alphabeta(brain, board, 0, -beta, -alpha, q_depth - 1)
            board.pop()
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val
        return alpha

    for mv in _order_moves(board):
        board.push(mv)
        val = -_alphabeta(brain, board, depth - 1, -beta, -alpha, q_depth)
        board.pop()
        if val >= beta:
            return beta
        if val > alpha:
            alpha = val
    return alpha


def choose_move(brain: Model, board: chess.Board, temperature: float = 0.4,
                search_depth: int = 2):
    """Search each root move `search_depth` plies deep with the brain as the
    leaf evaluator. Softmax-sample among the top scorers using `temperature`
    so games aren't deterministic; temperature=0 = strictly greedy."""
    legal = list(board.legal_moves)
    if not legal:
        return None, None, None

    # Fresh cache per move — positions rarely repeat across different roots
    # and the cache would just grow unbounded.
    _EVAL_CACHE.clear()

    pre_move_encoding = encode_board(board)
    root_scores = np.empty(len(legal), dtype=np.float32)

    if search_depth <= 0:
        # Cheap one-ply path (legacy behavior).
        batch = np.zeros((len(legal), 8, 8, 13), dtype=np.float32)
        for i, mv in enumerate(_order_moves(board)):
            board.push(mv); batch[i] = encode_board(board); board.pop()
        opp = brain.predict(batch, verbose=0).flatten()
        legal = _order_moves(board)
        root_scores = -opp
    else:
        alpha, beta = -2.0, 2.0
        ordered = _order_moves(board)
        legal = ordered
        for i, mv in enumerate(ordered):
            board.push(mv)
            # After we move, it's opponent's turn — negate their score.
            val = -_alphabeta(brain, board, search_depth - 1,
                              -beta, -alpha, q_depth=2)
            board.pop()
            root_scores[i] = val
            if val > alpha:
                alpha = val   # narrows the window for later root moves

    if temperature > 1e-6:
        logits = root_scores / temperature
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        idx = int(np.random.choice(len(legal), p=probs))
    else:
        idx = int(np.argmax(root_scores))

    chosen = legal[idx]
    return chosen, pre_move_encoding, float(root_scores[idx])


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
def format_history(history, scoreboard) -> str:
    """Multi-line history block printed above the board on every frame."""
    lines = []
    lines.append(BOLD + "=" * 60 + RESET)
    lines.append(BOLD + "Game history" + RESET)
    if not history:
        lines.append("  (no games completed yet)")
    else:
        for entry in history:
            lines.append("  " + entry)
    lines.append(BOLD +
        f"Total  |  Player 1: {scoreboard['P1_wins']} wins  |  "
        f"Player 2: {scoreboard['P2_wins']} wins  |  "
        f"Draws: {scoreboard['draws']}" + RESET)
    lines.append(BOLD + "=" * 60 + RESET)
    return "\n".join(lines)


def play_game(brain_white: Model, brain_black: Model, game_num: int,
              delay: float, max_plies: int = 400, temperature: float = 0.4,
              search_depth: int = 2, history=None, scoreboard=None):
    board = chess.Board()
    history = history if history is not None else []
    scoreboard = scoreboard if scoreboard is not None else \
        {"P1_wins": 0, "P2_wins": 0, "draws": 0}

    # Per-player memory: list of (encoded_position, immediate_material_after)
    memory = {chess.WHITE: [], chess.BLACK: []}

    ply = 0
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        mover = board.turn
        brain = brain_white if mover == chess.WHITE else brain_black
        t0 = time.time()
        move, pre_enc, score = choose_move(brain, board, temperature=temperature,
                                           search_depth=search_depth)
        move_ms = (time.time() - t0) * 1000
        if move is None:
            break

        board.push(move)
        material_after = material_balance(board, mover)
        memory[mover].append((pre_enc, material_after))

        clear_screen()
        # Persistent history block at the top of every frame.
        print(format_history(history, scoreboard))
        who = "Player 1 Brain (White)" if mover == chess.WHITE else "Player 2 Brain (Black)"
        header = (f"Game {game_num}  |  Ply {ply+1}  |  {who} plays {move.uci()}"
                  f"  |  eval {score:+.3f}  |  {move_ms:.0f} ms  |  d={search_depth}")
        render_board(board, header=header)
        # Show draw-pressure info so it's obvious WHY a game ended.
        print(f"\nFEN: {board.fen()}")
        print(f"halfmove clock (50-move rule): {board.halfmove_clock}"
              f"   |  can_claim_threefold: {board.can_claim_threefold_repetition()}"
              f"   |  in check: {board.is_check()}")
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
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="exploration temperature; 0 = greedy")
    parser.add_argument("--search-depth", type=int, default=2,
                        help="alpha-beta search depth in plies (0 = no search, "
                             "pure one-ply greedy). 2-3 is a good tradeoff; "
                             "4+ gets slow without a GPU.")
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
    history = []  # list of pre-formatted per-game summary lines

    for g in range(1, args.games + 1):
        memory, outcome, result = play_game(
            brain_p1, brain_p2, game_num=g,
            delay=args.delay, max_plies=args.max_plies,
            temperature=args.temperature,
            search_depth=args.search_depth,
            history=history, scoreboard=scoreboard,
        )

        # Update running scoreboard + append a history line in the requested
        # "Game N: player 1 win, player 2 loss" form.
        if result == "1-0":
            scoreboard["P1_wins"] += 1
            line = (f"Game {g}: player 1 WIN, player 2 loss   "
                    f"(result {result})")
        elif result == "0-1":
            scoreboard["P2_wins"] += 1
            line = (f"Game {g}: player 1 loss, player 2 WIN   "
                    f"(result {result})")
        else:
            scoreboard["draws"] += 1
            # `*` means the ply cap hit; a real drawn result claims 1/2-1/2.
            tag = "unfinished (ply cap)" if result == "*" else "drawn"
            line = (f"Game {g}: draw ({tag})           "
                    f"(result {result})")
        history.append(line)

        # Sanity check: foundation weights unchanged before training
        found_before = [w.numpy().copy() for w in foundation.weights]

        print("\nTraining learning heads (foundation stays frozen)...")
        train_head(brain_p1, memory[chess.WHITE], outcome[chess.WHITE], "P1")
        train_head(brain_p2, memory[chess.BLACK], outcome[chess.BLACK], "P2")

        # Verify frozen foundation didn't move
        found_after = [w.numpy() for w in foundation.weights]
        drift = max(np.max(np.abs(a - b)) for a, b in zip(found_before, found_after))
        print(f"  foundation weight drift: {drift:.2e}  (should be 0)")

        # Persistent running summary printed after each game too, so it stays
        # visible in the scrollback even when the next game starts clearing.
        print()
        print(format_history(history, scoreboard))
        print("-" * 60)

    print("\nFinal scoreboard:")
    print(f"  Player 1 Brain wins: {scoreboard['P1_wins']}")
    print(f"  Player 2 Brain wins: {scoreboard['P2_wins']}")
    print(f"  Draws:               {scoreboard['draws']}")


if __name__ == "__main__":
    main()

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
# Terminal rendering — SNES/16-bit pixel-art style using 24-bit truecolor
# ---------------------------------------------------------------------------
# Marble palette. Real marble has three visual layers:
#   1. A base tone (creamy off-white for Carrara, deep charcoal for Nero)
#   2. Subtle low-frequency shading (large soft gradients from mineral density)
#   3. Sharp high-frequency veins (calcite/quartz threads winding through)
# We synthesize (2) with cheap value-noise and (3) with a "turbulence" pattern.
# ---------------------------------------------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"

# Carrara-style light marble: warm cream base, faint gray veins
LIGHT_MARBLE_BASE = (245, 240, 230)
LIGHT_MARBLE_DARK = (220, 213, 200)   # subtle shading, not stormy
LIGHT_VEIN        = (135, 128, 118)   # medium gray veins
# Nero Marquina-style dark marble: near-black base, sharp white veins
DARK_MARBLE_BASE  = ( 32,  28,  26)
DARK_MARBLE_DARK  = ( 15,  12,  11)
DARK_VEIN         = (200, 195, 185)   # bright calcite veins

BEVEL_HI    = (255, 245, 225)   # crisp top/left edge (polished stone)
BEVEL_LO    = ( 15,  10,   8)   # near-black bottom/right edge
BORDER      = ( 25,  20,  18)   # frame around board
LABEL_BG    = ( 15,  10,   8)
LABEL_FG    = (230, 220, 200)

# Piece / board material palette.
# Mode: CHESS_MATERIAL=wood|marble (default wood).
#   wood   → light-oak / dark-walnut pieces + matching wood board
#   marble → Carrara / Nero Marquina pieces + matching marble board
WP_O = ( 55,  50,  45); WP_S = (165, 155, 140); WP_M = (215, 208, 195)
WP_B = (240, 233, 220); WP_H = (255, 252, 245); WP_J = (185,  40,  50)  # deep ruby
BP_O = (  0,   0,   0); BP_S = ( 12,  10,  10); BP_M = ( 42,  38,  38)
BP_B = ( 78,  72,  72); BP_H = (140, 130, 130); BP_J = ( 60, 130, 200)  # deep sapphire
WP_VEIN = ( 90,  85,  80)
BP_VEIN = (210, 205, 195)

# Wood colours (pieces + board when material == "wood")
LIGHT_WOOD_BASE  = (210, 170, 110)   # honey oak
LIGHT_WOOD_DARK  = (160, 115,  65)   # deeper oak rings
LIGHT_WOOD_GRAIN = (120,  80,  40)   # fine dark grain lines
DARK_WOOD_BASE   = ( 55,  32,  18)   # walnut body
DARK_WOOD_DARK   = ( 28,  16,   8)   # near-ebony rings
DARK_WOOD_GRAIN  = ( 90,  55,  30)   # lighter grain highlights

# Material mode + render size (fit ~800x800 GNOME terminal by default)
_MATERIAL = os.environ.get("CHESS_MATERIAL", "wood").strip().lower()
if _MATERIAL not in ("wood", "marble"):
    _MATERIAL = "wood"
# SQ = pixels per square edge in the raytraced image.
# Terminal footprint ≈ (8*SQ+frame)/2 columns and rows (X½ + ▀).
# SQ=16 → ~70×70 cells → fits an ~800×800 terminal window.
SQ = int(os.environ.get("CHESS_SQ", "16"))
SQ = max(8, min(SQ, 48))

def _fg(rgb): return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
def _bg(rgb): return f"\033[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


# ---------------------------------------------------------------------------
# Procedural textures: marble (veins) and wood (growth rings + grain).
# One big sheet is baked at init; each square/piece samples a unique offset.
# ---------------------------------------------------------------------------
def _value_noise_2d(h: int, w: int, cell: int, rng) -> np.ndarray:
    """Cheap 2D value noise: random grid of values, bilinear-interpolated.
    Returns a (h, w) float array in [0, 1]."""
    gh = h // cell + 2
    gw = w // cell + 2
    grid = rng.random((gh, gw)).astype(np.float32)
    ys = np.arange(h, dtype=np.float32) / cell
    xs = np.arange(w, dtype=np.float32) / cell
    y0 = ys.astype(np.int32);  y1 = y0 + 1
    x0 = xs.astype(np.int32);  x1 = x0 + 1
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    g00 = grid[np.ix_(y0, x0)]; g10 = grid[np.ix_(y1, x0)]
    g01 = grid[np.ix_(y0, x1)]; g11 = grid[np.ix_(y1, x1)]
    top = g00 * (1 - fx) + g01 * fx
    bot = g10 * (1 - fx) + g11 * fx
    return top * (1 - fy) + bot * fy

def _turbulence(h: int, w: int, rng, octaves: int = 5) -> np.ndarray:
    """Fractal noise: sum of value-noise at halving scales."""
    out = np.zeros((h, w), dtype=np.float32)
    amp = 1.0
    cell = max(h, w) // 2
    for _ in range(octaves):
        out += amp * _value_noise_2d(h, w, max(cell, 2), rng)
        amp *= 0.5
        cell = max(cell // 2, 2)
    out = (out - out.min()) / max(out.max() - out.min(), 1e-6)
    return out

def _make_marble(h: int, w: int, base_rgb, dark_rgb, vein_rgb, seed: int,
                 vein_sharpness: float = 8.0, vein_freq: float = 1.5,
                 vein_alpha_max: float = 0.4):
    """Return an (h, w, 3) uint8 marble texture."""
    rng = np.random.default_rng(seed)
    base_noise = _turbulence(h, w, rng, octaves=4)
    turb = _turbulence(h, w, rng, octaves=5)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    coord = (xs / w + 0.5 * ys / h) * (vein_freq * 2 * np.pi) + turb * 6.0
    vein_mask = np.abs(np.sin(coord))
    vein_mask = np.clip(vein_mask ** vein_sharpness, 0, 1)
    base = np.array(base_rgb, dtype=np.float32)
    dark = np.array(dark_rgb, dtype=np.float32)
    vein = np.array(vein_rgb, dtype=np.float32)
    body = base[None, None, :] * base_noise[..., None] + \
           dark[None, None, :] * (1 - base_noise[..., None])
    va = (1.0 - vein_mask)[..., None] * vein_alpha_max
    out = body * (1 - va) + vein[None, None, :] * va
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_wood(h: int, w: int, base_rgb, dark_rgb, grain_rgb, seed: int,
               ring_freq: float = 14.0, grain_strength: float = 0.55,
               ring_warp: float = 2.8):
    """Procedural wood: concentric growth rings + fine longitudinal grain."""
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * (0.45 + 0.1 * rng.random())
    cy = h * (0.40 + 0.2 * rng.random())
    warp = _turbulence(h, w, rng, octaves=4)
    dx = (xs - cx) / max(w, 1)
    dy = (ys - cy) / max(h, 1)
    r = np.sqrt(dx * dx + dy * dy) + (warp - 0.5) * ring_warp * 0.08
    rings = np.sin(r * ring_freq * 2.0 * np.pi)
    rings = 0.5 + 0.5 * rings
    rings = np.clip(rings ** 1.4, 0, 1)
    grain_noise = _turbulence(h, w, rng, octaves=6)
    grain = np.sin((ys / h * 40.0 + grain_noise * 8.0) * np.pi)
    grain = 0.5 + 0.5 * grain
    grain = np.clip(grain ** 2.2, 0, 1)
    base = np.array(base_rgb, dtype=np.float32)
    dark = np.array(dark_rgb, dtype=np.float32)
    grain_c = np.array(grain_rgb, dtype=np.float32)
    body = base[None, None, :] * (1 - rings[..., None]) + \
           dark[None, None, :] * rings[..., None]
    ga = grain[..., None] * grain_strength * 0.35
    out = body * (1 - ga) + grain_c[None, None, :] * ga
    val = _value_noise_2d(h, w, max(h // 8, 4), rng)
    out = out * (0.92 + 0.16 * val[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


# Pre-baked texture sheets (filled by _init_textures)
_BOARD_LIGHT = None
_BOARD_DARK  = None
_PIECE_LIGHT = None
_PIECE_DARK  = None

def _init_textures():
    """Bake board + piece sheets for the active material mode."""
    global _BOARD_LIGHT, _BOARD_DARK, _PIECE_LIGHT, _PIECE_DARK
    if _BOARD_LIGHT is not None:
        return
    board_h = 8 * SQ + 8
    board_w = board_h
    piece_sz = max(SQ * 3, 48)
    if _MATERIAL == "wood":
        _BOARD_LIGHT = _make_wood(
            board_h, board_w, LIGHT_WOOD_BASE, LIGHT_WOOD_DARK, LIGHT_WOOD_GRAIN,
            seed=31, ring_freq=9.0, grain_strength=0.45, ring_warp=2.4)
        _BOARD_DARK = _make_wood(
            board_h, board_w, DARK_WOOD_BASE, DARK_WOOD_DARK, DARK_WOOD_GRAIN,
            seed=32, ring_freq=10.0, grain_strength=0.40, ring_warp=2.2)
        _PIECE_LIGHT = _make_wood(
            piece_sz, piece_sz, LIGHT_WOOD_BASE, LIGHT_WOOD_DARK, LIGHT_WOOD_GRAIN,
            seed=21, ring_freq=11.0, grain_strength=0.6, ring_warp=3.2)
        _PIECE_DARK = _make_wood(
            piece_sz, piece_sz, DARK_WOOD_BASE, DARK_WOOD_DARK, DARK_WOOD_GRAIN,
            seed=22, ring_freq=13.0, grain_strength=0.5, ring_warp=2.6)
    else:
        _BOARD_LIGHT = _make_marble(
            board_h, board_w, LIGHT_MARBLE_BASE, LIGHT_MARBLE_DARK, LIGHT_VEIN,
            seed=1, vein_sharpness=16.0, vein_freq=3.5, vein_alpha_max=0.35)
        _BOARD_DARK = _make_marble(
            board_h, board_w, DARK_MARBLE_BASE, DARK_MARBLE_DARK, DARK_VEIN,
            seed=2, vein_sharpness=22.0, vein_freq=4.0, vein_alpha_max=0.30)
        _PIECE_LIGHT = _make_marble(
            piece_sz, piece_sz, WP_B, WP_M, WP_VEIN,
            seed=11, vein_sharpness=16.0, vein_freq=4.0, vein_alpha_max=0.28)
        _PIECE_DARK = _make_marble(
            piece_sz, piece_sz, BP_M, BP_S, BP_VEIN,
            seed=12, vein_sharpness=22.0, vein_freq=4.5, vein_alpha_max=0.22)

# Back-compat alias
def _init_marble():
    _init_textures()

# ---- Sprite atlas ---------------------------------------------------------
# 32x32 grids. Legend:
#   '.' transparent   'O' outline    'S' shadow
#   'M' midtone body  'B' base       'H' highlight    'J' jewel accent
SPRITES = {
    "P": (
        "................................",
        "................................",
        "................................",
        "................................",
        ".............OOOOO..............",
        "............OSMMMSO.............",
        "...........OSMBBBMSO............",
        "...........OMBBHBBMO............",
        "...........OMBBBBBMO............",
        "............OSMMMSO.............",
        ".............OOOOO..............",
        "...........OSMMMMMSO............",
        "..........OSMBBBBBMSO...........",
        ".........OSMBBHHHBBMSO..........",
        ".........OMBBBBBBBBBMO..........",
        "..........OSMBBBBBMSO...........",
        "...........OSMMMMMSO............",
        "..........OOOOOOOOOOO...........",
        ".........OSMMMMMMMMMSO..........",
        "........OSMBBBBBBBBBBMSO........",
        ".......OSMBBHHHHHBBBBBMSO.......",
        ".......OMBBBBBBBBBBBBBMO........",
        ".......OMBBBBBBBBBBBBBMO........",
        ".......OSMMBBBBBBBBBBMMSO.......",
        "........OSMMMMMMMMMMMMSO........",
        ".......OSSSSSSSSSSSSSSSSO.......",
        "......OSMMMMMMMMMMMMMMMMMSO.....",
        ".....OMBBBHHHHHBBBBBBBBBBBBMO...",
        ".....OMBBBBBBBBBBBBBBBBBBBBMO...",
        ".....OSMMMMMMMMMMMMMMMMMMMMSO...",
        ".....OOOOOOOOOOOOOOOOOOOOOOOO...",
        "................................",
    ),
    "N": (
        "................................",
        "................................",
        "..............OOOOO.............",
        ".............OSMMMMOO...........",
        "............OSMBBBBMOO..........",
        "...........OSMBBHHBBMO..........",
        "..........OSMBBHHHHBBMO.........",
        ".........OSMBBBBHHHHBBMO........",
        ".........OMBBBBBBHHHBBBMO.......",
        "........OSMBBBBBBBHHBBBMO.......",
        "........OMMMMBBBBBBBBBBBMO......",
        "........OMOOMMMMMBBBBBBBMO......",
        ".......OMOSSMMMMSMMMBBBBBMO.....",
        ".......OMOSMSMMSMMMMMBBBBBMO....",
        ".......OOOOSMSMSMMMMMMMBBBMO....",
        "..........OSMMSMMMMMMMMBBBMO....",
        "..........OSMBSMSMMMMMMMBBBMO...",
        ".........OMBBBBSMMMMMMMMBBBMO...",
        ".........OMBBHHBBSMMMMMMBBBMO...",
        "........OMBBHHHHBBBBMMMMBBBBMO..",
        "........OMBBBHHHHBBBBBMMMBBBBMO.",
        "........OMBBBBHHHBBBBBBBBBBBBMO.",
        "........OMBBBBBBBBBBBBBBBBBBBMO.",
        ".......OMBBBBBBBBBBBBBBBBBBBBBMO",
        ".......OMBBBBBBBBBBBBBBBBBBBBBMO",
        ".......OSMMMMMMMMMMMMMMMMMMMMMSO",
        "......OSMMBBBBBBBBBBBBBBBBBBBMMO",
        "......OMBBBHHHHHBBBBBBBBBBBBBBMO",
        "......OMBBBBBBBBBBBBBBBBBBBBBBMO",
        ".....OMMSSSSSSSSSSSSSSSSSSSSSSMO",
        ".....OOOOOOOOOOOOOOOOOOOOOOOOOOO",
        "................................",
    ),
    "B": (
        "................................",
        "................................",
        "...............OOO..............",
        "..............OJJJO.............",
        ".............OOJJJOO............",
        ".............OJJOJJO............",
        ".............OJJJJJO............",
        ".............OOJJJOO............",
        "..............OJJJO.............",
        "..............OOOOO.............",
        "............OSMMMMMSO...........",
        "...........OSMBBBBBMSO..........",
        "..........OSMBBHHHBBBMO.........",
        ".........OSMBBBHHHBBBBMO........",
        ".........OMBBBHHHHHBBBBMO.......",
        ".........OMBBBBHHHBBBBBMO.......",
        "........OSMBBBBBBBBBBBBMO.......",
        "........OMBBBBBBBBBBBBBMSO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OSMMBBBBBBBBBBBMSO......",
        ".........OSMMMMMMMMMMMMSO.......",
        "..........OOOOOOOOOOOOOO........",
        ".........OSMMMMMMMMMMMMMSO......",
        "........OSMBBBBBBBBBBBBBBMO.....",
        ".......OSMBBBHHHHHBBBBBBBBMO....",
        ".......OMBBBBBBBBBBBBBBBBBMO....",
        ".......OMBBBBBBBBBBBBBBBBBMO....",
        "......OSMMMMMMMMMMMMMMMMMMMSO...",
        "......OSSSSSSSSSSSSSSSSSSSSSSO..",
        ".....OMBBBHHHHHBBBBBBBBBBBBBBMO.",
        ".....OMBBBBBBBBBBBBBBBBBBBBBBMO.",
        ".....OOOOOOOOOOOOOOOOOOOOOOOOOO.",
    ),
    "R": (
        "................................",
        "................................",
        "................................",
        ".......OOO..OOO..OOO..OOO.......",
        ".......OMO..OMO..OMO..OMO.......",
        ".......OMOOOOMOOOOMOOOOMO.......",
        ".......OMMMMMMMMMMMMMMMMMO......",
        ".......OMBBBBBBBBBBBBBBBBMO.....",
        ".......OMBHHHHHHHBBBBBBBBMO.....",
        ".......OMBBBBBBBBBBBBBBBBMO.....",
        ".......OSMMMMMMMMMMMMMMMMSO.....",
        "........OOOOOOOOOOOOOOOOOO......",
        "........OSMMMMMMMMMMMMMMSO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OMBBHHHHHHBBBBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OMBBBBBHHHHHBBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OSMMMMMMMMMMMMMMSO......",
        ".........OOOOOOOOOOOOOOOO.......",
        ".......OSMMMMMMMMMMMMMMMMMMSO...",
        ".......OMBBBBBBBBBBBBBBBBBBBMO..",
        ".......OMBBHHHHHHHBBBBBBBBBBMO..",
        ".......OMBBBBBBBBBBBBBBBBBBBMO..",
        ".......OSSSSSSSSSSSSSSSSSSSSSO..",
        "......OMBBHHHHHHHBBBBBBBBBBBBMO.",
        "......OOOOOOOOOOOOOOOOOOOOOOOOO.",
    ),
    "Q": (
        "................................",
        "......O.....O.....O.....O.......",
        ".....OJO...OJO...OJO...OJO......",
        ".....OJO...OJO...OJO...OJO......",
        ".....OJO...OJO...OJO...OJO......",
        ".....OJOOOOOJOOOOOJOOOOOJO......",
        ".....OJJJJJJJJJJJJJJJJJJJJO.....",
        ".....OMMMMMMMMMMMMMMMMMMMMO.....",
        ".....OMBHHBBBHHBBBHHBBBHHBMO....",
        ".....OMBBBBBBBBBBBBBBBBBBBMO....",
        ".....OSMBBBBBBBBBBBBBBBBBMSO....",
        "......OSMMBBBBBBBBBBBBBMMSO.....",
        ".......OSMMMBBBBBBBBBMMMSO......",
        "........OSMMMMBBBBBMMMMSO.......",
        ".........OSMMMMBBBMMMMSO........",
        "..........OSMMMMMMMMMSO.........",
        "..........OOOOOOOOOOOOO.........",
        "..........OSMMMMMMMMMSO.........",
        ".........OSMBBBBBBBBBBMSO.......",
        "........OSMBBBHHHHHBBBBMSO......",
        ".......OSMBBBBBHHHBBBBBBMSO.....",
        ".......OMBBHHBBBBBBBBHHBBMO.....",
        ".......OMBBBBBBBBBBBBBBBBMO.....",
        ".......OSMBBBBBBBBBBBBBBMSO.....",
        "........OSMMMMMMMMMMMMMMSO......",
        "........OSSSSSSSSSSSSSSSSO......",
        "......OMBBBHHHHHBBBBBBBBBBMO....",
        "......OMBBBBBBBBBBBBBBBBBBMO....",
        ".....OSMMMMMMMMMMMMMMMMMMMMSO...",
        ".....OSSSSSSSSSSSSSSSSSSSSSSO...",
        "....OMBHHHHHHHBBBBBBBBBBBBBBMO..",
        "....OOOOOOOOOOOOOOOOOOOOOOOOOO..",
    ),
    "K": (
        "................................",
        "................................",
        "..............OOOOO.............",
        ".............OJJJJJO............",
        ".............OJJJJJO............",
        ".........OOOOOJJJJJOOOOO........",
        ".........OJJJJJJJJJJJJJO........",
        ".........OJJJJJJJJJJJJJO........",
        ".........OOOOOOJJJOOOOOO........",
        ".............OSMMMSO............",
        "............OSMBBBMSO...........",
        "...........OSMMBBBMMSO..........",
        "..........OSMBBHHHBBMSO.........",
        ".........OSMBBBHHHBBBMSO........",
        ".........OMBBBBHHHBBBBMO........",
        ".........OMBBBBBBBBBBBMO........",
        ".........OSMBBBBBBBBBMSO........",
        "..........OSMMBBBBBMMSO.........",
        "...........OSMMMMMMMSO..........",
        "..........OSMMMMMMMMMSO.........",
        ".........OSMBBBBBBBBBBMSO.......",
        "........OSMBBHHHHHHHBBBMSO......",
        "........OMBBBHHHHHHHHBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OMBBBBBBBBBBBBBBMO......",
        "........OSMMMMMMMMMMMMMMSO......",
        ".......OSMMMMMMMMMMMMMMMMSO.....",
        "......OSMBBBBBBBBBBBBBBBBBMO....",
        "......OMBBHHHHHHHBBBBBBBBBMO....",
        "......OSSSSSSSSSSSSSSSSSSSSO....",
        ".....OMBBHHHHHHHHBBBBBBBBBBMO...",
        ".....OOOOOOOOOOOOOOOOOOOOOOOO...",
    ),
}

# Build integer sprite masks once
_NATIVE_SPRITE = 32   # SPRITES atlas is authored at 32x32

def _compile_sprites():
    """Parse 32x32 atlas, then nearest-neighbor scale to SQ for TG-16 look."""
    out = {}
    for k, rows in SPRITES.items():
        m32 = np.zeros((_NATIVE_SPRITE, _NATIVE_SPRITE), dtype=np.int8)
        for r, row in enumerate(rows):
            if r >= _NATIVE_SPRITE:
                break
            for c, ch in enumerate(row.ljust(_NATIVE_SPRITE)[:_NATIVE_SPRITE]):
                if   ch == "O": m32[r, c] = 1
                elif ch == "S": m32[r, c] = 2
                elif ch == "M": m32[r, c] = 3
                elif ch == "B": m32[r, c] = 4
                elif ch == "H": m32[r, c] = 5
                elif ch == "J": m32[r, c] = 6
        if SQ == _NATIVE_SPRITE:
            out[k] = m32
        else:
            # Nearest-neighbor scale → crisp 16-bit pixels, no blur
            ys = (np.arange(SQ) * _NATIVE_SPRITE // SQ)
            xs = (np.arange(SQ) * _NATIVE_SPRITE // SQ)
            out[k] = m32[ys][:, xs]
    return out

_SPRITE_MASKS = None
def _sprite(kind: str):
    global _SPRITE_MASKS
    if _SPRITE_MASKS is None:
        _SPRITE_MASKS = _compile_sprites()
    return _SPRITE_MASKS[kind]


# ---------------------------------------------------------------------------
# 2D 16-bit (TurboGrafx-16 / SNES) board renderer
#
# Flat top-down view, crisp nearest-neighbor sprites, bevelled squares,
# limited-palette feel with dither on dark squares. No perspective, no
# raytracing — pure pixel art that fits an ~800x800 terminal at SQ=16.
# ---------------------------------------------------------------------------

# 16-bit style fixed palettes (piece body layers)
_PAL_WHITE = {
    1: (40,  35,  30),     # outline
    2: (150, 140, 125),    # shadow
    3: (200, 190, 175),    # midtone
    4: (235, 228, 215),    # base
    5: (255, 250, 240),    # highlight
    6: (200,  45,  55),    # jewel ruby
}
_PAL_BLACK = {
    1: (0,    0,   0),
    2: (18,  15,  14),
    3: (48,  42,  40),
    4: (78,  70,  68),
    5: (120, 110, 105),
    6: (55, 120, 190),     # jewel sapphire
}
# When material is wood, shift white/black piece palettes toward oak/walnut
_PAL_OAK = {
    1: (55,  35,  15),
    2: (130,  95,  50),
    3: (175, 135,  80),
    4: (210, 170, 110),
    5: (240, 210, 155),
    6: (185,  40,  50),
}
_PAL_WALNUT = {
    1: (8,    4,   2),
    2: (28,  16,   8),
    3: (48,  28,  16),
    4: (70,  42,  24),
    5: (100,  65,  40),
    6: (60, 130, 200),
}


def _piece_palette(color_white: bool):
    if _MATERIAL == "wood":
        return _PAL_OAK if color_white else _PAL_WALNUT
    return _PAL_WHITE if color_white else _PAL_BLACK


def _square_pixels(is_light: bool, file: int, rank: int):
    """2D square fill with texture + 16-bit bevel (top/left hi, bottom/right lo)."""
    _init_textures()
    sheet = _BOARD_LIGHT if is_light else _BOARD_DARK
    max_off = max(sheet.shape[0] - SQ, 0)
    y0 = (rank * 7 + file * 13) % (max_off + 1)
    x0 = (rank * 11 + file * 5 + 3) % (max_off + 1)
    px = sheet[y0:y0 + SQ, x0:x0 + SQ].copy()
    # Classic 16-bit bevel (1px)
    px[0, :]  = BEVEL_HI
    px[:, 0]  = BEVEL_HI
    px[SQ - 1, :] = BEVEL_LO
    px[:, SQ - 1] = BEVEL_LO
    # Optional corner soften so bevel meets cleanly
    px[0, SQ - 1] = tuple((a + b) // 2 for a, b in zip(BEVEL_HI, BEVEL_LO))
    px[SQ - 1, 0] = tuple((a + b) // 2 for a, b in zip(BEVEL_HI, BEVEL_LO))
    return px


def _paint_piece(sq_px, kind, color_white, file, rank):
    """Composite sprite onto square pixels (in-place). Transparent '.' skipped."""
    mask = _sprite(kind)
    pal = _piece_palette(color_white)
    # Wood mode: tint sprite body slightly with procedural grain from piece sheet
    if _MATERIAL == "wood":
        _init_textures()
        sheet = _PIECE_LIGHT if color_white else _PIECE_DARK
        H, W = sheet.shape[:2]
        off_y = (rank * 17 + file * 23 + 7) % max(H - SQ, 1)
        off_x = (file * 29 + rank * 13 + 5) % max(W - SQ, 1)
        grain = sheet[off_y:off_y + SQ, off_x:off_x + SQ]
        if grain.shape[0] != SQ or grain.shape[1] != SQ:
            grain = None
    else:
        grain = None

    for r in range(SQ):
        for c in range(SQ):
            layer = int(mask[r, c])
            if layer == 0:
                continue
            col = pal[layer]
            if grain is not None and layer in (2, 3, 4, 5):
                # Blend 25% grain into body layers for carved-wood feel
                g = grain[r, c].astype(np.float32)
                col = tuple(int(0.75 * col[i] + 0.25 * g[i]) for i in range(3))
            sq_px[r, c] = col
    return sq_px


def _render_board_2d(board: chess.Board) -> np.ndarray:
    """Build a flat 2D 16-bit style board image (H, W, 3) uint8."""
    _init_textures()
    H = W = 8 * SQ
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for rank in range(8):
        for file in range(8):
            is_light = ((rank + file) % 2) == 1
            sq = _square_pixels(is_light, file, rank)
            piece = board.piece_at(chess.square(file, rank))
            if piece:
                kind = piece.symbol().upper()
                _paint_piece(sq, kind, piece.color == chess.WHITE, file, rank)
            # rank 0 is near bottom of image (white's side) → row 7 in image coords
            y0 = (7 - rank) * SQ
            x0 = file * SQ
            img[y0:y0 + SQ, x0:x0 + SQ] = sq
    # Thin outer frame
    frame = 2
    out = np.full((H + 2 * frame, W + 2 * frame, 3), BORDER, dtype=np.uint8)
    out[frame:frame + H, frame:frame + W] = img
    return out


# ---------------------------------------------------------------------------
# 3D RENDERER — ray-marched primitives, Lambertian + specular shading,
# cast shadows onto the marble board.
#
# Coordinate frame (right-handed):
#   +x = file direction (a -> h)      board spans x in [0, 8]
#   +z = rank direction (1 -> 8)      board spans z in [0, 8]
#   +y = up                            board plane at y = 0
#
# Piece origin is centered on its square: (file + 0.5, 0, rank + 0.5).
# All piece heights are expressed as multiples of one square edge (=1.0).
# ---------------------------------------------------------------------------

# --- Primitive intersections (all return t along ray or np.inf) ------------
def _isect_plane_y(ro, rd, y):
    """Ray vs horizontal plane y=const. ro/rd shape (...,3)."""
    dy = rd[..., 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (y - ro[..., 1]) / dy
    t = np.where((dy < -1e-6) & (t > 1e-4), t, np.inf)
    return t

def _isect_sphere(ro, rd, center, radius):
    """Ray vs sphere. center shape (3,), radius scalar."""
    oc = ro - center
    b = np.einsum("...i,...i->...", oc, rd)
    c = np.einsum("...i,...i->...", oc, oc) - radius * radius
    disc = b * b - c
    hit = disc > 0
    sq = np.sqrt(np.where(hit, disc, 0))
    t0 = -b - sq
    t1 = -b + sq
    t = np.where((t0 > 1e-4) & hit, t0, np.inf)
    t = np.where((t == np.inf) & (t1 > 1e-4) & hit, t1, t)
    return t

def _isect_cylinder_y(ro, rd, cx, cz, radius, y_min, y_max):
    """Ray vs finite Y-axis cylinder. Returns t and whether the hit was on
    the side (True) or a cap (False) — we use this to compute the correct
    normal. Cheap: no caps rendered separately unless the ray misses the
    side."""
    dx = rd[..., 0]; dz = rd[..., 2]
    ox = ro[..., 0] - cx; oz = ro[..., 2] - cz
    a = dx * dx + dz * dz
    b = ox * dx + oz * dz
    c = ox * ox + oz * oz - radius * radius
    disc = b * b - a * c
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.where(disc > 0, disc, 0))
        t_side = (-b - sq) / np.where(a > 1e-12, a, 1e-12)
    y_at = ro[..., 1] + t_side * rd[..., 1]
    t_side = np.where((disc > 0) & (t_side > 1e-4) &
                      (y_at >= y_min) & (y_at <= y_max),
                      t_side, np.inf)
    # Cap intersections (top only — bottom is inside the board)
    t_top = _isect_plane_y(ro, rd, y_max)
    x_at = ro[..., 0] + t_top * rd[..., 0]
    z_at = ro[..., 2] + t_top * rd[..., 2]
    inside = (x_at - cx) ** 2 + (z_at - cz) ** 2 <= radius * radius
    t_top = np.where(inside & (t_top < np.inf), t_top, np.inf)
    return np.minimum(t_side, t_top), t_side <= t_top  # (t, is_side)

def _isect_cone_y(ro, rd, cx, cz, y_base, y_top, r_base, r_top):
    """Ray vs finite truncated cone along Y (frustum). Returns t (or inf).
    Solves the quadratic in cross-section radius r(y) = r_base + (y - y_base)/
    (y_top - y_base) * (r_top - r_base)."""
    dx = rd[..., 0]; dy = rd[..., 1]; dz = rd[..., 2]
    ox = ro[..., 0] - cx; oy = ro[..., 1] - y_base; oz = ro[..., 2] - cz
    h = y_top - y_base
    k = (r_top - r_base) / h              # radius slope
    # r(y) = r_base + k*(y - y_base). At intersection: x^2 + z^2 = r(y)^2
    # Let y = oy + t*dy, ry = r_base + k*(oy + t*dy)
    A = dx * dx + dz * dz - k * k * dy * dy
    B = ox * dx + oz * dz - k * dy * (r_base + k * oy)
    C = ox * ox + oz * oz - (r_base + k * oy) ** 2
    disc = B * B - A * C
    with np.errstate(divide="ignore", invalid="ignore"):
        sq = np.sqrt(np.where(disc > 0, disc, 0))
        t0 = (-B - sq) / np.where(np.abs(A) > 1e-12, A, 1e-12)
        t1 = (-B + sq) / np.where(np.abs(A) > 1e-12, A, 1e-12)
    def _keep(t):
        y_at = ro[..., 1] + t * dy
        return np.where((disc > 0) & (t > 1e-4) &
                        (y_at >= y_base) & (y_at <= y_top), t, np.inf)
    t0 = _keep(t0); t1 = _keep(t1)
    return np.minimum(t0, t1)

def _isect_torus_approx(ro, rd, cx, cy, cz, R, r):
    """Cheap 'torus' fake: bounding sphere with a null band. Not used —
    listed here so it's obvious we picked spheres/cylinders instead."""
    return np.inf

# --- Piece assembly --------------------------------------------------------
# Each piece is a list of primitives:
#   ("sphere", (dx,dy,dz), radius)
#   ("cyl",    (dx,dz), radius, y_min, y_max)
#   ("cone",   (dx,dz), y_base, y_top, r_base, r_top)
# All coordinates are relative to the square center at (fx, 0, fz).
def _piece_primitives(kind: str):
    # Common base for every piece: a wide short cylinder
    base = ("cyl", (0.0, 0.0), 0.34, 0.0, 0.10)
    plinth = ("cone", (0.0, 0.0), 0.10, 0.14, 0.34, 0.30)
    if kind == "P":
        return [
            base, plinth,
            ("cone", (0.0, 0.0), 0.14, 0.42, 0.28, 0.14),   # body
            ("sphere", (0.0, 0.50, 0.0), 0.13),              # head
        ]
    if kind == "R":
        return [
            base, plinth,
            ("cyl", (0.0, 0.0), 0.24, 0.14, 0.62),           # tower
            ("cyl", (0.0, 0.0), 0.28, 0.60, 0.72),           # top ring
            # Crenellations: 4 small cubes-approx as small cylinders
            ("cyl", ( 0.16, 0.00), 0.06, 0.72, 0.82),
            ("cyl", (-0.16, 0.00), 0.06, 0.72, 0.82),
            ("cyl", ( 0.00, 0.16), 0.06, 0.72, 0.82),
            ("cyl", ( 0.00,-0.16), 0.06, 0.72, 0.82),
        ]
    if kind == "N":  # knight — stylized L-shaped head
        return [
            base, plinth,
            ("cone", (0.0, 0.0), 0.14, 0.50, 0.28, 0.18),   # body/neck
            # Horse head: an elongated sphere (approx w/ two spheres)
            ("sphere", (0.00, 0.62,  0.00), 0.20),
            ("sphere", (0.00, 0.68, -0.16), 0.16),           # muzzle forward
            ("sphere", (0.00, 0.74, -0.24), 0.10),           # nose tip
            ("sphere", (0.00, 0.82,  0.08), 0.09),           # mane bump
        ]
    if kind == "B":
        return [
            base, plinth,
            ("cone", (0.0, 0.0), 0.14, 0.54, 0.26, 0.16),   # body
            ("cyl", (0.0, 0.0), 0.18, 0.54, 0.58),           # collar
            ("sphere", (0.0, 0.72, 0.0), 0.14),              # mitre body
            ("sphere", (0.0, 0.86, 0.0), 0.05),              # orb on top
        ]
    if kind == "Q":
        return [
            base, plinth,
            ("cone", (0.0, 0.0), 0.14, 0.60, 0.28, 0.16),   # body
            ("cyl", (0.0, 0.0), 0.22, 0.60, 0.66),           # crown base
            # Ring of 5 jewel spheres on the crown
            ("sphere", (0.00, 0.76, 0.00),  0.06),
            ("sphere", (0.15, 0.74, 0.00),  0.05),
            ("sphere", (-0.15, 0.74, 0.00), 0.05),
            ("sphere", (0.00, 0.74, 0.15),  0.05),
            ("sphere", (0.00, 0.74, -0.15), 0.05),
        ]
    if kind == "K":
        return [
            base, plinth,
            ("cone", (0.0, 0.0), 0.14, 0.66, 0.28, 0.16),   # body (tallest)
            ("cyl", (0.0, 0.0), 0.22, 0.66, 0.72),           # crown base
            ("sphere", (0.0, 0.78, 0.0), 0.08),              # crown ball
            # Cross on top: two thin cylinders
            ("cyl", (0.0, 0.0), 0.03, 0.82, 1.02),           # vertical
            ("cyl", (0.0, 0.0), 0.03, 0.92, 0.96),           # horizontal (approx w/ short thick cyl centered — visually reads as cross)
        ]
    return []

# --- Scene assembly + trace -------------------------------------------------
def _build_scene(board: chess.Board):
    """Flatten every piece into a list of world-space primitives with material
    tags ('W' white marble, 'B' black marble, 'J_R' ruby jewel, 'J_S' sapphire).
    Also record (file, rank) per primitive so marble sampling stays consistent
    per piece."""
    prims = []
    for rank in range(8):
        for file in range(8):
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            if not piece:
                continue
            kind = piece.symbol().upper()
            fx = file + 0.5
            # White (rank 0/1) should be NEAR the camera (small z).
            fz = rank + 0.5
            mat = "W" if piece.color == chess.WHITE else "B"
            jewel_mat = "J_R" if piece.color == chess.WHITE else "J_S"
            local = _piece_primitives(kind)
            # For queen and king, the jewel spheres on the crown get J_*
            for i, p in enumerate(local):
                is_jewel = (p[0] == "sphere" and
                            ((kind == "Q" and i >= 4) or
                             (kind == "K" and i in (5, 6)) or
                             (kind == "B" and i == len(local) - 1)))
                m = jewel_mat if is_jewel else mat
                if p[0] == "sphere":
                    _, (dx, dy, dz), r = p
                    prims.append(("sphere", fx + dx, dy, fz + dz, r, m, file, rank))
                elif p[0] == "cyl":
                    _, (dx, dz), r, ymin, ymax = p
                    prims.append(("cyl", fx + dx, fz + dz, r, ymin, ymax, m, file, rank))
                elif p[0] == "cone":
                    _, (dx, dz), yb, yt, rb, rt = p
                    prims.append(("cone", fx + dx, fz + dz, yb, yt, rb, rt, m, file, rank))
    return prims


def _intersect_scene(ro, rd, prims):
    """Return (t, prim_idx) per ray. prim_idx = -2 for board, -1 for miss."""
    t_best = _isect_plane_y(ro, rd, 0.0)
    idx_best = np.where(t_best < np.inf, -2, -1).astype(np.int32)
    for i, p in enumerate(prims):
        if p[0] == "sphere":
            _, cx, cy, cz, r, _mat, _f, _rk = p
            t = _isect_sphere(ro, rd, np.array([cx, cy, cz], dtype=np.float32), r)
        elif p[0] == "cyl":
            _, cx, cz, r, ymin, ymax, _mat, _f, _rk = p
            t, _side = _isect_cylinder_y(ro, rd, cx, cz, r, ymin, ymax)
        else:  # cone
            _, cx, cz, yb, yt, rb, rt, _mat, _f, _rk = p
            t = _isect_cone_y(ro, rd, cx, cz, yb, yt, rb, rt)
        closer = t < t_best
        t_best = np.where(closer, t, t_best)
        idx_best = np.where(closer, i, idx_best)
    return t_best, idx_best


def _shadow_ray(hit_pt, light_dir, prims):
    """Returns True per pixel where the hit point is in shadow.
    Cheap: only tests occlusion by pieces (board never shadows the board)."""
    ro = hit_pt + light_dir * 0.01
    n = ro.shape[0]
    in_shadow = np.zeros(n, dtype=bool)
    for p in prims:
        if in_shadow.all():
            break
        if p[0] == "sphere":
            _, cx, cy, cz, r, _m, _f, _rk = p
            t = _isect_sphere(ro, light_dir, np.array([cx, cy, cz], dtype=np.float32), r)
        elif p[0] == "cyl":
            _, cx, cz, r, ymin, ymax, _m, _f, _rk = p
            t, _ = _isect_cylinder_y(ro, light_dir, cx, cz, r, ymin, ymax)
        else:
            _, cx, cz, yb, yt, rb, rt, _m, _f, _rk = p
            t = _isect_cone_y(ro, light_dir, cx, cz, yb, yt, rb, rt)
        in_shadow |= (t < np.inf)
    return in_shadow


def _compute_normal(prim, hit_pt):
    """World-space normal for a hit on `prim` at hit_pt (n,3)."""
    kind = prim[0]
    if kind == "sphere":
        _, cx, cy, cz, r, _m, _f, _rk = prim
        n = hit_pt - np.array([cx, cy, cz], dtype=np.float32)
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    if kind == "cyl":
        _, cx, cz, r, ymin, ymax, _m, _f, _rk = prim
        # If y very close to top cap, normal = +Y; else side normal.
        top = np.abs(hit_pt[:, 1] - ymax) < 1e-3
        n = np.zeros_like(hit_pt)
        n[:, 0] = hit_pt[:, 0] - cx
        n[:, 2] = hit_pt[:, 2] - cz
        norm_side = np.linalg.norm(n[:, [0, 2]], axis=1, keepdims=True) + 1e-9
        n[:, [0, 2]] /= norm_side
        n[top] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return n
    # cone frustum: normal in the (x-cx, z-cz) plane tilted by the slope
    _, cx, cz, yb, yt, rb, rt, _m, _f, _rk = prim
    dx = hit_pt[:, 0] - cx
    dz = hit_pt[:, 2] - cz
    radial = np.sqrt(dx * dx + dz * dz) + 1e-9
    slope = (rt - rb) / (yt - yb)      # dr/dy
    # Surface tangent-in-radial-plane = (1, slope), so normal = (1, -slope)^perp
    n = np.zeros_like(hit_pt)
    n[:, 0] = dx / radial
    n[:, 2] = dz / radial
    n[:, 1] = -slope
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)


# --- Materials + shading ----------------------------------------------------
JEWEL_RUBY     = np.array([185,  40,  50], dtype=np.float32)
JEWEL_SAPPHIRE = np.array([ 60, 130, 200], dtype=np.float32)

def _sample_piece_at(hit_pt, prim, is_light):
    """Sample piece texture (wood or marble depending on _MATERIAL).

    Cylindrical UV so grain/rings wrap naturally around each piece body.
    """
    _init_textures()
    sheet = _PIECE_LIGHT if is_light else _PIECE_DARK
    _f, _rk = prim[-2], prim[-1]
    H, W = sheet.shape[:2]
    off_y = (_rk * 17 + _f * 23 + 7) % max(H - 8, 1)
    off_x = (_f  * 29 + _rk * 13 + 5) % max(W - 8, 1)
    cx = _f + 0.5
    cz = _rk + 0.5
    ang = np.arctan2(hit_pt[:, 0] - cx, hit_pt[:, 2] - cz)
    u = ((ang / (2 * np.pi) + 0.5) * (W // 2) +
         hit_pt[:, 1] * 4.0).astype(np.int32) % max(W // 2, 1)
    v = (hit_pt[:, 1] * 18.0 +
         np.sqrt((hit_pt[:, 0] - cx) ** 2 + (hit_pt[:, 2] - cz) ** 2) * 9.0
         ).astype(np.int32) % max(H // 2, 1)
    ys = np.clip(off_y + v, 0, H - 1)
    xs = np.clip(off_x + u, 0, W - 1)
    return sheet[ys, xs].astype(np.float32)


def _sample_board_at(hit_pt):
    """Sample board texture (wood or marble) with checkerboard light/dark."""
    _init_textures()
    fx = np.clip(np.floor(hit_pt[:, 0]).astype(np.int32), 0, 7)
    fz = np.clip(np.floor(hit_pt[:, 2]).astype(np.int32), 0, 7)
    is_light = ((fz + fx) % 2) == 1
    L = _BOARD_LIGHT; D = _BOARD_DARK
    H, W = L.shape[:2]
    u = (hit_pt[:, 0] * (W / 8.0)).astype(np.int32) % W
    v = (hit_pt[:, 2] * (H / 8.0)).astype(np.int32) % H
    out = np.empty((hit_pt.shape[0], 3), dtype=np.float32)
    out[is_light]  = L[v[is_light],  u[is_light]].astype(np.float32)
    out[~is_light] = D[v[~is_light], u[~is_light]].astype(np.float32)
    return out


def _shade(hit_pt, normal, view_dir, base_color, in_shadow,
           light_dir, specular_pow=64.0, specular_strength=0.35,
           ambient=0.30):
    """Phong-ish shading. Everything is (n,3) except the scalars."""
    ndotl = np.clip((normal * light_dir).sum(axis=1), 0.0, 1.0)
    ndotl = np.where(in_shadow, 0.0, ndotl)
    diffuse = ndotl[:, None] * base_color
    # Specular via half-vector
    half = light_dir + view_dir
    half = half / (np.linalg.norm(half, axis=1, keepdims=True) + 1e-9)
    ndoth = np.clip((normal * half).sum(axis=1), 0.0, 1.0)
    spec = (ndoth ** specular_pow)[:, None] * 255.0 * specular_strength
    spec = np.where(in_shadow[:, None], 0.0, spec)
    ambient_col = ambient * base_color
    out = ambient_col + diffuse + spec
    return np.clip(out, 0, 255)


# --- Main board render (2D 16-bit by default; 3D still available via CHESS_3D=1)
_RENDER_CACHE = {}   # fen -> rendered numpy array (H, W, 3)
_RENDER_CACHE_MAX = 32

def _render_board_pixels(board: chess.Board) -> np.ndarray:
    """Render the current position. Default is flat 2D TurboGrafx-16 style.
    Set CHESS_3D=1 to use the slow ray-marched 3D path instead.
    Results are cached by FEN so redraws of the same position are free."""
    key = (board.board_fen() + (" w" if board.turn else " b")
           + "|" + _MATERIAL + "|" + ("3d" if os.environ.get("CHESS_3D") else "2d")
           + f"|sq{SQ}")
    cached = _RENDER_CACHE.get(key)
    if cached is not None:
        return cached
    if os.environ.get("CHESS_3D"):
        out = _render_board_pixels_impl(board)
    else:
        out = _render_board_2d(board)
    if len(_RENDER_CACHE) >= _RENDER_CACHE_MAX:
        _RENDER_CACHE.pop(next(iter(_RENDER_CACHE)))
    _RENDER_CACHE[key] = out
    return out


def _render_board_pixels_impl(board: chess.Board) -> np.ndarray:
    """Render the current position as a 3D scene. Returns (H, W, 3) uint8."""
    _init_textures()
    H = W = 8 * SQ            # terminal footprint scales with SQ

    # Camera: elevated behind the near edge, looking at the board center.
    cam = np.array([4.0, 6.8, -3.5], dtype=np.float32)
    target = np.array([4.0, 0.0, 4.0], dtype=np.float32)
    up_world = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fwd = target - cam
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_world); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    fov = 0.55   # radians half-angle-ish

    # Build ray directions in world space (H*W, 3)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    ndc_x = (xs / (W - 1)) * 2 - 1
    ndc_y = 1 - (ys / (H - 1)) * 2
    aspect = W / H
    dir_cam = (right[None, None, :] * (ndc_x * fov * aspect)[..., None] +
               up[None, None, :]   * (ndc_y * fov)[..., None] +
               fwd[None, None, :])
    dir_cam /= np.linalg.norm(dir_cam, axis=2, keepdims=True)
    rd = dir_cam.reshape(-1, 3)
    ro = np.broadcast_to(cam, rd.shape).copy()

    prims = _build_scene(board)

    t, idx = _intersect_scene(ro, rd, prims)
    hit_pt = ro + rd * t[:, None]

    # Backdrop color for misses (dark room)
    img = np.zeros_like(rd)
    img[:] = np.array([15, 12, 12], dtype=np.float32)

    # ---- Board hits (idx == -2) ----
    mb = idx == -2
    if mb.any():
        base = _sample_board_at(hit_pt[mb])
        normal = np.zeros((mb.sum(), 3), dtype=np.float32)
        normal[:, 1] = 1.0
        light_dir = np.array([-0.5, 0.85, -0.2], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        light_dir_b = np.broadcast_to(light_dir, normal.shape)
        in_sh = _shadow_ray(hit_pt[mb], light_dir_b, prims)
        view_dir = -rd[mb]
        if _MATERIAL == "wood":
            img[mb] = _shade(hit_pt[mb], normal, view_dir, base, in_sh,
                             light_dir_b, specular_pow=24.0,
                             specular_strength=0.10, ambient=0.50)
        else:
            img[mb] = _shade(hit_pt[mb], normal, view_dir, base, in_sh,
                             light_dir_b, specular_pow=48.0,
                             specular_strength=0.15, ambient=0.55)

    # ---- Piece hits (idx >= 0) ----
    hit_piece = idx >= 0
    if hit_piece.any():
        light_dir = np.array([-0.5, 0.85, -0.2], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        for i, p in enumerate(prims):
            mask = idx == i
            if not mask.any():
                continue
            hp = hit_pt[mask]
            normal = _compute_normal(p, hp)
            ld_b = np.broadcast_to(light_dir, normal.shape)
            # Only shadow-ray the front-facing pixels — back-facing ones are
            # already dark from Lambertian and shadow makes no visible change.
            front = (normal * ld_b).sum(axis=1) > 0.02
            in_sh = np.zeros(hp.shape[0], dtype=bool)
            if front.any():
                in_sh[front] = _shadow_ray(hp[front], ld_b[front], prims)
            mat = p[-3]
            if mat == "J_R":
                base = np.broadcast_to(JEWEL_RUBY,     (hp.shape[0], 3)).astype(np.float32)
                spec_pow, spec_str, amb = 128.0, 0.7, 0.25
            elif mat == "J_S":
                base = np.broadcast_to(JEWEL_SAPPHIRE, (hp.shape[0], 3)).astype(np.float32)
                spec_pow, spec_str, amb = 128.0, 0.7, 0.25
            elif mat == "W":
                base = _sample_piece_at(hp, p, is_light=True)
                if _MATERIAL == "wood":
                    spec_pow, spec_str, amb = 32.0, 0.22, 0.38   # satin oak
                else:
                    spec_pow, spec_str, amb = 64.0, 0.35, 0.35   # polished marble
            else:
                base = _sample_piece_at(hp, p, is_light=False)
                if _MATERIAL == "wood":
                    spec_pow, spec_str, amb = 28.0, 0.18, 0.28   # satin walnut
                else:
                    spec_pow, spec_str, amb = 64.0, 0.30, 0.20
            view_dir = -rd[mask]
            img[mask] = _shade(hp, normal, view_dir, base, in_sh, ld_b,
                               specular_pow=spec_pow,
                               specular_strength=spec_str, ambient=amb)

    img = img.reshape(H, W, 3).clip(0, 255).astype(np.uint8)
    # Frame around it
    frame = 2
    out = np.full((H + 2 * frame, W + 2 * frame, 3), BORDER, dtype=np.uint8)
    out[frame:frame + H, frame:frame + W] = img
    return out

def _pixels_to_terminal(img: np.ndarray) -> str:
    """Convert an (H, W, 3) image into terminal text using '▀' half-blocks.

    Terminal cells are ~2:1 tall vs wide, so with plain half-blocks a square
    image renders as a stretched-wide rectangle. We downsample X by 2 (average
    each pair of pixel columns) so on-screen aspect ratio ends up ~correct,
    and the board fits in ~130 columns instead of 260.

    Half-block: fg = top pixel, bg = bottom pixel → 2 pixel rows per cell."""
    H, W, _ = img.shape
    if W % 2 == 1:  # pad to even width
        img = np.hstack([img, np.full((H, 1, 3), BORDER, dtype=np.uint8)])
        W += 1
    # Horizontal downsample: average column pairs
    img = ((img[:, 0::2].astype(np.uint16) +
            img[:, 1::2].astype(np.uint16)) // 2).astype(np.uint8)
    W = img.shape[1]
    if H % 2 == 1:  # pad odd height
        img = np.vstack([img, np.full((1, W, 3), BORDER, dtype=np.uint8)])
        H += 1
    out = []
    for y in range(0, H, 2):
        top = img[y]
        bot = img[y + 1]
        last_top = last_bot = None
        line = []
        for x in range(W):
            t = tuple(int(v) for v in top[x])
            b = tuple(int(v) for v in bot[x])
            if t != last_top:
                line.append(f"\033[38;2;{t[0]};{t[1]};{t[2]}m")
                last_top = t
            if b != last_bot:
                line.append(f"\033[48;2;{b[0]};{b[1]};{b[2]}m")
                last_bot = b
            line.append("▀")
        line.append(RESET)
        out.append("".join(line))
    return "\n".join(out)


def _render_board_pixelart(board: chess.Board, header: str = "") -> str:
    """Full pixel-art renderer with file/rank labels."""
    img = _render_board_pixels(board)

    label_fg = _fg(LABEL_FG); label_bg = _bg(LABEL_BG)
    # File labels: one letter per 16-px square = ~8 chars, centered under board.
    # Each square renders as (SQ // 2) terminal columns wide (we downsample
    # X by 2) and (SQ // 2) terminal rows tall (half-block halves rows).
    cell_cols = SQ // 2
    cell_rows = SQ // 2

    def build_file_row():
        pad = " " * 3  # room for rank label prefix
        cells = "".join(chr(ord("a") + f).center(cell_cols) for f in range(8))
        return pad + cells

    def build_rank_prefix(rank):
        return f" {rank+1} "

    # Compose lines: rank label + one board line
    board_lines = _pixels_to_terminal(img).split("\n")
    lines = []
    if header:
        lines.append(BOLD + header + RESET)
    lines.append(build_file_row())
    frame_rows = 1  # 2 px frame / 2 = 1 terminal row
    idx = 0
    while idx < frame_rows:
        lines.append("   " + board_lines[idx])
        idx += 1
    for rank in range(7, -1, -1):
        rank_lbl = build_rank_prefix(rank)
        mid = cell_rows // 2
        for r in range(cell_rows):
            prefix = rank_lbl if r == mid else "   "
            lines.append(prefix + board_lines[idx])
            idx += 1
    while idx < len(board_lines):
        lines.append("   " + board_lines[idx])
        idx += 1
    lines.append(build_file_row())
    return "\n".join(lines)


# --- Plain ASCII fallback --------------------------------------------------
UNICODE = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}
LIGHT_BG = "\033[47m"; DARK_BG = "\033[100m"
FG_BLACK = "\033[30m"; FG_WHITE = "\033[97m"

def _render_board_ascii(board: chess.Board, header: str = "") -> str:
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
    return "\n".join(lines)


def render_board(board: chess.Board, header: str = ""):
    if os.environ.get("CHESS_ASCII"):
        print(_render_board_ascii(board, header=header))
    else:
        print(_render_board_pixelart(board, header=header))


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

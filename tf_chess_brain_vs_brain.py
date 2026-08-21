#!/usr/bin/env python3
"""
TensorFlow Chess: Brain vs Brain  —  Graphical Edition
======================================================

Two players, each with their OWN brain model:
  - Player 1 Brain (White)
  - Player 2 Brain (Black)

Each brain has the SAME architecture:
  - Frozen foundation layers (read-only, shared initialization, then locked)
  - Trainable "learning head" layers on top (each player's own)

They play chess against each other on a realistic Tkinter board.
After every game, each brain trains ONLY its learning head on the
move outcomes it experienced (win = reinforce chosen moves,
loss = discourage them). The frozen foundation never changes.

Run:
    python tf_chess_brain_vs_brain_gui.py
    python tf_chess_brain_vs_brain_gui.py --games 20 --delay 0.3
"""

import os
import sys
import subprocess
import shutil
import threading
import queue

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
        from PIL import Image, ImageTk  # noqa: F401
    except ImportError:
        _pip_install("Pillow")
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
import logging
import traceback
import numpy as np
import chess
import tensorflow as tf
from tensorflow.keras import layers, Model
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Error / debug logging  (file + console)
# ---------------------------------------------------------------------------
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_FILE = os.path.join(
    _LOG_DIR,
    f"chess_brain_{time.strftime('%Y%m%d_%H%M%S')}.log",
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("chess_brain")
log.info("Logging to %s", _LOG_FILE)

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
# ---------------------------------------------------------------------------
_EVAL_CACHE = {}
_MATE_SCORE = 1.0

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
        return -_MATE_SCORE
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
    score = 0.7 * brain_val + 0.3 * np.tanh(mat * 1.5)
    _EVAL_CACHE[key] = score
    return score

def _order_moves(board: chess.Board):
    """Cheap move ordering: captures & promotions first, then checks, then rest."""
    def key(mv):
        score = 0
        if board.is_capture(mv):
            victim = board.piece_at(mv.to_square)
            attacker = board.piece_at(mv.from_square)
            v_val = _PIECE_VAL.get(victim.piece_type, 0) if victim else 1
            a_val = _PIECE_VAL.get(attacker.piece_type, 0) if attacker else 0
            score += 100 + 10 * v_val - a_val
        if mv.promotion:
            score += 90
        if board.gives_check(mv):
            score += 5
        return -score
    return sorted(board.legal_moves, key=key)

def _alphabeta(brain: Model, board: chess.Board, depth: int,
               alpha: float, beta: float, q_depth: int = 2) -> float:
    """Negamax alpha-beta. Returns score from side-to-move POV in [-1, 1]."""
    term = _terminal_score(board)
    if term is not None:
        return term

    if depth <= 0:
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
    leaf evaluator. Softmax-sample among the top scorers using `temperature`."""
    legal = list(board.legal_moves)
    if not legal:
        return None, None, None

    _EVAL_CACHE.clear()

    pre_move_encoding = encode_board(board)
    root_scores = np.empty(len(legal), dtype=np.float32)

    if search_depth <= 0:
        batch = np.zeros((len(legal), 8, 8, 13), dtype=np.float32)
        ordered = _order_moves(board)
        for i, mv in enumerate(ordered):
            board.push(mv)
            batch[i] = encode_board(board)
            board.pop()
        opp = brain.predict(batch, verbose=0).flatten()
        legal = ordered
        root_scores = -opp
    else:
        alpha, beta = -2.0, 2.0
        ordered = _order_moves(board)
        legal = ordered
        for i, mv in enumerate(ordered):
            board.push(mv)
            val = -_alphabeta(brain, board, search_depth - 1,
                              -beta, -alpha, q_depth=2)
            board.pop()
            root_scores[i] = val
            if val > alpha:
                alpha = val

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
# Reward shaping
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
    return v / 39.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_head(brain: Model, memory_list, outcome: float, name: str):
    if not memory_list:
        return None
    X = np.stack([m[0] for m in memory_list])
    materials = np.array([m[1] for m in memory_list], dtype=np.float32)
    targets = np.clip(0.25 * materials + 0.75 * outcome, -1.0, 1.0)
    hist = brain.fit(X, targets, epochs=1, batch_size=32, verbose=0)
    loss = float(hist.history["loss"][0])
    return loss


# ---------------------------------------------------------------------------
# Graphical Chess Board (Tkinter)
# ---------------------------------------------------------------------------
SQUARE_SIZE = 64
BOARD_SIZE = SQUARE_SIZE * 8
LIGHT_COLOR = "#f0d9b5"
DARK_COLOR = "#b58863"
HIGHLIGHT_COLOR = "#cdd26a"
LAST_MOVE_COLOR = "#aaa23a"
CHECK_COLOR = "#e74c3c"

PIECE_IMAGE_NAMES = {
    'r': 'black_rook', 'n': 'black_knight', 'b': 'black_bishop',
    'q': 'black_queen', 'k': 'black_king', 'p': 'black_pawn',
    'R': 'white_rook', 'N': 'white_knight', 'B': 'white_bishop',
    'Q': 'white_queen', 'K': 'white_king', 'P': 'white_pawn',
}


class ChessGUI:
    def __init__(self, root, args):
        self.root = root
        self.args = args
        self.root.title("TensorFlow Chess: Brain vs Brain")
        self.root.resizable(False, False)

        # Locate piece images (same folder as script, or cwd, or artifacts)
        self.piece_dir = self._find_piece_dir()
        self.piece_images = {}
        self._load_images()

        # State
        self.board = chess.Board()
        self.last_move = None
        self.running = False
        self.stop_requested = False
        self.msg_queue = queue.Queue()

        # Build models
        self._build_models()

        # Scoreboard / history
        self.scoreboard = {"P1_wins": 0, "P2_wins": 0, "draws": 0}
        self.history = []

        # Layout
        self._build_ui()

        # Process UI messages from worker thread
        self.root.after(100, self._process_queue)

    def _find_piece_dir(self):
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "chess_pieces"),
            os.path.join(os.getcwd(), "chess_pieces"),
            "/home/workdir/artifacts/chess_pieces",
            "chess_pieces",
        ]
        for d in candidates:
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "white_king.png")):
                return d
        return candidates[0]

    def _load_images(self):
        target = int(SQUARE_SIZE * 0.85)
        for symbol, name in PIECE_IMAGE_NAMES.items():
            path = os.path.join(self.piece_dir, f"{name}.png")
            if not os.path.isfile(path):
                # Fallback: blank transparent
                img = Image.new("RGBA", (target, target), (0, 0, 0, 0))
            else:
                img = Image.open(path).convert("RGBA")
                img = img.resize((target, target), Image.LANCZOS)
            self.piece_images[symbol] = ImageTk.PhotoImage(img)

    def _build_models(self):
        if self.args.load and os.path.isdir(self.args.load):
            print(f"Loading pretrained brains from {self.args.load}...")
            foundation = tf.keras.models.load_model(
                os.path.join(self.args.load, "foundation.keras"), compile=False)
            for layer in foundation.layers:
                layer.trainable = False
            brain_p1 = tf.keras.models.load_model(
                os.path.join(self.args.load, "brain_player1.keras"), compile=False)
            brain_p2 = tf.keras.models.load_model(
                os.path.join(self.args.load, "brain_player2.keras"), compile=False)
            brain_p1.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
            brain_p2.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        else:
            print("Building shared frozen foundation...")
            foundation = build_foundation(seed=42)
            print("Building Player 1 Brain (White)...")
            brain_p1 = build_brain("player1", foundation, head_seed=101)
            print("Building Player 2 Brain (Black)...")
            brain_p2 = build_brain("player2", foundation, head_seed=202)

        self.foundation = foundation
        self.brain_p1 = brain_p1
        self.brain_p2 = brain_p2

        frozen = sum(np.prod(w.shape) for w in foundation.weights)
        t1 = sum(np.prod(w.shape) for w in brain_p1.trainable_weights)
        t2 = sum(np.prod(w.shape) for w in brain_p2.trainable_weights)
        print(f"  foundation params (FROZEN): {frozen:,}")
        print(f"  P1 trainable head params: {t1:,}")
        print(f"  P2 trainable head params: {t2:,}")

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")

        # Left: board
        board_frame = ttk.Frame(main)
        board_frame.grid(row=0, column=0, rowspan=2, padx=(0, 12))

        self.canvas = tk.Canvas(
            board_frame,
            width=BOARD_SIZE,
            height=BOARD_SIZE,
            highlightthickness=1,
            highlightbackground="#333",
        )
        self.canvas.pack()

        # Coordinates labels (optional aesthetic)
        coord_frame = ttk.Frame(board_frame)
        coord_frame.pack(fill="x")
        for i, letter in enumerate("abcdefgh"):
            ttk.Label(coord_frame, text=letter, width=4, anchor="center").grid(row=0, column=i)

        # Right panel
        side = ttk.Frame(main)
        side.grid(row=0, column=1, sticky="nw")

        ttk.Label(side, text="Brain vs Brain", font=("Helvetica", 14, "bold")).pack(anchor="w")
        ttk.Label(side, text="Player 1 = White   |   Player 2 = Black",
                  font=("Helvetica", 9)).pack(anchor="w", pady=(0, 8))

        # Status
        self.status_var = tk.StringVar(value="Ready. Press Start.")
        status_lbl = ttk.Label(side, textvariable=self.status_var, wraplength=280,
                               font=("Helvetica", 10))
        status_lbl.pack(anchor="w", pady=(0, 6))

        self.eval_var = tk.StringVar(value="Eval: —")
        ttk.Label(side, textvariable=self.eval_var, font=("Courier", 10)).pack(anchor="w")

        self.move_var = tk.StringVar(value="Last move: —")
        ttk.Label(side, textvariable=self.move_var, font=("Courier", 10)).pack(anchor="w", pady=(0, 8))

        # Scoreboard
        score_frame = ttk.LabelFrame(side, text="Scoreboard", padding=6)
        score_frame.pack(fill="x", pady=(0, 8))
        self.score_var = tk.StringVar(value="P1: 0   P2: 0   Draws: 0")
        ttk.Label(score_frame, textvariable=self.score_var,
                  font=("Helvetica", 11, "bold")).pack()

        # History
        hist_frame = ttk.LabelFrame(side, text="Game History", padding=4)
        hist_frame.pack(fill="both", expand=True, pady=(0, 8))
        self.hist_text = tk.Text(hist_frame, width=36, height=12, font=("Courier", 9),
                                 state="disabled", wrap="word")
        scroll = ttk.Scrollbar(hist_frame, command=self.hist_text.yview)
        self.hist_text.configure(yscrollcommand=scroll.set)
        self.hist_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Controls
        ctrl = ttk.Frame(side)
        ctrl.pack(fill="x", pady=(4, 0))
        self.start_btn = ttk.Button(ctrl, text="Start", command=self.start_games)
        self.start_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self.request_stop, state="disabled")
        self.stop_btn.pack(side="left")

        # Settings summary
        settings = (
            f"Games: {self.args.games}  |  Depth: {self.args.search_depth}\n"
            f"Temp: {self.args.temperature}  |  Delay: {self.args.delay}s"
        )
        ttk.Label(side, text=settings, font=("Helvetica", 8), foreground="#555").pack(
            anchor="w", pady=(10, 0)
        )
        ttk.Label(
            side,
            text=f"Log: {_LOG_FILE}",
            font=("Helvetica", 7),
            foreground="#888",
            wraplength=280,
        ).pack(anchor="w", pady=(4, 0))

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        # Squares
        for row in range(8):
            for col in range(8):
                x0 = col * SQUARE_SIZE
                y0 = row * SQUARE_SIZE
                x1 = x0 + SQUARE_SIZE
                y1 = y0 + SQUARE_SIZE
                color = LIGHT_COLOR if (row + col) % 2 == 0 else DARK_COLOR
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # Highlight last move
        if self.last_move is not None:
            for sq in (self.last_move.from_square, self.last_move.to_square):
                col = chess.square_file(sq)
                row = 7 - chess.square_rank(sq)
                x0 = col * SQUARE_SIZE
                y0 = row * SQUARE_SIZE
                self.canvas.create_rectangle(
                    x0, y0, x0 + SQUARE_SIZE, y0 + SQUARE_SIZE,
                    fill=LAST_MOVE_COLOR, outline="", stipple="gray50"
                )

        # Check highlight
        if self.board.is_check():
            king_sq = self.board.king(self.board.turn)
            if king_sq is not None:
                col = chess.square_file(king_sq)
                row = 7 - chess.square_rank(king_sq)
                x0 = col * SQUARE_SIZE
                y0 = row * SQUARE_SIZE
                self.canvas.create_rectangle(
                    x0, y0, x0 + SQUARE_SIZE, y0 + SQUARE_SIZE,
                    fill=CHECK_COLOR, outline="", stipple="gray25"
                )

        # Pieces
        for sq, piece in self.board.piece_map().items():
            col = chess.square_file(sq)
            row = 7 - chess.square_rank(sq)
            img = self.piece_images.get(piece.symbol())
            if img:
                cx = col * SQUARE_SIZE + SQUARE_SIZE // 2
                cy = row * SQUARE_SIZE + SQUARE_SIZE // 2
                self.canvas.create_image(cx, cy, image=img)

        # Rank numbers on left edge
        for row in range(8):
            rank = 8 - row
            self.canvas.create_text(
                6, row * SQUARE_SIZE + 10,
                text=str(rank), anchor="nw",
                fill="#333" if (row % 2 == 0) else "#eee",
                font=("Helvetica", 9, "bold")
            )

    def _process_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                kind = msg[0]
                if kind == "board":
                    self.board = msg[1]
                    self.last_move = msg[2]
                    self.draw_board()
                elif kind == "status":
                    self.status_var.set(msg[1])
                elif kind == "eval":
                    self.eval_var.set(msg[1])
                elif kind == "move":
                    self.move_var.set(msg[1])
                elif kind == "score":
                    self.score_var.set(msg[1])
                elif kind == "history":
                    self.hist_text.configure(state="normal")
                    self.hist_text.insert("end", msg[1] + "\n")
                    self.hist_text.see("end")
                    self.hist_text.configure(state="disabled")
                elif kind == "done":
                    self.running = False
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.status_var.set(msg[1])
                elif kind == "error":
                    messagebox.showerror("Error", msg[1])
                    self.running = False
                    self.start_btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(80, self._process_queue)

    def start_games(self):
        if self.running:
            return
        self.running = True
        self.stop_requested = False
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.hist_text.configure(state="normal")
        self.hist_text.delete("1.0", "end")
        self.hist_text.configure(state="disabled")
        self.scoreboard = {"P1_wins": 0, "P2_wins": 0, "draws": 0}
        self.history = []
        t = threading.Thread(target=self._run_loop, daemon=True)
        t.start()

    def request_stop(self):
        self.stop_requested = True
        self.status_var.set("Stopping after current game...")

    def _run_loop(self):
        try:
            log.info(
                "Starting %d games  depth=%d  temp=%.2f  delay=%.2f",
                self.args.games, self.args.search_depth,
                self.args.temperature, self.args.delay,
            )
            for g in range(1, self.args.games + 1):
                if self.stop_requested:
                    log.info("Stop requested — exiting after game %d", g - 1)
                    break
                self.msg_queue.put(("status", f"Game {g}/{self.args.games} — starting..."))
                log.info("=== Game %d / %d ===", g, self.args.games)
                try:
                    memory, outcome, result = self._play_one_game(g)
                except Exception as game_err:
                    log.exception("Game %d crashed", g)
                    self.msg_queue.put((
                        "history",
                        f"Game {g}: ERROR — {game_err}"
                    ))
                    self.msg_queue.put((
                        "status",
                        f"Game {g} error (see log): {game_err}"
                    ))
                    continue

                # Update scoreboard
                if result == "1-0":
                    self.scoreboard["P1_wins"] += 1
                    line = f"Game {g}: Player 1 WIN, Player 2 loss  ({result})"
                elif result == "0-1":
                    self.scoreboard["P2_wins"] += 1
                    line = f"Game {g}: Player 1 loss, Player 2 WIN  ({result})"
                else:
                    self.scoreboard["draws"] += 1
                    tag = "unfinished (ply cap)" if result == "*" else "drawn"
                    line = f"Game {g}: draw ({tag})  ({result})"
                self.history.append(line)
                log.info(line)
                self.msg_queue.put(("history", line))
                self.msg_queue.put((
                    "score",
                    f"P1: {self.scoreboard['P1_wins']}   "
                    f"P2: {self.scoreboard['P2_wins']}   "
                    f"Draws: {self.scoreboard['draws']}"
                ))

                # Train heads
                self.msg_queue.put(("status", f"Game {g} finished — training heads..."))
                try:
                    found_before = [w.numpy().copy() for w in self.foundation.weights]
                    loss1 = train_head(self.brain_p1, memory[chess.WHITE],
                                       outcome[chess.WHITE], "P1")
                    loss2 = train_head(self.brain_p2, memory[chess.BLACK],
                                       outcome[chess.BLACK], "P2")
                    found_after = [w.numpy() for w in self.foundation.weights]
                    drift = max(
                        np.max(np.abs(a - b))
                        for a, b in zip(found_before, found_after)
                    )
                    loss1_s = f"{loss1:.4f}" if loss1 is not None else "—"
                    loss2_s = f"{loss2:.4f}" if loss2 is not None else "—"
                    train_msg = (
                        f"Trained P1 loss={loss1_s}  "
                        f"P2 loss={loss2_s}  "
                        f"(foundation drift {drift:.1e})"
                    )
                    log.info("Game %d training: %s", g, train_msg)
                    self.msg_queue.put(("status", train_msg))
                except Exception as train_err:
                    log.exception("Training failed after game %d", g)
                    self.msg_queue.put((
                        "status",
                        f"Training error (see log): {train_err}"
                    ))
                time.sleep(0.4)

            final = (
                f"Finished.  P1 wins: {self.scoreboard['P1_wins']}  |  "
                f"P2 wins: {self.scoreboard['P2_wins']}  |  "
                f"Draws: {self.scoreboard['draws']}"
            )
            log.info(final)
            self.msg_queue.put(("done", final))
        except Exception as e:
            tb = traceback.format_exc()
            log.error("Fatal error in game loop:\n%s", tb)
            self.msg_queue.put(("error", f"{e}\n\n{tb}\n\nFull log: {_LOG_FILE}"))

    def _play_one_game(self, game_num: int):
        board = chess.Board()
        memory = {chess.WHITE: [], chess.BLACK: []}
        self.last_move = None
        self.msg_queue.put(("board", board.copy(), None))

        ply = 0
        max_plies = self.args.max_plies
        delay = self.args.delay

        while not board.is_game_over(claim_draw=True) and ply < max_plies:
            if self.stop_requested:
                break

            mover = board.turn
            brain = self.brain_p1 if mover == chess.WHITE else self.brain_p2
            who = "Player 1 (White)" if mover == chess.WHITE else "Player 2 (Black)"

            self.msg_queue.put(("status", f"Game {game_num}  |  Ply {ply+1}  |  {who} thinking..."))

            t0 = time.time()
            try:
                move, pre_enc, score = choose_move(
                    brain, board,
                    temperature=self.args.temperature,
                    search_depth=self.args.search_depth,
                )
            except Exception:
                log.exception(
                    "choose_move failed  game=%d ply=%d side=%s fen=%s",
                    game_num, ply + 1, who, board.fen(),
                )
                raise
            move_ms = (time.time() - t0) * 1000

            if move is None:
                log.warning("No legal move at game=%d ply=%d fen=%s", game_num, ply + 1, board.fen())
                break

            board.push(move)
            material_after = material_balance(board, mover)
            memory[mover].append((pre_enc, material_after))
            self.last_move = move

            log.debug(
                "Game %d ply %d  %s  %s  eval=%+.3f  %.0fms",
                game_num, ply + 1, who, move.uci(), score, move_ms,
            )
            self.msg_queue.put(("board", board.copy(), move))
            self.msg_queue.put(("eval", f"Eval: {score:+.3f}   ({move_ms:.0f} ms)"))
            self.msg_queue.put(("move", f"Last move: {move.uci()}  ({who})"))
            self.msg_queue.put((
                "status",
                f"Game {game_num}  |  Ply {ply+1}  |  {who} played {move.uci()}"
            ))

            ply += 1
            if delay > 0:
                time.sleep(delay)

        # Outcome
        result = board.result(claim_draw=True)
        if result == "1-0":
            outcome = {chess.WHITE: 1.0, chess.BLACK: -1.0}
        elif result == "0-1":
            outcome = {chess.WHITE: -1.0, chess.BLACK: 1.0}
        else:
            outcome = {chess.WHITE: 0.0, chess.BLACK: 0.0}

        return memory, outcome, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TensorFlow Chess Brain vs Brain (GUI)")
    parser.add_argument("--games", type=int, default=5, help="number of games")
    parser.add_argument("--delay", type=float, default=0.25,
                        help="seconds between moves (0 = no delay)")
    parser.add_argument("--max-plies", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="exploration temperature; 0 = greedy")
    parser.add_argument("--search-depth", type=int, default=2,
                        help="alpha-beta search depth in plies")
    parser.add_argument("--load", type=str, default=None,
                        help="Directory with pretrained .keras models")
    args = parser.parse_args()

    root = tk.Tk()
    # Try a slightly nicer theme if available
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    app = ChessGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()

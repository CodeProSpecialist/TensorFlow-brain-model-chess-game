"""
Pretrain foundation + parallel head training
============================================

Pipeline:
  1. Generate a dataset of positions labeled by Stockfish evaluations.
  2. Pretrain the SHARED foundation on that dataset (supervised regression
     to Stockfish's centipawn evaluation, squashed to [-1, 1]).
  3. FREEZE the foundation (read-only from here on).
  4. Build two brains — Player 1 and Player 2 — each with its own
     trainable learning head on top of the frozen foundation.
  5. Train each head 20,000 steps in PARALLEL (one Python thread per brain).
  6. Save both brains + the frozen foundation to disk. They can then be
     loaded by tf_chess_brain_vs_brain.py to play head-to-head.

Run:
    python pretrain_parallel.py
    python pretrain_parallel.py --dataset-size 8000 --pretrain-epochs 6 --steps 20000
"""

import os
import sys
import platform
import subprocess
import shutil
import importlib
import stat
import urllib.request
import zipfile
import tarfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ---------------------------------------------------------------------------
# BOOTSTRAP: make sure everything we need is installed and ready before we
# import the heavy libraries (TensorFlow, chess). Runs on every launch; skips
# any step that's already satisfied.
# ---------------------------------------------------------------------------
def _pick_bootstrap_dir():
    """Prefer a dir next to the script; fall back to ~/.cache if that
    filesystem is noexec (some containers/CI mount user dirs that way)."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chess_deps"),
        os.path.join(os.path.expanduser("~"), ".cache", "chess_deps"),
        os.path.join("/tmp", "chess_deps"),
    ]
    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            # Test executability: write a tiny sh script, chmod +x, exec it.
            test = os.path.join(d, ".exec_test.sh")
            with open(test, "w") as f:
                f.write("#!/bin/sh\nexit 0\n")
            os.chmod(test, 0o755)
            try:
                subprocess.check_call([test], stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL)
                os.remove(test)
                return d
            except (PermissionError, subprocess.CalledProcessError, OSError):
                try: os.remove(test)
                except OSError: pass
                continue
        except OSError:
            continue
    # Last resort: return the script-adjacent dir anyway.
    return candidates[0]


BOOTSTRAP_DIR = _pick_bootstrap_dir()
os.makedirs(BOOTSTRAP_DIR, exist_ok=True)


def _pip_install(*pkgs):
    print(f"[bootstrap] pip install {' '.join(pkgs)}")
    cmd = [sys.executable, "-m", "pip", "install", "--quiet",
           "--disable-pip-version-check", *pkgs]
    # --break-system-packages is required on Debian/Ubuntu PEP 668 setups;
    # older pip versions don't recognise it, so retry without it on failure.
    try:
        subprocess.check_call(cmd + ["--break-system-packages"])
    except subprocess.CalledProcessError:
        subprocess.check_call(cmd)


def _detect_gpu() -> bool:
    """True if an NVIDIA GPU appears usable (driver present)."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"],
                                      stderr=subprocess.DEVNULL, timeout=5)
        return b"GPU" in out
    except Exception:
        return False


def _ensure_tensorflow():
    try:
        import tensorflow  # noqa: F401
        return
    except ImportError:
        pass
    if _detect_gpu():
        print("[bootstrap] NVIDIA GPU detected -> installing 'tensorflow'")
        _pip_install("tensorflow")
    else:
        print("[bootstrap] No GPU detected -> installing 'tensorflow-cpu'")
        _pip_install("tensorflow-cpu")


def _ensure_python_chess():
    try:
        import chess  # noqa: F401
        import chess.engine  # noqa: F401
        return
    except ImportError:
        _pip_install("chess")


def _ensure_numpy():
    try:
        import numpy  # noqa: F401
    except ImportError:
        _pip_install("numpy")


# ---- Stockfish acquisition ------------------------------------------------
# The engine is only used to label the pretraining dataset. If a system
# stockfish is present anywhere sane, use it. Otherwise download a static
# binary to BOOTSTRAP_DIR and remember its path.
STOCKFISH_PATH = None


def _find_system_stockfish():
    for cand in ["stockfish", "/usr/games/stockfish",
                 "/usr/local/bin/stockfish", "/opt/homebrew/bin/stockfish"]:
        p = shutil.which(cand) if os.path.basename(cand) == cand else \
            (cand if os.path.isfile(cand) and os.access(cand, os.X_OK) else None)
        if p:
            return p
    return None


def _download(url: str, dest: str):
    print(f"[bootstrap] downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def _download_stockfish() -> str:
    """Download a static Stockfish binary for the current OS/arch."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Prebuilt binaries hosted by the Stockfish project on GitHub releases.
    # sf_17.1 is the current stable at the time this script was written.
    base = ("https://github.com/official-stockfish/Stockfish/releases/"
            "download/sf_17.1")
    if system == "linux" and machine in ("x86_64", "amd64"):
        asset = "stockfish-ubuntu-x86-64-avx2.tar"
        exe_rel = os.path.join("stockfish", "stockfish-ubuntu-x86-64-avx2")
        kind = "tar"
    elif system == "linux" and machine in ("aarch64", "arm64"):
        asset = "stockfish-android-armv8.tar"
        exe_rel = os.path.join("stockfish", "stockfish-android-armv8")
        kind = "tar"
    elif system == "darwin":
        asset = "stockfish-macos-m1-apple-silicon.tar" if machine == "arm64" \
                else "stockfish-macos-x86-64-avx2.tar"
        exe_rel = os.path.join("stockfish", asset.replace(".tar", ""))
        kind = "tar"
    elif system == "windows":
        asset = "stockfish-windows-x86-64-avx2.zip"
        exe_rel = os.path.join("stockfish", "stockfish-windows-x86-64-avx2.exe")
        kind = "zip"
    else:
        raise RuntimeError(
            f"No prebuilt Stockfish for {system}/{machine}. "
            "Please install stockfish manually and put it on PATH.")

    url = f"{base}/{asset}"
    archive_path = os.path.join(BOOTSTRAP_DIR, asset)
    _download(url, archive_path)

    print(f"[bootstrap] extracting {asset}")
    if kind == "zip":
        with zipfile.ZipFile(archive_path) as z:
            z.extractall(BOOTSTRAP_DIR)
    else:
        with tarfile.open(archive_path) as t:
            t.extractall(BOOTSTRAP_DIR)

    exe_path = os.path.join(BOOTSTRAP_DIR, exe_rel)
    if not os.path.isfile(exe_path):
        # Some archives lay out slightly differently; find any executable.
        for root, _, files in os.walk(os.path.join(BOOTSTRAP_DIR, "stockfish")):
            for name in files:
                if name.startswith("stockfish") and not name.endswith(
                        (".txt", ".md", ".nnue")):
                    exe_path = os.path.join(root, name)
                    break
    if system != "windows":
        st = os.stat(exe_path)
        os.chmod(exe_path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return exe_path


def _ensure_stockfish() -> str:
    p = _find_system_stockfish()
    if p:
        print(f"[bootstrap] using system stockfish at {p}")
        return p
    # Cache path within our bootstrap dir
    marker = os.path.join(BOOTSTRAP_DIR, "stockfish_path.txt")
    if os.path.isfile(marker):
        cached = open(marker).read().strip()
        if os.path.isfile(cached) and os.access(cached, os.X_OK):
            print(f"[bootstrap] using cached stockfish at {cached}")
            return cached
    p = _download_stockfish()
    # Sanity check: does it respond to UCI?
    try:
        proc = subprocess.run([p], input="uci\nquit\n", capture_output=True,
                              text=True, timeout=10)
        if "id name Stockfish" not in proc.stdout:
            raise RuntimeError(f"Stockfish at {p} did not respond to 'uci'")
    except Exception as e:
        raise RuntimeError(f"Downloaded stockfish failed to run: {e}")
    with open(marker, "w") as f:
        f.write(p)
    print(f"[bootstrap] stockfish ready at {p}")
    return p


def _bootstrap():
    _ensure_numpy()
    _ensure_python_chess()
    _ensure_tensorflow()
    global STOCKFISH_PATH
    STOCKFISH_PATH = _ensure_stockfish()


_bootstrap()

# ---------------------------------------------------------------------------
# Now the heavy imports (safe: everything above ensured they're installed).
# ---------------------------------------------------------------------------
import argparse
import math
import time
import threading
import numpy as np
import chess
import chess.engine
import tensorflow as tf
from tensorflow.keras import layers, Model

tf.get_logger().setLevel("ERROR")

# STOCKFISH_PATH is set during _bootstrap() above.

# ---------------------------------------------------------------------------
# Board encoding (same as the play script)
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
    planes = np.zeros((8, 8, 13), dtype=np.float32)
    for sq, piece in board.piece_map().items():
        row = 7 - (sq // 8)
        col = sq % 8
        planes[row, col, PIECE_TO_PLANE[(piece.piece_type, piece.color)]] = 1.0
    planes[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    return planes


# ---------------------------------------------------------------------------
# Dataset generation: random-ish games; every N plies, evaluate with Stockfish
# ---------------------------------------------------------------------------
def cp_to_score(cp: int) -> float:
    """Squash centipawn eval to [-1, 1] (from side-to-move POV)."""
    return math.tanh(cp / 400.0)

def generate_dataset(n_positions: int, engine_depth: int = 8,
                     sample_every: int = 3, seed: int = 0):
    """Play random games; every `sample_every` plies, evaluate the position
    with Stockfish and record (encoded_board, score-from-side-to-move-POV)."""
    print(f"Generating {n_positions} Stockfish-labeled positions "
          f"(depth={engine_depth})...")
    rng = np.random.default_rng(seed)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    X = np.zeros((n_positions, 8, 8, 13), dtype=np.float32)
    y = np.zeros((n_positions,),         dtype=np.float32)

    filled = 0
    t0 = time.time()
    try:
        while filled < n_positions:
            board = chess.Board()
            ply = 0
            while not board.is_game_over(claim_draw=True) and ply < 120 \
                    and filled < n_positions:
                if ply % sample_every == 0:
                    try:
                        info = engine.analyse(
                            board, chess.engine.Limit(depth=engine_depth))
                        score_obj = info["score"].pov(board.turn)
                        cp = score_obj.score(mate_score=10000)
                        if cp is not None:
                            X[filled] = encode_board(board)
                            y[filled] = cp_to_score(cp)
                            filled += 1
                            if filled % 500 == 0 or filled == n_positions:
                                dt = time.time() - t0
                                rate = filled / max(dt, 1e-6)
                                print(f"  {filled:>6}/{n_positions}  "
                                      f"({rate:.1f} pos/s)")
                    except chess.engine.EngineError:
                        pass

                legal = list(board.legal_moves)
                if not legal:
                    break
                # Mostly random moves with an occasional engine move,
                # so the dataset covers both messy and sane positions.
                if rng.random() < 0.15:
                    try:
                        best = engine.play(board,
                                           chess.engine.Limit(depth=6)).move
                        move = best if best in legal else rng.choice(legal)
                    except chess.engine.EngineError:
                        move = rng.choice(legal)
                else:
                    move = legal[int(rng.integers(len(legal)))]
                board.push(move)
                ply += 1
    finally:
        engine.quit()

    print(f"  done in {time.time()-t0:.1f}s")
    return X, y


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_foundation(seed: int = 42) -> Model:
    """Foundation architecture (trainable during pretraining; frozen after)."""
    tf.random.set_seed(seed)
    inp = layers.Input(shape=(8, 8, 13), name="board")
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      name="found_conv1")(inp)
    x = layers.Conv2D(32, 3, padding="same", activation="relu",
                      name="found_conv2")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu",
                      name="found_conv3")(x)
    x = layers.Flatten(name="found_flat")(x)
    x = layers.Dense(128, activation="relu", name="found_dense")(x)
    return Model(inp, x, name="foundation")


def build_pretrain_model(foundation: Model) -> Model:
    """Foundation + a throwaway regression head, only used for pretraining."""
    inp = layers.Input(shape=(8, 8, 13))
    feat = foundation(inp)
    out = layers.Dense(1, activation="tanh", name="pretrain_score")(feat)
    m = Model(inp, out, name="pretrain")
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def build_brain(name: str, foundation: Model, head_seed: int) -> Model:
    """Frozen foundation + this player's own trainable head."""
    tf.random.set_seed(head_seed)
    inp = layers.Input(shape=(8, 8, 13), name=f"{name}_in")
    feat = foundation(inp, training=False)     # frozen path
    h = layers.Dense(64, activation="relu", name=f"{name}_head1")(feat)
    h = layers.Dense(32, activation="relu", name=f"{name}_head2")(h)
    out = layers.Dense(1, activation="tanh", name=f"{name}_score")(h)
    m = Model(inp, out, name=name)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


# ---------------------------------------------------------------------------
# Parallel head training
# ---------------------------------------------------------------------------
class HeadTrainer(threading.Thread):
    """Trains one brain's head for N mini-batch steps in a background thread.

    TF ops release the GIL, so two of these running concurrently give real
    parallel work on CPU — and would share the GPU if one were present.
    """
    def __init__(self, name: str, brain: Model, X: np.ndarray, y: np.ndarray,
                 steps: int, batch_size: int = 64, report_every: int = 1000):
        super().__init__(name=name, daemon=True)
        self.brain = brain
        self.X = X
        self.y = y
        self.steps = steps
        self.batch_size = batch_size
        self.report_every = report_every
        self.losses = []
        self.wall_time = 0.0

    def run(self):
        n = len(self.X)
        rng = np.random.default_rng(hash(self.name) & 0xFFFFFFFF)
        t0 = time.time()
        loss_accum = 0.0
        loss_count = 0
        for step in range(1, self.steps + 1):
            idx = rng.integers(0, n, size=self.batch_size)
            # Add a tiny per-brain perturbation to the target so the two heads
            # actually diverge instead of converging to identical predictors.
            noise = rng.normal(0.0, 0.05, size=self.batch_size).astype(np.float32)
            targets = np.clip(self.y[idx] + noise, -1.0, 1.0)
            loss = self.brain.train_on_batch(self.X[idx], targets)
            loss_val = float(loss[0] if isinstance(loss, (list, tuple)) else loss)
            loss_accum += loss_val
            loss_count += 1
            if step % self.report_every == 0:
                avg = loss_accum / loss_count
                self.losses.append((step, avg))
                print(f"  [{self.name}] step {step:>6}/{self.steps}  "
                      f"avg_loss={avg:.4f}  "
                      f"elapsed={time.time()-t0:.1f}s")
                loss_accum = 0.0
                loss_count = 0
        self.wall_time = time.time() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-size", type=int, default=4000,
                        help="Number of Stockfish-labeled positions")
    parser.add_argument("--engine-depth", type=int, default=8)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--pretrain-batch", type=int, default=128)
    parser.add_argument("--steps", type=int, default=20000,
                        help="Head training steps per brain (parallel)")
    parser.add_argument("--head-batch", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="./chess_models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------
    dataset_path = os.path.join(args.out_dir, "stockfish_dataset.npz")
    if os.path.exists(dataset_path):
        print(f"Loading cached dataset from {dataset_path}")
        data = np.load(dataset_path)
        X, y = data["X"], data["y"]
        if len(X) < args.dataset_size:
            print(f"  cache has {len(X)}, regenerating for {args.dataset_size}")
            X, y = generate_dataset(args.dataset_size, args.engine_depth)
            np.savez(dataset_path, X=X, y=y)
    else:
        X, y = generate_dataset(args.dataset_size, args.engine_depth)
        np.savez(dataset_path, X=X, y=y)
        print(f"Saved dataset to {dataset_path}")

    # Train/val split
    n = len(X)
    perm = np.random.default_rng(0).permutation(n)
    X, y = X[perm], y[perm]
    n_val = max(200, n // 10)
    X_val, y_val = X[:n_val], y[:n_val]
    X_tr,  y_tr  = X[n_val:], y[n_val:]
    print(f"Dataset: train={len(X_tr)}, val={len(X_val)}")

    # -----------------------------------------------------------------------
    # 2. Pretrain the foundation (trainable during this step only)
    # -----------------------------------------------------------------------
    print("\n=== Pretraining the shared foundation on Stockfish evaluations ===")
    foundation = build_foundation(seed=42)
    pretrain_model = build_pretrain_model(foundation)
    print(f"  pretrain params: "
          f"{sum(np.prod(w.shape) for w in pretrain_model.trainable_weights):,}")
    pretrain_model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch,
        verbose=2,
    )

    # -----------------------------------------------------------------------
    # 3. FREEZE the foundation
    # -----------------------------------------------------------------------
    for layer in foundation.layers:
        layer.trainable = False
    print("\nFoundation frozen (read-only from now on).")
    frozen_snapshot = [w.numpy().copy() for w in foundation.weights]

    # -----------------------------------------------------------------------
    # 4. Build the two brains
    # -----------------------------------------------------------------------
    print("\n=== Building Player 1 and Player 2 brains ===")
    brain_p1 = build_brain("player1", foundation, head_seed=101)
    brain_p2 = build_brain("player2", foundation, head_seed=202)
    def trainable_count(m):
        return sum(np.prod(w.shape) for w in m.trainable_weights)
    print(f"  P1 trainable head params: {trainable_count(brain_p1):,}")
    print(f"  P2 trainable head params: {trainable_count(brain_p2):,}")

    # -----------------------------------------------------------------------
    # 5. Parallel head training (one thread per brain)
    # -----------------------------------------------------------------------
    print(f"\n=== Training both heads for {args.steps} steps in parallel ===")
    p1_trainer = HeadTrainer("P1", brain_p1, X_tr, y_tr,
                             steps=args.steps, batch_size=args.head_batch)
    p2_trainer = HeadTrainer("P2", brain_p2, X_tr, y_tr,
                             steps=args.steps, batch_size=args.head_batch)

    t_start = time.time()
    p1_trainer.start()
    p2_trainer.start()
    p1_trainer.join()
    p2_trainer.join()
    wall = time.time() - t_start

    print(f"\nParallel training complete.")
    print(f"  P1 wall time: {p1_trainer.wall_time:.1f}s")
    print(f"  P2 wall time: {p2_trainer.wall_time:.1f}s")
    print(f"  Combined wall time: {wall:.1f}s  "
          f"(vs serial {p1_trainer.wall_time + p2_trainer.wall_time:.1f}s)")

    # Verify the frozen foundation did NOT move during head training
    drift = max(np.max(np.abs(a - b.numpy()))
                for a, b in zip(frozen_snapshot, foundation.weights))
    print(f"  foundation weight drift after parallel training: {drift:.2e}  "
          f"(should be 0)")

    # -----------------------------------------------------------------------
    # 6. Evaluate + save
    # -----------------------------------------------------------------------
    p1_val_loss, p1_val_mae = brain_p1.evaluate(X_val, y_val, verbose=0)
    p2_val_loss, p2_val_mae = brain_p2.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation vs Stockfish labels:")
    print(f"  P1 brain: mse={p1_val_loss:.4f}  mae={p1_val_mae:.4f}")
    print(f"  P2 brain: mse={p2_val_loss:.4f}  mae={p2_val_mae:.4f}")

    foundation.save(os.path.join(args.out_dir, "foundation.keras"))
    brain_p1.save(os.path.join(args.out_dir, "brain_player1.keras"))
    brain_p2.save(os.path.join(args.out_dir, "brain_player2.keras"))
    print(f"\nSaved to {args.out_dir}/:")
    print("  foundation.keras, brain_player1.keras, brain_player2.keras")


if __name__ == "__main__":
    main()

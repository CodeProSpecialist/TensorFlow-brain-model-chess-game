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
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import math
import time
import threading
import subprocess
import numpy as np
import chess
import chess.engine
import tensorflow as tf
from tensorflow.keras import layers, Model

tf.get_logger().setLevel("ERROR")

STOCKFISH_PATH = "/usr/games/stockfish"

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

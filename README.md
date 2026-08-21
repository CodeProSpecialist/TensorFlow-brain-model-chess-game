# TensorFlow-brain-model-chess-game
TensorFlow brain model chess game. Brain 1 playing Brain 2. 

```
# Step 1: pretrain (dataset gen dominates wall time; tune to taste)

python3 pretrain_parallel.py --dataset-size 8000 --engine-depth 8 \
    --pretrain-epochs 6 --steps 20000

# Step 2: watch them play

python3 tf_chess_brain_vs_brain.py --load ./chess_models --games 10 --delay 0.15
```

# Wood pieces + wood board (default), sized for ~800×800 terminal
python tf_chess_brain_vs_brain.py

# Marble pieces + marble board
CHESS_MATERIAL=marble python tf_chess_brain_vs_brain.py

# Optional size tweak (default SQ=16)
CHESS_SQ=18 python tf_chess_brain_vs_brain.py

**Notes on the parallelism.** Two brains sharing one frozen foundation graph is subtle — Keras `train_on_batch` on each brain runs forward through the shared frozen ops and backward only into that brain's own head weights, so there's no gradient collision between threads. Each thread has its own optimizer state. The tiny per-brain target noise (`0.05` std) ensures P1 and P2 heads don't converge to identical predictors despite training on the same data.

**Tuning knobs.** `--dataset-size` and `--engine-depth` dominate wall time (Stockfish generation is the slow part; my box hit ~420 pos/s at depth 5). This default has Head training at 20k steps × batch 64 = ~1.3M positions seen per brain, well past the point where the heads have saturated on this dataset — you'll see the loss plateau in the per-1000-step reports.

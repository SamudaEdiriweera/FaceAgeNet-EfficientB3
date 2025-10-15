""" Paths, constants, parameters """
"""Centralized configuration: paths, hyperparams, seeds."""

from pathlib import Path

# ------ Paths (relative to repo root) ------
ROOT = Path(__file__).resolve().parents[1]
DATA_PARTS = [ROOT/"data"/"part1", ROOT/"data"/"part2", ROOT/"data"/"part3"]
CHECKPOINT_PATH = ROOT/"checkpoints"/"ckpt.weights.h5"
EXPORT_DIR = ROOT/"models"

# ------ Data / Training Hyperparameters ------
IMG_SIZE = (300, 300)  # width, height for EfficientNetB3 native resolution
BATCH_SIZE = 16  # GPU memory friendly size (16 for B3, 8 for B4, 4 for B5) 
SEED = 42  # reproducibility seed for tf, np, py, etc.
NUM_BINS = 20  # for stratified splits on continuous age

# --- Training schedule ---
EPOCHS_STAGE1 = 3 # head-only training
EPOCHS_STAGE2 = 3 # fine-tuning last ~50 layers
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5

# --- Misc ---
VALID_EXTS = {".jpg", ".jpeg", ".png"}
AGE_MAX = 120  # for sanity checks
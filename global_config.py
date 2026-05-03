from model_design import GPTConfig
import os
import numpy as np # =========================
# imports
# =========================
from model_design import GPTConfig
import os
import numpy as np

# =========================
# core dataset / experiment constants
# =========================
BASE_WIDTH = 64

TRAIN_SPLIT = .98
TEST_SPLIT = .01
VALIDATION_SPLIT = .01

RANDOM_SEED = 42

SVG_THRESHOLD = 5000
TOKEN_THRESHOLD = 2048

VOCAB_SIZE = 8000
BLOCK_SIZE = 512
BATCH_SIZE = 32

# =========================
# training schedule settings
# =========================
LR_SWEEP_STEPS = 1000
WARMUP_STEPS = 500
EPOCHS = 20
BEST_LR = 0.001

LEARNING_RATES = np.logspace(-3.5, -5, 6)
VAL_LOSS_ITERATIONS = 100

# =========================
# system / runtime settings
# =========================
SAVE_DIR = "/training_checkpoints"

TRAIN_TOKEN_PATH = 'data/train_set.bin'
TEST_TOKEN_PATH = 'data/test_set.bin'
VAL_TOKEN_PATH = 'data/val_set.bin'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORES = os.cpu_count() // 2

# =========================
# standard model configs (no μP)
# =========================
STANDARD_CONFIG = {
    "tiny": GPTConfig("tiny", 128, 4, 4, 512, VOCAB_SIZE, BLOCK_SIZE, False, 0),
    "small": GPTConfig("small", 192, 6, 6, 768, VOCAB_SIZE, BLOCK_SIZE, False, 0),
    "medium": GPTConfig("medium", 384, 6, 6, 1536, VOCAB_SIZE, BLOCK_SIZE, False, 0),
    "large": GPTConfig("large", 512, 10, 8, 2048, VOCAB_SIZE, BLOCK_SIZE, False, 0),
    "xl": GPTConfig("xl", 640, 12, 10, 2560, VOCAB_SIZE, BLOCK_SIZE, False, 0),
}

# =========================
# μP model configs
# =========================
MUP_CONFIG = {
    "m_tiny": GPTConfig("tiny", 128, 4, 4, 512, VOCAB_SIZE, BLOCK_SIZE, True, 0),
    "m_small": GPTConfig("small", 192, 6, 6, 768, VOCAB_SIZE, BLOCK_SIZE, True, 0),
    "m_medium": GPTConfig("medium", 384, 6, 6, 1536, VOCAB_SIZE, BLOCK_SIZE, True, 0),
    "m_large": GPTConfig("large", 512, 10, 8, 2048, VOCAB_SIZE, BLOCK_SIZE, True, 0),
    "m_xl": GPTConfig("xl", 640, 12, 10, 2560, VOCAB_SIZE, BLOCK_SIZE, True, 0),
}

# =========================
# special μp scaling anchors
# =========================
# base shape used for mup scaling reference
BASE_CONFIG = GPTConfig("base", 128, 4, 4, 512, VOCAB_SIZE, BLOCK_SIZE, True, 0)

# delta config for width scaling comparisons
DELTA_CONFIG = GPTConfig("delta", 256, 4, 4, 1024, VOCAB_SIZE, BLOCK_SIZE, True, 0)

# =========================
# best single-run config
# =========================
BESTEST_CONFIG = {
    'bestest': GPTConfig("large", 512, 10, 8, 2048, VOCAB_SIZE, BLOCK_SIZE, False, 0)
}
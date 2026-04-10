"""
config.py — Central configuration for MIGN-XAI project.

Every hyperparameter, path, and constant lives here.
Change values here; all other scripts import from this file.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data" / "raw"
PROC_DIR   = ROOT_DIR / "data" / "processed"
CKPT_DIR   = ROOT_DIR / "checkpoints"
RESULT_DIR = ROOT_DIR / "results"
LOG_DIR    = ROOT_DIR / "logs"

# Create directories if they don't exist
for d in [DATA_DIR, PROC_DIR, CKPT_DIR, RESULT_DIR, LOG_DIR,
          RESULT_DIR / "xai", RESULT_DIR / "plots"]:
    os.makedirs(d, exist_ok=True)

# Valid split strategies
SPLIT_STRATEGIES = ["random", "cell_blind", "drug_blind"]

# ─────────────────────────────────────────────────────────────────────────────
# DATA URLS  (all free, no login required)
# ─────────────────────────────────────────────────────────────────────────────

GDSC1_URL = (
    "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/"
    "GDSC1_fitted_dose_response_27Oct23.xlsx"
)
GDSC2_URL = (
    "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/"
    "GDSC2_fitted_dose_response_27Oct23.xlsx"
)
GDSC_COMPOUNDS_URL = (
    "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/"
    "screened_compounds_rel_8.5.csv"
)

# DepMap portal (CCLE)
DEPMAP_EXPRESSION_URL = (
    "https://figshare.com/ndownloader/files/34989919"  # OmicsExpressionProteinCodingGenesTPMLogp1
)
DEPMAP_MUTATIONS_URL = (
    "https://figshare.com/ndownloader/files/34989940"  # OmicsSomaticMutations
)
DEPMAP_MODEL_URL = (
    "https://figshare.com/ndownloader/files/34989919"  # Model.csv
)

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

RMSE_THRESHOLD        = 0.3       # Remove drug-cell pairs with poor curve fit
MIN_CELL_LINES        = 5         # Min cell lines a drug must screen
N_LANDMARK_GENES      = 978       # LINCS L1000 landmark gene count
N_PATHWAY_FEATURES    = 50        # Number of MSigDB Hallmark pathways
MIN_PATHWAY_GENES     = 5         # Min genes per pathway to include it

# Mutation: restrict to cancer-relevant genes (COSMIC Cancer Gene Census)
USE_COSMIC_GENES      = True
MAX_MUT_GENES         = 735       # Max mutation features

# ─────────────────────────────────────────────────────────────────────────────
# MOLECULAR GRAPH
# ─────────────────────────────────────────────────────────────────────────────

ATOM_SYMBOLS          = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'other']
NODE_FEATURE_DIM      = 45        # Atom feature vector dimension
EDGE_FEATURE_DIM      = 10        # Bond feature vector dimension

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────

GIN_HIDDEN_DIM        = 256       # Hidden dim inside GIN layers
GIN_OUT_DIM           = 256       # Output dim from drug encoder
GIN_N_LAYERS          = 3         # Number of GINEConv layers
GIN_DROPOUT           = 0.1       # Dropout inside GIN

CELL_HIDDEN_DIMS      = [1024, 512, 256, 128]   # MLP layer widths
CELL_DROPOUT          = 0.3       # Dropout in cell encoder

ATTN_HEADS            = 4         # Number of cross-attention heads
ATTN_DIM              = 128       # Cross-attention key/value dimension

# Prediction head: input = GIN_OUT_DIM + ATTN_DIM + CELL_HIDDEN_DIMS[-1]
HEAD_HIDDEN           = 256
HEAD_DROPOUT          = 0.3

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

SEED                  = 42
BATCH_SIZE            = 256
EVAL_BATCH_SIZE       = 512
N_EPOCHS              = 200
LEARNING_RATE         = 3e-4
WEIGHT_DECAY          = 1e-4
L1_LAMBDA             = 1e-4      # L1 reg on cell encoder layer 1
HUBER_DELTA           = 1.0       # Huber loss transition point
CLIP_GRAD_NORM        = 1.0       # Gradient clipping
PATIENCE              = 20        # Early stopping patience
SCHEDULER_T_MAX       = 100       # Cosine annealing period
SCHEDULER_ETA_MIN     = 1e-6

# Label Distribution Smoothing
LDS_BINS              = 100
LDS_KERNEL_WIDTH      = 5

# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

SPLIT_STRATEGIES      = ["random", "cell_blind", "drug_blind"]
TEST_CELL_FRACTION    = 0.10      # Fraction of cells held out for cell_blind
TEST_DRUG_FRACTION    = 0.10      # Fraction of drugs held out for drug_blind

# ─────────────────────────────────────────────────────────────────────────────
# XAI
# ─────────────────────────────────────────────────────────────────────────────

SHAP_N_BACKGROUND     = 100       # Background samples for KernelSHAP
SHAP_N_EXPLAIN        = 200       # Number of samples to explain per drug
SHAP_NSAMPLES         = 100       # KernelSHAP integration samples
XAI_VALIDATION_DRUGS  = [         # Drugs with known mechanisms for validation
    "Erlotinib", "Vemurafenib", "Olaparib", "Palbociclib",
    "Imatinib",  "Gefitinib",   "Crizotinib", "Trametinib",
]
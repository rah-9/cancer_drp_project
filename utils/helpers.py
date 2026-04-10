# ═══════════════════════════════════════════════════════════════════════════
#  utils/helpers.py  —  Seed, logging, checkpointing
# ═══════════════════════════════════════════════════════════════════════════

import os
import random
import logging
import numpy as np
import torch
from pathlib import Path


def set_seed(seed: int = 42):
    """
    Set all random seeds for full reproducibility.

    Locks:
      - Python random
      - NumPy random
      - PyTorch CPU + all GPUs
      - cuDNN deterministic mode (disables auto-tuner)
      - Python hash seed
      - PyTorch deterministic algorithms (warn_only for scatter ops)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PyTorch 1.11+: force deterministic algorithms where possible
    # warn_only=True prevents crash when no deterministic impl exists
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        # Older PyTorch versions don't support warn_only
        pass


def get_device() -> torch.device:
    """Auto-select GPU if available, else CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  Using CPU (training will be slow on large datasets)")
    return device


def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    """Set up a logger that writes to both console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def save_checkpoint(model, optimiser, epoch: int, metrics: dict,
                    filepath: Path, is_best: bool = False):
    """Save model checkpoint."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch"    : epoch,
        "metrics"  : metrics,
        "model"    : model.state_dict(),
        "optimiser": optimiser.state_dict(),
    }
    torch.save(checkpoint, filepath)
    if is_best:
        best_path = filepath.parent / "best_model.pt"
        torch.save(checkpoint, best_path)


def load_checkpoint(model, filepath: Path, optimiser=None, device=None):
    """Load model checkpoint. Returns (model, optimiser, epoch, metrics)."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimiser is not None:
        optimiser.load_state_dict(checkpoint["optimiser"])
    return model, optimiser, checkpoint["epoch"], checkpoint.get("metrics", {})


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
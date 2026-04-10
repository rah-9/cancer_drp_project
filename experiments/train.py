"""
experiments/train.py
─────────────────────────────────────────────────────────────────────────────
Main training script for MIGN-XAI.

Usage:
    # Train with cell-blind split (recommended, most rigorous):
    python experiments/train.py --split cell_blind

    # Train with random split (fastest, least rigorous):
    python experiments/train.py --split random --epochs 100

    # Full hyperparameter search with Optuna:
    python experiments/train.py --split cell_blind --tune --n_trials 50

Training loop details:
  - Loss      : Huber Loss (robust to IC50 outliers)
  - Optimiser : AdamW with cosine annealing LR schedule
  - Regularisation: L1 on cell encoder layer 1 (gene sparsity)
  - LDS       : Label Distribution Smoothing for IC50 imbalance
  - Grad clip : max_norm=1.0 prevents gradient explosions
  - Early stop: patience=20 epochs on validation Pearson r
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as GeoLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PROC_DIR, CKPT_DIR, LOG_DIR, RESULT_DIR,
                    BATCH_SIZE, EVAL_BATCH_SIZE, N_EPOCHS, LEARNING_RATE,
                    WEIGHT_DECAY, L1_LAMBDA, HUBER_DELTA, CLIP_GRAD_NORM,
                    PATIENCE, SCHEDULER_T_MAX, SCHEDULER_ETA_MIN, SEED,
                    LDS_BINS, LDS_KERNEL_WIDTH)
from data.dataset import load_all_data
from models.mign_xai import build_model
from utils.metrics_and_helpers import (compute_metrics, print_metrics,
                                       get_lds_weights, set_seed, get_device,
                                       setup_logger, save_checkpoint)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE PASS — collect all predictions and targets
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Run model in eval mode over a DataLoader.
    Returns (y_true, y_pred) as numpy arrays.
    """
    model.eval()
    y_true_all, y_pred_all = [], []

    for batch in loader:
        batch  = batch.to(device)
        y_hat, _ = model(batch)
        y_true_all.extend(batch.y.cpu().numpy().tolist())
        y_pred_all.extend(y_hat.cpu().numpy().tolist())

    return np.array(y_true_all), np.array(y_pred_all)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimiser, device,
                lds_weight_map: dict = None, l1_lambda: float = L1_LAMBDA):
    """
    Run one full training epoch.

    Parameters
    ----------
    model         : MIGN_XAI model
    loader        : Training DataLoader
    criterion     : Loss function (HuberLoss)
    optimiser     : AdamW optimiser
    device        : Computation device
    lds_weight_map: Dict {ic50_value: weight} for LDS (optional)
    l1_lambda     : L1 regularisation coefficient for cell encoder layer 1

    Returns
    -------
    avg_loss : Mean training loss over the epoch
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = batch.to(device)
        optimiser.zero_grad()

        # Forward pass
        y_hat, _ = model(batch)

        # Base loss: Huber on each sample
        y_true = batch.y.squeeze()
        loss   = criterion(y_hat, y_true)

        # L1 regularisation on first cell encoder layer
        # Encourages sparse gene selection → better SHAP interpretability
        if l1_lambda > 0 and hasattr(model, "cell_encoder") and hasattr(model.cell_encoder, "get_layer1_weights"):
            l1_reg = sum(
                p.abs().sum()
                for p in model.cell_encoder.get_layer1_weights()
            )
            loss = loss + l1_lambda * l1_reg

        # Backward pass
        loss.backward()

        # Gradient clipping: prevents exploding gradients in early training
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

        optimiser.step()
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(split_strategy: str = "cell_blind",
          n_epochs:       int  = N_EPOCHS,
          lr:             float = LEARNING_RATE,
          use_lds:        bool  = True,
          seed:           int   = SEED,
          tag:            str   = ""):
    """
    Full training pipeline.

    Parameters
    ----------
    split_strategy : "random" | "cell_blind" | "drug_blind"
    n_epochs       : Maximum number of training epochs
    lr             : Initial learning rate
    use_lds        : Whether to apply Label Distribution Smoothing
    seed           : Random seed
    tag            : Optional experiment tag for saving files

    Returns
    -------
    best_metrics : Dict of best validation metrics
    """
    set_seed(seed)
    device = get_device()

    exp_name = f"mign_xai_{split_strategy}{('_'+tag) if tag else ''}"
    logger   = setup_logger(exp_name, LOG_DIR / f"{exp_name}.log")
    logger.info(f"Starting experiment: {exp_name}")
    logger.info(f"Split strategy: {split_strategy} | Epochs: {n_epochs} | LR: {lr}")

    # ── Load data ──────────────────────────────────────────────────────────
    logger.info("Loading data...")
    splitter = load_all_data("gdsc1")
    train_ds, val_ds, test_ds = splitter.get_split(split_strategy, seed=seed)
    cell_in = train_ds[0].cell.shape[-1]

    # ── Label Distribution Smoothing ───────────────────────────────────────
    if use_lds:
        train_labels = train_ds.pairs["ln_ic50"].values.astype(np.float32)
        lds_weights = get_lds_weights(train_labels, LDS_BINS, LDS_KERNEL_WIDTH)
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(lds_weights),
            num_samples=len(train_ds),
            replacement=True,
        )
        train_loader = GeoLoader(train_ds, batch_size=BATCH_SIZE,
                                 sampler=sampler, num_workers=0)
        logger.info(f"LDS enabled. Weight range: "
                    f"[{lds_weights.min():.2f}, {lds_weights.max():.2f}]")
    else:
        train_loader = GeoLoader(train_ds, batch_size=BATCH_SIZE,
                                 shuffle=True, num_workers=0)

    val_loader  = GeoLoader(val_ds,  batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = GeoLoader(test_ds, batch_size=EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(cell_in=cell_in, device=device)

    # ── Resource monitoring ────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} trainable")
    logger.info(f"Train batches/epoch: {len(train_loader)}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated() / 1e6
        logger.info(f"GPU memory after model load: {alloc:.0f} MB")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                    f"Total: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    import time
    train_start_time = time.time()

    # ── Loss, optimiser, scheduler ─────────────────────────────────────────
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser,
                                  T_max=SCHEDULER_T_MAX,
                                  eta_min=SCHEDULER_ETA_MIN)

    # ── Training loop ──────────────────────────────────────────────────────
    best_val_r      = -1.0
    best_val_metrics = {}
    patience_ctr    = 0
    history         = []

    logger.info("\n" + "─"*70)
    logger.info(f"{'Epoch':>6} | {'TrainLoss':>10} | {'Val Pearson r':>13} | "
                f"{'Val RMSE':>9} | {'Val AUROC':>9} | {'LR':>8}")
    logger.info("─"*70)

    for epoch in range(1, n_epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────
        avg_loss = train_epoch(model, train_loader, criterion,
                               optimiser, device)

        # ── Validate ───────────────────────────────────────────────────────
        y_true_val, y_pred_val = run_inference(model, val_loader, device)
        val_metrics = compute_metrics(y_true_val, y_pred_val)

        current_lr = optimiser.param_groups[0]["lr"]
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "lr": current_lr,
        })

        # ── Logging ────────────────────────────────────────────────────────
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"{epoch:>6} | {avg_loss:>10.4f} | "
                f"{val_metrics['pearson_r']:>13.4f} | "
                f"{val_metrics['rmse']:>9.4f} | "
                f"{val_metrics['auroc']:>9.4f} | "
                f"{current_lr:>8.2e}"
            )

        # ── Early stopping ─────────────────────────────────────────────────
        val_r = val_metrics["pearson_r"]
        if val_r > best_val_r:
            best_val_r       = val_r
            best_val_metrics = val_metrics.copy()
            patience_ctr     = 0
            save_checkpoint(
                model, optimiser, epoch, val_metrics,
                filepath = CKPT_DIR / f"{exp_name}_epoch{epoch}.pt",
                is_best  = True,
            )
            logger.info(f"  ★ New best! Val Pearson r = {val_r:.4f} (saved checkpoint)")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                logger.info(f"\n  Early stopping at epoch {epoch}. "
                            f"Best val Pearson r = {best_val_r:.4f}")
                break

    # ── Training time + resource summary ───────────────────────────────────
    train_time = time.time() - train_start_time
    logger.info(f"\nTraining completed in {train_time/60:.1f} minutes "
                f"({train_time:.0f}s)")
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1e6
        logger.info(f"Peak GPU memory: {peak_mem:.0f} MB")

    # ── Final evaluation on test set ───────────────────────────────────────
    logger.info("\n" + "="*70)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*70)

    # Load best checkpoint
    best_ckpt = CKPT_DIR / "best_model.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    y_true_test, y_pred_test = run_inference(model, test_loader, device)
    test_metrics = compute_metrics(y_true_test, y_pred_test)

    logger.info("Test metrics:")
    print_metrics(test_metrics, prefix="TEST")

    # ── Save results ───────────────────────────────────────────────────────
    results = {
        "experiment"      : exp_name,
        "split_strategy"  : split_strategy,
        "best_val_metrics": best_val_metrics,
        "test_metrics"    : test_metrics,
        "config": {
            "n_epochs"  : n_epochs,
            "lr"        : lr,
            "use_lds"   : use_lds,
            "seed"      : seed,
        }
    }

    results_file = RESULT_DIR / f"{exp_name}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    # Save history CSV
    pd.DataFrame(history).to_csv(
        RESULT_DIR / f"{exp_name}_history.csv", index=False)

    # Save predictions for analysis
    pd.DataFrame({
        "y_true": y_true_test,
        "y_pred": y_pred_test,
    }).to_csv(RESULT_DIR / f"{exp_name}_predictions.csv", index=False)

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIGN-XAI model")
    parser.add_argument("--split",   type=str, default="cell_blind",
                        choices=["random", "cell_blind", "drug_blind"],
                        help="Data split strategy")
    parser.add_argument("--epochs",  type=int, default=N_EPOCHS,
                        help="Max training epochs")
    parser.add_argument("--lr",      type=float, default=LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--no_lds",  action="store_true",
                        help="Disable Label Distribution Smoothing")
    parser.add_argument("--seed",    type=int, default=SEED)
    parser.add_argument("--tag",     type=str, default="",
                        help="Experiment tag for file naming")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  MIGN-XAI: Training")
    print("="*70)
    print(f"  Split strategy: {args.split}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  LDS:            {not args.no_lds}")
    print(f"  Seed:           {args.seed}")
    print("="*70 + "\n")

    metrics = train(
        split_strategy = args.split,
        n_epochs       = args.epochs,
        lr             = args.lr,
        use_lds        = not args.no_lds,
        seed           = args.seed,
        tag            = args.tag,
    )
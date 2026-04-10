"""
experiments/evaluate.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive evaluation of a trained MIGN-XAI model across all three
split strategies and per-cancer-type breakdowns.

Usage:
    # Evaluate best saved model on all splits:
    python experiments/evaluate.py

    # Evaluate on a specific split only:
    python experiments/evaluate.py --split cell_blind

    # Evaluate a specific checkpoint:
    python experiments/evaluate.py --checkpoint checkpoints/model_epoch50.pt
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from torch_geometric.loader import DataLoader as GeoLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PROC_DIR, CKPT_DIR, RESULT_DIR, EVAL_BATCH_SIZE, SEED,
                    SPLIT_STRATEGIES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM)
from data.dataset import load_all_data
from models.mign_xai import MIGN_XAI
from utils.metrics_and_helpers import (compute_metrics, print_metrics,
                                       set_seed, get_device)


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """
    Run model in eval mode over a DataLoader.
    Returns (y_true, y_pred, drug_ids, cosmic_ids) as arrays.
    """
    model.eval()
    y_true_all, y_pred_all = [], []
    drug_ids_all, cosmic_ids_all = [], []

    for batch in loader:
        batch = batch.to(device)
        y_hat, _ = model(batch)
        y_true_all.extend(batch.y.cpu().numpy().tolist())
        y_pred_all.extend(y_hat.cpu().numpy().tolist())

        # Collect IDs for per-drug / per-cancer analysis
        if hasattr(batch, "drug_id"):
            drug_ids_all.extend(
                batch.drug_id if isinstance(batch.drug_id, list)
                else [batch.drug_id] * batch.y.shape[0])
        if hasattr(batch, "cosmic_id"):
            cosmic_ids_all.extend(
                batch.cosmic_id if isinstance(batch.cosmic_id, list)
                else [batch.cosmic_id] * batch.y.shape[0])

    return (np.array(y_true_all), np.array(y_pred_all),
            drug_ids_all, cosmic_ids_all)


# ─────────────────────────────────────────────────────────────────────────────
# PER-DRUG EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def per_drug_metrics(y_true, y_pred, drug_ids, min_pairs=10):
    """
    Compute metrics for each individual drug.

    Returns DataFrame with per-drug Pearson r, RMSE, and sample count.
    Only drugs with >= min_pairs test samples are included.
    """
    df = pd.DataFrame({
        "y_true": y_true, "y_pred": y_pred, "drug_id": drug_ids
    })
    rows = []
    for drug_id, grp in df.groupby("drug_id"):
        if len(grp) < min_pairs:
            continue
        m = compute_metrics(grp["y_true"].values, grp["y_pred"].values)
        m["drug_id"] = drug_id
        m["n_pairs"] = len(grp)
        rows.append(m)
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# SCATTER PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(y_true, y_pred, metrics_dict, split_name, save_path=None):
    """Predicted vs True ln(IC50) scatter plot with Pearson annotation."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.15, s=8, c="#1B6B5A", edgecolor="none")

    # Identity line
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "k--", linewidth=1, alpha=0.5, label="y=x")

    # Annotation
    text = (f"Pearson r  = {metrics_dict['pearson_r']:.4f}\n"
            f"Spearman ρ = {metrics_dict['spearman_rho']:.4f}\n"
            f"RMSE       = {metrics_dict['rmse']:.4f}\n"
            f"AUROC      = {metrics_dict['auroc']:.4f}")
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="#E8F5E9", alpha=0.8))

    ax.set_xlabel("True ln(IC50)", fontsize=12)
    ax.set_ylabel("Predicted ln(IC50)", fontsize=12)
    ax.set_title(f"MIGN-XAI — {split_name} Split", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# FAILURE CASE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def failure_case_analysis(y_true, y_pred, drug_ids, cosmic_ids,
                          split_name, n_worst=10):
    """
    Analyse the worst predictions to identify failure patterns.

    This is CRITICAL for understanding model limitations:
    - Are failures concentrated on specific drugs?
    - Are they on extremely sensitive or resistant cell lines?
    - Is error correlated with IC50 magnitude?

    Parameters
    ----------
    y_true, y_pred : arrays of true and predicted ln(IC50)
    drug_ids       : list of drug identifiers
    cosmic_ids     : list of cell line COSMIC IDs
    split_name     : name of the split (for logging)
    n_worst        : number of worst predictions to report

    Returns
    -------
    failures_df : DataFrame with worst predictions and error analysis
    """
    errors = np.abs(y_true - y_pred)

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": errors,
        "drug_id": drug_ids[:len(y_true)] if drug_ids else ["unknown"] * len(y_true),
        "cosmic_id": cosmic_ids[:len(y_true)] if cosmic_ids else ["unknown"] * len(y_true),
    })

    # Sort by worst error
    df = df.sort_values("abs_error", ascending=False).reset_index(drop=True)

    print(f"\n  FAILURE CASE ANALYSIS ({split_name})")
    print("  " + "-" * 55)

    # Top N worst predictions
    print(f"  Top {n_worst} worst predictions:")
    print(f"  {'#':<4} {'Drug':<12} {'Cell':<12} {'True':>8} {'Pred':>8} {'Error':>8}")
    for i, row in df.head(n_worst).iterrows():
        print(f"  {i+1:<4} {str(row['drug_id']):<12} {str(row['cosmic_id']):<12} "
              f"{row['y_true']:>8.3f} {row['y_pred']:>8.3f} {row['abs_error']:>8.3f}")

    # Error vs IC50 range analysis
    print(f"\n  Error by IC50 range:")
    bins = pd.cut(df["y_true"], bins=5, labels=[
        "Very Sensitive", "Sensitive", "Moderate", "Resistant", "Very Resistant"])
    for label in bins.cat.categories:
        subset = df[bins == label]
        if len(subset) > 0:
            print(f"    {label:<16}: n={len(subset):>5}, "
                  f"MAE={subset['abs_error'].mean():.4f}, "
                  f"max_err={subset['abs_error'].max():.4f}")

    # Drug concentration of failures (are failures in specific drugs?)
    top_fail_drugs = df.head(50).groupby("drug_id").size().sort_values(ascending=False)
    if len(top_fail_drugs) > 0:
        print(f"\n  Drugs most concentrated in top-50 failures:")
        for drug, count in top_fail_drugs.head(5).items():
            print(f"    Drug {drug}: {count} failures in top 50 "
                  f"({100*count/50:.0f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY ESTIMATION (MC DROPOUT)
# ─────────────────────────────────────────────────────────────────────────────

def uncertainty_evaluation(model, test_loader, device, n_forward=20):
    """
    Run MC Dropout uncertainty estimation over the test set.

    Analyses:
    - Mean prediction uncertainty (std across forward passes)
    - Error-uncertainty correlation (higher uncertainty ↔ higher error?)
    - Calibration: % of true values within predicted ±2σ interval

    A good model should have:
    - Positive error-uncertainty correlation (knows when it's unsure)
    - High coverage in the ±2σ interval (~95% for well-calibrated)

    Parameters
    ----------
    model        : MIGN_XAI model
    test_loader  : DataLoader for test set
    device       : torch device
    n_forward    : Number of MC forward passes

    Returns
    -------
    dict with uncertainty statistics
    """
    from scipy.stats import spearmanr

    all_means, all_stds, all_trues = [], [], []

    for batch in test_loader:
        batch = batch.to(device)
        mean, std, _ = model.mc_dropout_predict(batch, n_forward=n_forward)
        all_means.extend(mean.tolist())
        all_stds.extend(std.tolist())
        all_trues.extend(batch.y.cpu().numpy().tolist())

    means = np.array(all_means)
    stds  = np.array(all_stds)
    trues = np.array(all_trues)
    errors = np.abs(trues - means)

    # Error-uncertainty correlation
    rho, p_val = spearmanr(stds, errors)

    # Calibration: what % of truths fall within ±2σ of predicted mean
    in_2sigma = np.mean(np.abs(trues - means) <= 2 * stds) * 100

    # Calibration: ±1σ
    in_1sigma = np.mean(np.abs(trues - means) <= 1 * stds) * 100

    print(f"\n  MC DROPOUT UNCERTAINTY ({n_forward} forward passes)")
    print("  " + "-" * 55)
    print(f"    Mean uncertainty (σ): {stds.mean():.4f}")
    print(f"    Median uncertainty:   {np.median(stds):.4f}")
    print(f"    Error-uncertainty ρ:  {rho:.4f} (p={p_val:.2e})")
    if rho > 0.3:
        print(f"    -> GOOD: Model knows when it's uncertain")
    elif rho > 0.1:
        print(f"    -> MODERATE: Weak uncertainty awareness")
    else:
        print(f"    -> POOR: Model uncertainty is uncalibrated")

    print(f"    Coverage ±1σ: {in_1sigma:.1f}% (ideal: ~68%)")
    print(f"    Coverage ±2σ: {in_2sigma:.1f}% (ideal: ~95%)")

    return {
        "mean_uncertainty": round(float(stds.mean()), 4),
        "median_uncertainty": round(float(np.median(stds)), 4),
        "error_uncertainty_spearman": round(float(rho), 4),
        "error_uncertainty_pvalue": float(p_val),
        "coverage_1sigma_pct": round(float(in_1sigma), 1),
        "coverage_2sigma_pct": round(float(in_2sigma), 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(splits=None, checkpoint_path=None):
    """
    Run inference on all requested split strategies and save results.

    Parameters
    ----------
    splits          : List of split strategies (default: all three)
    checkpoint_path : Path to checkpoint (default: best_model.pt)
    """
    set_seed(SEED)
    device = get_device()

    if splits is None:
        splits = SPLIT_STRATEGIES

    # ── Load data ──────────────────────────────────────────────────────────
    splitter = load_all_data("gdsc1")
    cell_in  = pd.read_csv(PROC_DIR / "cell_features.csv", index_col=0).shape[1]

    # ── Load model ─────────────────────────────────────────────────────────
    model = MIGN_XAI(
        node_in=NODE_FEATURE_DIM,
        edge_in=EDGE_FEATURE_DIM,
        cell_in=cell_in,
    ).to(device)

    ckpt_path = Path(checkpoint_path) if checkpoint_path else CKPT_DIR / "best_model.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"  Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")
    else:
        print(f"  [WARN] No checkpoint at {ckpt_path}. Evaluating untrained model.")
    model.eval()

    # ── Evaluate each split ────────────────────────────────────────────────
    all_results = {}
    plots_dir = RESULT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("  MIGN-XAI: Comprehensive Evaluation")
    print("=" * 70)

    for split in splits:
        print(f"\n{'─' * 70}")
        print(f"  Split: {split}")
        print(f"{'─' * 70}")

        _, _, test_ds = splitter.get_split(split, seed=SEED)
        test_loader = GeoLoader(test_ds, batch_size=EVAL_BATCH_SIZE,
                                shuffle=False, num_workers=0)

        y_true, y_pred, drug_ids, cosmic_ids = run_inference(
            model, test_loader, device)
        metrics = compute_metrics(y_true, y_pred)

        print_metrics(metrics, prefix=f"  {split.upper()} TEST")
        print(f"  Test samples: {len(y_true):,}")

        # Per-drug breakdown
        drug_df = per_drug_metrics(y_true, y_pred, drug_ids)
        if not drug_df.empty:
            print(f"\n  Per-drug statistics ({len(drug_df)} drugs with ≥10 pairs):")
            print(f"    Pearson r  — median={drug_df['pearson_r'].median():.4f}, "
                  f"mean={drug_df['pearson_r'].mean():.4f}")

        # Scatter plot
        plot_scatter(y_true, y_pred, metrics, split,
                     save_path=plots_dir / f"scatter_{split}.png")
        print(f"  Saved scatter plot: {plots_dir}/scatter_{split}.png")

        # Save predictions
        pred_df = pd.DataFrame({
            "y_true": y_true, "y_pred": y_pred,
            "drug_id": drug_ids, "cosmic_id": cosmic_ids
        })
        pred_df.to_csv(RESULT_DIR / f"eval_{split}_predictions.csv", index=False)

        # Save per-drug metrics
        if not drug_df.empty:
            drug_df.to_csv(RESULT_DIR / f"eval_{split}_per_drug.csv", index=False)

        all_results[split] = {
            "metrics": metrics,
            "n_test_pairs": len(y_true),
            "per_drug_median_r": round(float(drug_df["pearson_r"].median()), 4)
            if not drug_df.empty else None,
        }

        # ── Failure case analysis ──────────────────────────────────────────
        failure_report = failure_case_analysis(
            y_true, y_pred, drug_ids, cosmic_ids, split)
        if failure_report is not None:
            failure_report.to_csv(
                RESULT_DIR / f"eval_{split}_failures.csv", index=False)

        # ── MC Dropout uncertainty ─────────────────────────────────────────
        if split == "cell_blind":
            # Run uncertainty on the hardest split only (expensive)
            print(f"  Running MC Dropout uncertainty estimation ...")
            uncert_report = uncertainty_evaluation(model, test_loader, device)
            if uncert_report:
                all_results[split]["uncertainty"] = uncert_report

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  {'Split':<15} {'Pearson r':>10} {'Spearman':>10} "
          f"{'RMSE':>8} {'AUROC':>8} {'N pairs':>10}")
    print("  " + "-" * 65)
    for split_name, res in all_results.items():
        m = res["metrics"]
        print(f"  {split_name:<15} {m['pearson_r']:>10.4f} "
              f"{m['spearman_rho']:>10.4f} {m['rmse']:>8.4f} "
              f"{m['auroc']:>8.4f} {res['n_test_pairs']:>10,}")

    # ── Save results JSON ──────────────────────────────────────────────────
    results_file = RESULT_DIR / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained MIGN-XAI")
    parser.add_argument("--split", type=str, default=None,
                        choices=SPLIT_STRATEGIES,
                        help="Evaluate on a single split (default: all)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint")
    args = parser.parse_args()

    splits = [args.split] if args.split else None
    evaluate(splits=splits, checkpoint_path=args.checkpoint)

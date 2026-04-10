# ═══════════════════════════════════════════════════════════════════════════
#  utils/metrics.py — All evaluation metrics for drug response prediction
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              roc_auc_score, average_precision_score,
                              f1_score)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    threshold: float = None) -> dict:
    """
    Compute all evaluation metrics for drug response prediction.

    Regression metrics (primary task — predicting ln(IC50)):
      Pearson r     : Linear correlation. Primary ranking metric.
      Spearman ρ    : Rank correlation. Robust to non-Gaussian errors.
      RMSE          : Root mean squared error in ln(µM) units.
      MAE           : Mean absolute error — clinically interpretable.

    Binary classification metrics (secondary task):
      For each cell line, "sensitive" = ln(IC50) below drug-specific
      or global median threshold.
      AUROC  : Area under ROC — insensitive to class imbalance.
      AUPR   : Area under precision-recall — better for imbalanced data.
      F1     : Harmonic mean of precision and recall.

    Parameters
    ----------
    y_true     : Ground truth ln(IC50) values
    y_pred     : Predicted ln(IC50) values
    threshold  : IC50 threshold for binary classification (default: median)

    Returns
    -------
    dict of metric names to float values
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # ── Regression ────────────────────────────────────────────────────────
    pearson_r,  p_r = pearsonr(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    # ── Binary classification ─────────────────────────────────────────────
    threshold = threshold if threshold is not None else float(np.median(y_true))
    # Sensitive = low IC50 → label = 1
    y_bin  = (y_true < threshold).astype(int)
    # Model scores: lower predicted IC50 → higher probability of sensitivity
    scores = -y_pred

    try:
        auroc = float(roc_auc_score(y_bin, scores))
        aupr  = float(average_precision_score(y_bin, scores))
    except ValueError:
        auroc = aupr = float("nan")

    y_pred_bin = (y_pred < threshold).astype(int)
    f1 = float(f1_score(y_bin, y_pred_bin, zero_division=0))

    return {
        "pearson_r"   : round(float(pearson_r),  4),
        "spearman_rho": round(float(spearman_rho), 4),
        "rmse"        : round(rmse, 4),
        "mae"         : round(mae,  4),
        "auroc"       : round(auroc, 4),
        "aupr"        : round(aupr,  4),
        "f1"          : round(f1,    4),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Pretty-print a metrics dict."""
    header = f"  {prefix} " if prefix else "  "
    print(f"{header}Pearson r={metrics['pearson_r']:.4f} | "
          f"Spearman rho={metrics['spearman_rho']:.4f} | "
          f"RMSE={metrics['rmse']:.4f} | MAE={metrics['mae']:.4f} | "
          f"AUROC={metrics['auroc']:.4f} | AUPR={metrics['aupr']:.4f} | "
          f"F1={metrics['f1']:.4f}")


def statistical_significance_test(y_true, y_pred_a, y_pred_b,
                                  model_a_name="Model A",
                                  model_b_name="Model B"):
    """
    Wilcoxon signed-rank test for pairwise comparison of two models.

    Tests whether Model A's per-sample squared errors are significantly
    different from Model B's. This is the recommended non-parametric test
    for paired model comparison (more robust than paired t-test for
    non-Gaussian error distributions like IC50).

    H0: median(error_A^2 - error_B^2) = 0  (no difference)
    H1: the two error distributions differ

    Parameters
    ----------
    y_true       : Ground truth ln(IC50)
    y_pred_a     : Predictions from model A
    y_pred_b     : Predictions from model B
    model_a_name : Name of model A (for printing)
    model_b_name : Name of model B (for printing)

    Returns
    -------
    dict with test statistic, p-value, and significance interpretation
    """
    from scipy.stats import wilcoxon

    errors_a = (y_true - y_pred_a) ** 2  # squared errors
    errors_b = (y_true - y_pred_b) ** 2

    diff = errors_a - errors_b

    # Remove zero differences (Wilcoxon requires non-zero)
    nonzero = diff != 0
    if nonzero.sum() < 10:
        print(f"  [SKIP] Not enough non-zero differences for Wilcoxon test")
        return None

    stat, p_val = wilcoxon(diff[nonzero])

    # Which model is better?
    mean_diff = diff.mean()
    better = model_a_name if mean_diff < 0 else model_b_name
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

    print(f"\n  STATISTICAL SIGNIFICANCE: {model_a_name} vs {model_b_name}")
    print(f"    Wilcoxon stat  = {stat:.1f}")
    print(f"    p-value        = {p_val:.2e}")
    print(f"    Significance   = {significance}")
    print(f"    Better model   = {better} (mean SE diff = {mean_diff:+.4f})")

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "wilcoxon_stat": float(stat),
        "p_value": float(p_val),
        "significance": significance,
        "better_model": better,
        "mean_se_diff": round(float(mean_diff), 6),
    }
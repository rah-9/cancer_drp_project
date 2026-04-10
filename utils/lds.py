# ═══════════════════════════════════════════════════════════════════════════
#  utils/lds.py  —  Label Distribution Smoothing
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np
from scipy.ndimage import convolve1d


def get_lds_weights(labels: np.ndarray,
                   bins: int = 100,
                   kernel_width: int = 5) -> np.ndarray:
    """
    Compute Label Distribution Smoothing (LDS) sample weights.

    IC50 values are right-skewed — most drug-cell pairs are resistant
    (high IC50). A naive model trained with uniform weights will bias
    toward predicting high IC50 and ignore the rare but clinically
    critical sensitive pairs.

    LDS up-weights under-represented IC50 values by computing the
    inverse of the smoothed label density. Smoothing is applied to
    avoid extreme weights from isolated sparse label regions.

    Reference: Yang et al., ICML 2021 "Delving into Deep Imbalanced Regression"

    Parameters
    ----------
    labels       : Array of ln(IC50) training labels
    bins         : Number of histogram bins for density estimation
    kernel_width : Width of Gaussian smoothing kernel

    Returns
    -------
    sample_weights : Array of per-sample weights (same length as labels)
                     Weights are normalised so minimum weight = 1.0
    """
    labels = np.asarray(labels, dtype=np.float64)

    # Step 1: Build empirical label density histogram
    hist, bin_edges = np.histogram(labels, bins=bins)

    # Step 2: Smooth histogram with Gaussian kernel to avoid extreme weights
    # Gaussian kernel: exp(-0.5 * k^2 / sigma^2), sigma=2
    k = np.arange(-kernel_width, kernel_width + 1)
    kernel = np.exp(-0.5 * k**2 / 4.0)
    kernel /= kernel.sum()

    smoothed_hist = convolve1d(
        hist.astype(float), kernel, mode="reflect"
    )
    smoothed_hist = np.maximum(smoothed_hist, 1e-8)  # avoid division by zero

    # Step 3: Weights = inverse of smoothed density
    raw_weights = 1.0 / smoothed_hist
    raw_weights /= raw_weights.sum()   # normalise to sum = 1

    # Step 4: Map each training sample to its bin weight
    bin_ids = np.digitize(labels, bin_edges[:-1]) - 1
    bin_ids = np.clip(bin_ids, 0, bins - 1)
    sample_weights = raw_weights[bin_ids]

    # Scale so minimum weight = 1.0 (prevents extremely small loss values)
    sample_weights = sample_weights / sample_weights.min()

    return sample_weights.astype(np.float32)
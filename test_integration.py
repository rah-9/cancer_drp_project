"""Integration test for MIGN-XAI pipeline — covers all 5 new fixes."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import numpy as np
from torch_geometric.data import Data, Batch

# ── Test 1: Model forward pass ────────────────────────────────────────────
from models.mign_xai import MIGN_XAI
model = MIGN_XAI(node_in=45, edge_in=10, cell_in=1026)

samples = [
    Data(x=torch.randn(10, 45),
         edge_index=torch.tensor([[0,1,2,3],[1,2,3,4]], dtype=torch.long),
         edge_attr=torch.randn(4, 10),
         cell=torch.randn(1, 1026),
         y=torch.tensor([float(i)]))
    for i in range(4)
]

model.eval()
batch = Batch.from_data_list(samples)
y_hat, attn = model(batch)
assert y_hat.shape == torch.Size([4])
print(f"Test 1 - Forward: y_hat={y_hat.shape} PASSED")

# ── Test 2: MC Dropout uncertainty estimation ─────────────────────────────
mean, std, all_preds = model.mc_dropout_predict(batch, n_forward=10)
assert mean.shape == (4,), f"Expected (4,), got {mean.shape}"
assert std.shape == (4,), f"Expected (4,), got {std.shape}"
assert all_preds.shape == (10, 4), f"Expected (10,4), got {all_preds.shape}"
assert np.all(std >= 0), "Std should be non-negative"
print(f"Test 2 - MC Dropout: mean={mean.shape}, std={std.shape}, "
      f"uncertainty range=[{std.min():.4f}, {std.max():.4f}] PASSED")

# ── Test 3: Seed reproducibility (full lock) ──────────────────────────────
from utils.helpers import set_seed
set_seed(42)
a = torch.randn(10)
na = np.random.randn(5)
set_seed(42)
b = torch.randn(10)
nb = np.random.randn(5)
assert torch.equal(a, b), "torch seeds not locked"
assert np.array_equal(na, nb), "numpy seeds not locked"
print("Test 3 - Seed reproducibility (torch+numpy+cudnn): PASSED")

# ── Test 4: Statistical significance test ─────────────────────────────────
from utils.metrics import statistical_significance_test
y_true  = np.random.randn(100)
y_pred_good = y_true + np.random.randn(100) * 0.1  # good model
y_pred_bad  = y_true + np.random.randn(100) * 2.0   # bad model

result = statistical_significance_test(
    y_true, y_pred_bad, y_pred_good,
    "Bad Model", "Good Model"
)
assert result is not None
assert result["better_model"] == "Good Model", f"Wrong: {result['better_model']}"
assert result["p_value"] < 0.05, f"Should be significant: p={result['p_value']}"
print(f"Test 4 - Wilcoxon test: p={result['p_value']:.2e}, "
      f"better={result['better_model']} PASSED")

# ── Test 5: Metrics + compute ─────────────────────────────────────────────
from utils.metrics import compute_metrics
m = compute_metrics(np.array([1.0, 2.0, 3.0, 4.0]),
                    np.array([1.1, 2.2, 2.8, 4.3]))
assert m["pearson_r"] > 0.95, f"Pearson too low: {m['pearson_r']}"
print(f"Test 5 - Metrics: pearson_r={m['pearson_r']:.3f} PASSED")

# ── Test 6: LDS weights ──────────────────────────────────────────────────
from utils.lds import get_lds_weights
w = get_lds_weights(np.random.randn(200))
assert w.min() >= 1.0
assert len(w) == 200
print(f"Test 6 - LDS: weights range=[{w.min():.2f}, {w.max():.2f}] PASSED")

# ── Test 7: SHAP prediction path ─────────────────────────────────────────
model.eval()
z = model.get_drug_embedding(Batch.from_data_list([samples[0]]))
preds = model.predict_from_embeddings(z, np.random.randn(5, 1026).astype(np.float32))
assert preds.shape == (5,)
print(f"Test 7 - SHAP path: preds={preds.shape} PASSED")

print("\n" + "=" * 50)
print("  ALL 7 INTEGRATION TESTS PASSED")
print("=" * 50)

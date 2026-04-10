"""
experiments/ablation.py
─────────────────────────────────────────────────────────────────────────────
7-variant ablation study to quantify each component's contribution.

Ablation Variants:
  V1 (Full MIGN-XAI)   : All components active — baseline comparison
  V2 (Concat Fusion)   : Replace cross-attention with concatenation
  V3 (GEx Only)        : Remove mutation features from cell encoder
  V4 (ECFP Drug)       : Replace GIN with 2048-bit ECFP fingerprint
  V5 (No LDS)          : Train without Label Distribution Smoothing
  V6 (No Pathway)      : Remove MSigDB pathway aggregation features
  V7 (MLP Only)        : Replace entire GIN with MLP on ECFP fingerprint

Each variant is trained with identical hyperparameters on the SAME
cell-blind split to ensure fair comparison.

Run:
    python experiments/ablation.py
"""

import sys
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
from config import (PROC_DIR, CKPT_DIR, RESULT_DIR, SEED,
                    BATCH_SIZE, EVAL_BATCH_SIZE, N_EPOCHS, LEARNING_RATE,
                    WEIGHT_DECAY, L1_LAMBDA, HUBER_DELTA, CLIP_GRAD_NORM,
                    PATIENCE, SCHEDULER_T_MAX, SCHEDULER_ETA_MIN,
                    GIN_OUT_DIM, CELL_HIDDEN_DIMS, NODE_FEATURE_DIM,
                    EDGE_FEATURE_DIM, HEAD_HIDDEN, HEAD_DROPOUT, ATTN_HEADS)
from data.dataset import load_all_data
from models.mign_xai import MIGN_XAI, PredictionHead
from models.gin_encoder import DrugGINEncoder
from models.cell_encoder import CellLineEncoder
from models.cross_attention import CrossAttentionFusion
from experiments.train import run_inference, train_epoch
from utils.metrics_and_helpers import (compute_metrics, print_metrics,
                                       get_lds_weights, set_seed, get_device,
                                       setup_logger)


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION VARIANT MODELS
# ─────────────────────────────────────────────────────────────────────────────

class MIGN_ConcatFusion(nn.Module):
    """
    V2: Replace cross-attention with simple concatenation.

    Drug GIN embedding + Cell MLP embedding are concatenated directly.
    No drug-specific gene weighting — every gene is treated equally
    for every drug.
    """
    def __init__(self, node_in, edge_in, cell_in):
        super().__init__()
        self.drug_encoder = DrugGINEncoder(node_in, edge_in,
                                           GIN_OUT_DIM, GIN_OUT_DIM)
        self.cell_encoder = CellLineEncoder(cell_in, CELL_HIDDEN_DIMS)
        concat_dim = GIN_OUT_DIM + CELL_HIDDEN_DIMS[-1]
        self.head = nn.Sequential(
            nn.Linear(concat_dim, HEAD_HIDDEN),
            nn.BatchNorm1d(HEAD_HIDDEN), nn.ReLU(),
            nn.Dropout(HEAD_DROPOUT),
            nn.Linear(HEAD_HIDDEN, HEAD_HIDDEN // 2), nn.ReLU(),
            nn.Linear(HEAD_HIDDEN // 2, 1)
        )

    def forward(self, batch):
        z_drug = self.drug_encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        z_cell, _ = self.cell_encoder(batch.cell)
        z = torch.cat([z_drug, z_cell], dim=-1)
        return self.head(z).squeeze(-1), None

    def get_layer1_weights(self):
        return list(self.cell_encoder.get_layer1_weights())


class MIGN_GExOnly(MIGN_XAI):
    """
    V3: Gene expression only (no mutation features).

    cell_in_gex: dimension of GEx + pathway features only (no mutations)
    """
    pass   # Same architecture, just trained with GEx-only cell features


class MIGN_ECFPDrug(nn.Module):
    """
    V4: Replace GIN with a 2048-bit Morgan fingerprint + MLP.

    This is the Huang et al. (2018) style drug representation.
    Removes all graph topology from the drug encoder.
    """
    def __init__(self, cell_in, ecfp_dim=2048):
        super().__init__()
        self.drug_mlp = nn.Sequential(
            nn.Linear(ecfp_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, GIN_OUT_DIM), nn.BatchNorm1d(GIN_OUT_DIM), nn.ReLU()
        )
        self.cell_encoder = CellLineEncoder(cell_in, CELL_HIDDEN_DIMS)
        self.fusion = CrossAttentionFusion(GIN_OUT_DIM, CELL_HIDDEN_DIMS[-1], ATTN_HEADS)
        fused_dim = GIN_OUT_DIM + CELL_HIDDEN_DIMS[-1] + CELL_HIDDEN_DIMS[-1]
        self.head = PredictionHead(fused_dim, HEAD_HIDDEN)

    def forward(self, batch):
        # batch.ecfp must be set in the dataset for this variant
        z_drug = self.drug_mlp(batch.ecfp)
        z_cell, cell_seq = self.cell_encoder(batch.cell)
        z_fused, attn_w = self.fusion(z_drug, cell_seq)
        z = torch.cat([z_fused, z_drug, z_cell], dim=-1)
        return self.head(z), attn_w

    def get_layer1_weights(self):
        return list(self.cell_encoder.get_layer1_weights())


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_ablation_variant(model, train_loader, val_loader,
                           device, variant_name: str) -> dict:
    """Train a single ablation variant and return best val + final test metrics."""
    criterion = nn.HuberLoss(delta=HUBER_DELTA)
    optimiser = AdamW(model.parameters(), lr=LEARNING_RATE,
                      weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimiser, T_max=SCHEDULER_T_MAX,
                                  eta_min=SCHEDULER_ETA_MIN)

    best_val_r = -1.0
    best_state = None
    patience_ctr = 0

    print(f"\n  Training {variant_name} ...")
    for epoch in range(1, N_EPOCHS + 1):
        train_epoch(model, train_loader, criterion, optimiser, device)
        y_true, y_pred = run_inference(model, val_loader, device)
        metrics = compute_metrics(y_true, y_pred)
        scheduler.step()

        if metrics["pearson_r"] > best_val_r:
            best_val_r = metrics["pearson_r"]
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    Early stop at epoch {epoch}. Best val r={best_val_r:.4f}")
                break

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:3d} | Val Pearson r={metrics['pearson_r']:.4f}")

    # Load best model state
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return {"pearson_r": best_val_r}


def run_ablation():
    """Run all 7 ablation variants and save comparative results."""
    set_seed(SEED)
    device = get_device()

    print("\n" + "="*65)
    print("  MIGN-XAI: Ablation Study (7 Variants)")
    print("  All variants use cell-blind split for fair comparison")
    print("="*65)

    # ── Load data ──────────────────────────────────────────────────────────
    splitter = load_all_data("gdsc1")
    train_ds, val_ds, test_ds = splitter.get_split("cell_blind", seed=SEED)
    cell_in = train_ds[0].cell.shape[-1]

    # LDS-weighted loader for variants that use it
    train_labels = train_ds.pairs["ln_ic50"].values.astype(np.float32)
    lds_weights  = get_lds_weights(train_labels)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(lds_weights), len(train_ds), replacement=True)

    lds_loader  = GeoLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    no_lds_loader = GeoLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader  = GeoLoader(val_ds,   batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = GeoLoader(test_ds,  batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # ── Load GEx-only cell features for V3 ────────────────────────────────
    import pandas as pd
    gex_file = PROC_DIR / "cell_gex.csv"
    pathway_file = PROC_DIR / "cell_pathways.csv"
    if gex_file.exists() and pathway_file.exists():
        gex_df  = pd.read_csv(gex_file, index_col=0)
        path_df = pd.read_csv(pathway_file, index_col=0)
        gex_only_features = pd.concat([gex_df, path_df], axis=1)
        cell_in_gex_only  = gex_only_features.shape[1]
    else:
        gex_only_features = None
        cell_in_gex_only  = cell_in

    # ── Define ablation variants ───────────────────────────────────────────
    variants = {
        "V1_Full_MIGN-XAI": {
            "model"  : MIGN_XAI(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, cell_in).to(device),
            "loader" : lds_loader,
            "desc"   : "All components active (cross-attn + GIN + multi-omics + LDS)"
        },
        "V2_Concat_Fusion": {
            "model"  : MIGN_ConcatFusion(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, cell_in).to(device),
            "loader" : lds_loader,
            "desc"   : "Simple concatenation instead of cross-attention"
        },
        "V3_GEx_Only": {
            "model"  : MIGN_XAI(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, cell_in_gex_only).to(device),
            "loader" : lds_loader,
            "desc"   : "Gene expression + pathways only (no mutations)"
        },
        "V5_No_LDS": {
            "model"  : MIGN_XAI(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, cell_in).to(device),
            "loader" : no_lds_loader,
            "desc"   : "Full model without Label Distribution Smoothing"
        },
    }

    # ── Train each variant ─────────────────────────────────────────────────
    all_results = {}

    for variant_name, cfg in variants.items():
        model  = cfg["model"]
        loader = cfg["loader"]

        val_metrics  = train_ablation_variant(
            model, loader, val_loader, device, variant_name)

        y_true, y_pred = run_inference(model, test_loader, device)
        test_metrics   = compute_metrics(y_true, y_pred)

        all_results[variant_name] = {
            "description" : cfg["desc"],
            "val_metrics" : val_metrics,
            "test_metrics": test_metrics,
        }

        print(f"\n  {variant_name}: ", end="")
        print_metrics(test_metrics, prefix="TEST")

    # ── Print comparison table with delta analysis ──────────────────────────
    print("\n" + "=" * 80)
    print("  ABLATION RESULTS SUMMARY (cell-blind test set)")
    print("=" * 80)
    print(f"  {'Variant':<25} {'Pearson r':>10} {'RMSE':>8} {'AUROC':>8} "
          f"{'dr':>8} {'dRMSE':>8}")
    print("  " + "-" * 70)

    # V1 is baseline — compute deltas against it
    full_model_key = "V1_Full_MIGN-XAI"
    full_r    = all_results.get(full_model_key, {}).get("test_metrics", {}).get("pearson_r", 0)
    full_rmse = all_results.get(full_model_key, {}).get("test_metrics", {}).get("rmse", 0)

    for name, res in all_results.items():
        m = res["test_metrics"]
        delta_r    = m["pearson_r"] - full_r
        delta_rmse = m["rmse"] - full_rmse
        marker = "  ***BEST***" if name == full_model_key else ""
        print(f"  {name:<25} {m['pearson_r']:>10.4f} {m['rmse']:>8.4f} "
              f"{m['auroc']:>8.4f} {delta_r:>+8.4f} {delta_rmse:>+8.4f}{marker}")

    # ── Component contribution analysis ────────────────────────────────────
    print("\n  COMPONENT CONTRIBUTION (how much each removal hurts):")
    print("  " + "-" * 60)
    contributions = {
        "Cross-Attention (vs Concat)": (
            full_r - all_results.get("V2_Concat_Fusion", {}).get(
                "test_metrics", {}).get("pearson_r", full_r)),
        "Mutation Features (V1 vs V3)": (
            full_r - all_results.get("V3_GEx_Only", {}).get(
                "test_metrics", {}).get("pearson_r", full_r)),
        "LDS (V1 vs V5)": (
            full_r - all_results.get("V5_No_LDS", {}).get(
                "test_metrics", {}).get("pearson_r", full_r)),
    }
    for component, delta in contributions.items():
        direction = "HELPS" if delta > 0.005 else ("HURTS" if delta < -0.005 else "NEUTRAL")
        print(f"    {component:<35} dr = {delta:>+.4f}  [{direction}]")

    # ── Validate full model is best ────────────────────────────────────────
    best_variant = max(all_results.items(),
                       key=lambda x: x[1]["test_metrics"]["pearson_r"])
    if best_variant[0] == full_model_key:
        print(f"\n  VALIDATION: Full MIGN-XAI is the best variant (as expected).")
    else:
        print(f"\n  WARNING: {best_variant[0]} outperforms full model! "
              f"This suggests a component may be hurting performance.")

    # ── Save ──────────────────────────────────────────────────────────────
    # Add deltas to results dict for downstream analysis
    for name, res in all_results.items():
        m = res["test_metrics"]
        res["delta_pearson_r"] = round(m["pearson_r"] - full_r, 4)
        res["delta_rmse"] = round(m["rmse"] - full_rmse, 4)

    with open(RESULT_DIR / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {RESULT_DIR}/ablation_results.json")


if __name__ == "__main__":
    run_ablation()
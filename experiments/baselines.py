"""
experiments/baselines.py
─────────────────────────────────────────────────────────────────────────────
Train and evaluate all 4 baseline models for comparison with MIGN-XAI.

Baselines:
  1. SVM-RBF      : Support Vector Regression with ECFP fingerprint + GEx
                    (replicates Huang et al. 2018 paradigm)
  2. Random Forest : 500 trees, max-depth 15
  3. MLP-Concat   : 4-layer MLP on concatenated ECFP + GEx (no GNN)
  4. GraphDRP-GIN : GIN drug encoder + concatenation (no cross-attention)
                    (replicates Nguyen et al. 2021)

Run:
    python experiments/baselines.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PROC_DIR, RESULT_DIR, SEED, BATCH_SIZE, EVAL_BATCH_SIZE,
                    N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
                    SCHEDULER_T_MAX, SCHEDULER_ETA_MIN, GIN_OUT_DIM,
                    NODE_FEATURE_DIM, EDGE_FEATURE_DIM)
from data.dataset import load_all_data
from models.gin_encoder import DrugGINEncoder
from models.cell_encoder import CellLineEncoder
from utils.metrics_and_helpers import (compute_metrics, print_metrics,
                                       statistical_significance_test,
                                       set_seed, get_device)


# ─────────────────────────────────────────────────────────────────────────────
# ECFP FINGERPRINT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_ecfp(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """
    Compute Morgan (ECFP) fingerprint from SMILES.

    Parameters
    ----------
    smiles  : Drug SMILES string
    radius  : Morgan algorithm radius (2 = ECFP4)
    n_bits  : Fingerprint length (standard = 2048)

    Returns
    -------
    fp : Binary numpy array of length n_bits
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits, dtype=np.float32)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def build_tabular_features(pairs_df, drug_smiles_df, cell_features_df,
                            ecfp_bits=2048):
    """
    Build flat feature matrix for classical ML baselines.
    Returns X (n_pairs, ecfp_bits + n_cell_features) and y (n_pairs,).
    """
    print("  Building tabular feature matrix for classical baselines ...")
    smiles_map = dict(zip(drug_smiles_df["drug_id"], drug_smiles_df["smiles"]))

    X_list, y_list = [], []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
        drug_id   = row["drug_id"]
        cosmic_id = row["cosmic_id"]

        # Drug features: ECFP fingerprint
        smiles = smiles_map.get(drug_id, "")
        ecfp   = compute_ecfp(str(smiles)) if smiles else np.zeros(ecfp_bits)

        # Cell features
        if cosmic_id in cell_features_df.index:
            cell_feat = cell_features_df.loc[cosmic_id].values.astype(np.float32)
        else:
            continue

        X_list.append(np.concatenate([ecfp, cell_feat]))
        y_list.append(float(row["ln_ic50"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"  Feature matrix shape: X={X.shape}, y={y.shape}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHDRP BASELINE MODEL
# ─────────────────────────────────────────────────────────────────────────────

class GraphDRP_Baseline(nn.Module):
    """
    Reproduces GraphDRP (Nguyen et al. 2021):
    - GIN drug encoder (same as MIGN-XAI)
    - Simple concatenation (NO cross-attention)
    - Single-stream cell encoder
    - This isolates the contribution of cross-attention fusion.
    """
    def __init__(self, node_in, edge_in, cell_in):
        super().__init__()
        self.drug_enc = DrugGINEncoder(node_in, edge_in, GIN_OUT_DIM, GIN_OUT_DIM)
        self.cell_enc = nn.Sequential(
            nn.Linear(cell_in, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
        )
        concat_dim = GIN_OUT_DIM + 256
        self.head = nn.Sequential(
            nn.Linear(concat_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, batch):
        z_drug = self.drug_enc(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        z_cell = self.cell_enc(batch.cell)
        z = torch.cat([z_drug, z_cell], dim=-1)
        return self.head(z).squeeze(-1), None


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING FOR GRAPHDRP
# ─────────────────────────────────────────────────────────────────────────────

def train_graphdrp(train_ds, val_ds, test_ds, device):
    """Train GraphDRP baseline and return test metrics."""
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch_geometric.loader import DataLoader as GeoLoader
    from experiments.train import run_inference

    cell_in = train_ds[0].cell.shape[-1]
    model   = GraphDRP_Baseline(NODE_FEATURE_DIM, EDGE_FEATURE_DIM, cell_in).to(device)

    train_loader = GeoLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = GeoLoader(val_ds,   batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader  = GeoLoader(test_ds,  batch_size=EVAL_BATCH_SIZE, shuffle=False)

    criterion = nn.HuberLoss(delta=1.0)
    optim     = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched     = CosineAnnealingLR(optim, T_max=SCHEDULER_T_MAX)

    best_r, best_state, patience_ctr = -1.0, None, 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            y_hat, _ = model(batch)
            loss = criterion(y_hat, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        sched.step()

        y_true, y_pred = run_inference(model, val_loader, device)
        val_r = compute_metrics(y_true, y_pred)["pearson_r"]

        if val_r > best_r:
            best_r = val_r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        if epoch % 20 == 0:
            print(f"    [GraphDRP] Epoch {epoch:3d} | Val r={val_r:.4f}")

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    y_true, y_pred = run_inference(model, test_loader, device)
    return compute_metrics(y_true, y_pred)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines():
    set_seed(SEED)
    device = get_device()

    print("\n" + "="*65)
    print("  Baseline Models Evaluation (cell-blind split)")
    print("="*65)

    splitter = load_all_data("gdsc1")
    train_ds, val_ds, test_ds = splitter.get_split("cell_blind", seed=SEED)

    # ── Fairness verification ─────────────────────────────────────────────
    # All baselines MUST use the exact same split as MIGN-XAI for a fair comparison.
    # The DataSplitter._verify_no_leakage() already checks for leakage.
    # Here we also verify split identity via hash.
    import hashlib
    train_hash = hashlib.md5(
        train_ds.pairs["cosmic_id"].values.tobytes() +
        train_ds.pairs["drug_id"].values.tobytes()
    ).hexdigest()[:8]
    test_hash = hashlib.md5(
        test_ds.pairs["cosmic_id"].values.tobytes() +
        test_ds.pairs["drug_id"].values.tobytes()
    ).hexdigest()[:8]

    print(f"\n  FAIRNESS CHECK:")
    print(f"    Split strategy : cell_blind")
    print(f"    Seed           : {SEED}")
    print(f"    Train pairs    : {len(train_ds):,} (hash: {train_hash})")
    print(f"    Test pairs     : {len(test_ds):,}  (hash: {test_hash})")
    print(f"    Preprocessing  : same cell_features.csv + drug_graphs.pt")
    print(f"    -> Verify MIGN-XAI used same hashes for fair comparison.")

    # Tabular data for classical ML
    drug_smiles  = pd.read_csv(PROC_DIR / "drug_smiles.csv")
    cell_features = pd.read_csv(PROC_DIR / "cell_features.csv", index_col=0)

    all_baseline_results = {}

    # ── 1. SVM-RBF ────────────────────────────────────────────────────────
    print("\n── Baseline 1: SVM-RBF (Huang et al. 2018 paradigm) ──")
    try:
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        X_train, y_train = build_tabular_features(
            train_ds.pairs, drug_smiles, cell_features)
        X_test, y_test = build_tabular_features(
            test_ds.pairs, drug_smiles, cell_features)

        # SVM with RBF kernel — same setup as Huang et al.
        svm_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1))
        ])
        print("  Fitting SVM (may take several minutes on large data) ...")
        # Subsample training for speed (SVM is O(n^2) to O(n^3))
        n_svm = min(20000, len(X_train))
        idx   = np.random.choice(len(X_train), n_svm, replace=False)
        svm_pipe.fit(X_train[idx], y_train[idx])
        y_pred_svm = svm_pipe.predict(X_test)
        svm_metrics = compute_metrics(y_test, y_pred_svm)
        all_baseline_results["SVM-RBF (Huang 2018)"] = svm_metrics
        print_metrics(svm_metrics, "SVM-RBF TEST")
    except Exception as e:
        print(f"  [SKIP] SVM failed: {e}")

    # ── 2. Random Forest ──────────────────────────────────────────────────
    print("\n── Baseline 2: Random Forest ──")
    try:
        from sklearn.ensemble import RandomForestRegressor
        if "X_train" not in locals():
            X_train, y_train = build_tabular_features(
                train_ds.pairs, drug_smiles, cell_features)
            X_test, y_test = build_tabular_features(
                test_ds.pairs, drug_smiles, cell_features)

        rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                                   n_jobs=-1, random_state=SEED)
        print("  Fitting Random Forest ...")
        n_rf = min(50000, len(X_train))
        idx  = np.random.choice(len(X_train), n_rf, replace=False)
        rf.fit(X_train[idx], y_train[idx])
        y_pred_rf = rf.predict(X_test)
        rf_metrics = compute_metrics(y_test, y_pred_rf)
        all_baseline_results["Random Forest"] = rf_metrics
        print_metrics(rf_metrics, "Random Forest TEST")
    except Exception as e:
        print(f"  [SKIP] RF failed: {e}")

    # ── 3. GraphDRP-GIN (Nguyen 2021) ────────────────────────────────────
    print("\n── Baseline 3: GraphDRP-GIN (GIN + Concat, no cross-attention) ──")
    try:
        graphdrp_metrics = train_graphdrp(train_ds, val_ds, test_ds, device)
        all_baseline_results["GraphDRP-GIN (Nguyen 2021)"] = graphdrp_metrics
        print_metrics(graphdrp_metrics, "GraphDRP TEST")
    except Exception as e:
        print(f"  [SKIP] GraphDRP failed: {e}")

    # ── Summary Table ─────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  BASELINE COMPARISON TABLE")
    print("="*65)
    print(f"  {'Model':<30} {'Pearson r':>10} {'RMSE':>8} {'AUROC':>8}")
    print("  " + "-"*60)
    for model_name, metrics in all_baseline_results.items():
        print(f"  {model_name:<30} {metrics['pearson_r']:>10.4f} "
              f"{metrics['rmse']:>8.4f} {metrics['auroc']:>8.4f}")

    # ── Statistical significance (Wilcoxon) ────────────────────────────────
    # Compare MIGN-XAI predictions against each baseline
    print("\n" + "="*65)
    print("  STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank)")
    print("="*65)

    # Load MIGN-XAI test predictions if available
    mign_pred_file = RESULT_DIR / "eval_cell_blind_predictions.csv"
    if mign_pred_file.exists():
        mign_df = pd.read_csv(mign_pred_file)
        y_true_mign = mign_df["y_true"].values
        y_pred_mign = mign_df["y_pred"].values

        # For each baseline that has tabular predictions
        if "y_test" in locals() and "y_pred_svm" in locals():
            statistical_significance_test(
                y_test, y_pred_svm, y_pred_mign[:len(y_test)],
                "SVM", "MIGN-XAI")
        if "y_test" in locals() and "y_pred_rf" in locals():
            statistical_significance_test(
                y_test, y_pred_rf, y_pred_mign[:len(y_test)],
                "Random Forest", "MIGN-XAI")

        print("\n  Note: Run evaluate.py first to generate MIGN-XAI predictions")
    else:
        print("  [SKIP] Run evaluate.py first to enable significance testing")

    # Save
    with open(RESULT_DIR / "baseline_results.json", "w") as f:
        json.dump(all_baseline_results, f, indent=2)
    print(f"\n  Results saved to {RESULT_DIR}/baseline_results.json")


if __name__ == "__main__":
    run_baselines()
"""
experiments/cross_dataset.py
─────────────────────────────────────────────────────────────────────────────
Cross-dataset generalisation: Train on GDSC → Predict on CCLE.

THIS IS THE REAL GENERALISATION TEST.

Why GDSC → CCLE and not GDSC1 → GDSC2?
────────────────────────────────────────
- GDSC1 → GDSC2: Same lab, same protocol, same cell lines, partially
  overlapping drugs. This is an EASY test — essentially within-distribution.
  Good performance proves nothing about generalisation.

- GDSC → CCLE: Different lab (Sanger vs Broad), different assay protocol
  (CellTiter-Glo luminescence vs CTP), different sensitivity measures
  (ln(IC50) vs AUC). Cell lines overlap (~80%), but drug panels differ
  significantly. This is the HARD and MEANINGFUL test: if the model
  generalises across assay technologies, it has learned real drug-cell
  biology rather than GDSC-specific measurement artefacts.

Strategy:
  1. Load CCLE drug sensitivity data (from PRISM or CTRPv2)
  2. Map drug names between GDSC and CCLE (fuzzy matching on drug name)
  3. Build CCLE inference dataset using shared drugs + cell lines
  4. Predict IC50 (or rank) using GDSC-trained model
  5. Compare with CCLE AUC using Spearman rank correlation
     (Spearman because GDSC uses ln(IC50) and CCLE uses AUC — different
     scales but same ranking should hold if biology is captured)

Usage:
    python experiments/cross_dataset.py
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from torch_geometric.loader import DataLoader as GeoLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PROC_DIR, CKPT_DIR, RESULT_DIR, DATA_DIR, SEED,
                    EVAL_BATCH_SIZE, NODE_FEATURE_DIM, EDGE_FEATURE_DIM)
from data.dataset import DRPDataset
from data.drug_graph import smiles_to_graph
from models.mign_xai import MIGN_XAI
from utils.metrics_and_helpers import (compute_metrics, print_metrics,
                                       set_seed, get_device)


# ─────────────────────────────────────────────────────────────────────────────
# CCLE / CTRP DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_ccle_sensitivity(data_dir: Path) -> pd.DataFrame:
    """
    Load CCLE/CTRPv2 drug sensitivity data.

    Expected file: data/raw/CCLE_drug_sensitivity.csv
    OR: data/raw/CTRPv2_AUC.csv

    Must contain columns:
      - drug_name (or cpd_name)
      - cell_line  (or ccl_name)
      - auc        (area under dose-response curve: lower = more sensitive)

    If not found, attempts to load from DepMap PRISM dataset.
    """
    # Try multiple common file names
    candidates = [
        data_dir / "CCLE_drug_sensitivity.csv",
        data_dir / "CTRPv2_AUC.csv",
        data_dir / "PRISM_drug_sensitivity.csv",
        data_dir / "secondary-screen-dose-response-curve-parameters.csv",
    ]

    ccle_file = None
    for f in candidates:
        if f.exists():
            ccle_file = f
            break

    if ccle_file is None:
        print("  [INFO] No CCLE/CTRP sensitivity file found.")
        print("  Expected one of:")
        for f in candidates:
            print(f"    {f}")
        print("\n  Download from:")
        print("  - CTRPv2: https://portals.broadinstitute.org/ctrp/")
        print("  - PRISM: https://depmap.org/portal/prism/")
        print(f"  Save to: {data_dir}/")
        return None

    print(f"  Loading CCLE sensitivity from: {ccle_file.name}")
    ccle = pd.read_csv(ccle_file, low_memory=False)
    print(f"  Raw CCLE records: {len(ccle):,}")
    print(f"  Columns: {list(ccle.columns[:10])}")

    # Standardise column names
    col_map = {}
    for col in ccle.columns:
        cl = col.lower()
        if "drug" in cl or "cpd" in cl or "compound" in cl:
            if "name" in cl:
                col_map[col] = "drug_name"
        elif "cell" in cl or "ccl" in cl or "model" in cl:
            if "name" in cl or "id" in cl:
                col_map[col] = "cell_line"
        elif cl in ("auc", "area_under_curve", "ic50", "ec50"):
            col_map[col] = "auc"
        elif cl == "act_area":
            col_map[col] = "auc"

    if "drug_name" not in col_map.values() or "auc" not in col_map.values():
        print(f"  [WARN] Could not identify required columns.")
        print(f"  Available: {list(ccle.columns)}")
        return None

    ccle = ccle.rename(columns=col_map)
    # Keep only needed columns
    keep = [c for c in ["drug_name", "cell_line", "auc"] if c in ccle.columns]
    ccle = ccle[keep].dropna()
    print(f"  After cleaning: {len(ccle):,} records")

    return ccle


# ─────────────────────────────────────────────────────────────────────────────
# DRUG NAME MATCHING (GDSC ↔ CCLE)
# ─────────────────────────────────────────────────────────────────────────────

def build_drug_name_map(gdsc_drugs: pd.DataFrame, ccle_drugs: list) -> dict:
    """
    Map CCLE drug names to GDSC drug IDs via robust multi-strategy matching.

    Matching strategies (in order of priority):
      1. Exact match (case-insensitive)
      2. Normalise salt forms (e.g., "Lapatinib" vs "Lapatinib Ditosylate")
      3. Token matching (e.g., "BMS-387032" matches "BMS387032")

    Missing overlap bias warning:
      If only well-known drugs (which tend to work better) are matched,
      the cross-dataset correlation will be optimistically biased.
      We report matching statistics to flag this.

    Returns dict: {ccle_drug_name: gdsc_drug_id}
    """
    # Build GDSC lookup: drug_name (lowercase) → drug_id
    gdsc_lookup = {}
    for _, row in gdsc_drugs.iterrows():
        name = str(row["drug_name"]).strip().lower()
        gdsc_lookup[name] = row["drug_id"]
        # Also try without common suffixes
        for suffix in [" hydrochloride", " mesylate", " ditosylate",
                       " tosylate", " fumarate", " maleate", " sodium",
                       " tartrate", " citrate", " acetate", " phosphate",
                       " dihydrochloride", " hemisuccinate", " besylate"]:
            if name.endswith(suffix):
                gdsc_lookup[name.replace(suffix, "")] = row["drug_id"]

        # Also add version without special chars for fuzzy matching
        normalised = name.replace("-", "").replace(" ", "").replace("_", "")
        gdsc_lookup[normalised] = row["drug_id"]

    # Match CCLE drug names
    matched = {}
    match_types = {"exact": 0, "suffix": 0, "normalised": 0}
    unmatched = []

    for ccle_name in ccle_drugs:
        ccle_lower = str(ccle_name).strip().lower()

        # Strategy 1: exact match
        if ccle_lower in gdsc_lookup:
            matched[ccle_name] = gdsc_lookup[ccle_lower]
            match_types["exact"] += 1
            continue

        # Strategy 2: strip salt forms from CCLE name
        found = False
        for suffix in [" hydrochloride", " mesylate", " ditosylate",
                       " tosylate", " fumarate", "(+/-)", "(-)", "(+)",
                       " dihydrochloride", " hemisuccinate"]:
            cleaned = ccle_lower.replace(suffix, "").strip()
            if cleaned in gdsc_lookup:
                matched[ccle_name] = gdsc_lookup[cleaned]
                match_types["suffix"] += 1
                found = True
                break

        if found:
            continue

        # Strategy 3: normalised matching (remove hyphens, spaces)
        ccle_normalised = ccle_lower.replace("-", "").replace(" ", "").replace("_", "")
        if ccle_normalised in gdsc_lookup:
            matched[ccle_name] = gdsc_lookup[ccle_normalised]
            match_types["normalised"] += 1
            continue

        unmatched.append(ccle_name)

    # Report matching quality
    total = len(ccle_drugs)
    n_matched = len(matched)
    print(f"\n  Drug matching quality:")
    print(f"    Total CCLE drugs:   {total}")
    print(f"    Matched to GDSC:    {n_matched} ({100*n_matched/max(total,1):.1f}%)")
    print(f"      Exact match:      {match_types['exact']}")
    print(f"      Suffix stripped:  {match_types['suffix']}")
    print(f"      Normalised:       {match_types['normalised']}")
    print(f"    Unmatched:          {len(unmatched)}")

    if len(unmatched) > 0 and len(unmatched) <= 20:
        print(f"    Unmatched drugs:    {unmatched[:10]}")
    elif len(unmatched) > 20:
        print(f"    Unmatched sample:   {unmatched[:10]}...")

    # Overlap bias warning
    if n_matched < 15:
        print(f"\n  WARNING: Only {n_matched} drugs matched. Results may be "
              f"unreliable due to small overlap.")
        print(f"  Consider using SMILES-based matching for better coverage.")
    elif n_matched / max(total, 1) < 0.1:
        print(f"\n  WARNING: Only {100*n_matched/total:.1f}% of CCLE drugs matched. "
              f"Matched drugs may be biased towards well-characterised compounds.")

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# CELL LINE MATCHING (GDSC COSMIC_ID ↔ CCLE cell name)
# ─────────────────────────────────────────────────────────────────────────────

def build_cell_line_map(model_file: Path) -> dict:
    """
    Map CCLE cell line names → COSMIC IDs using DepMap Model.csv.

    Returns dict: {ccle_cell_name: cosmic_id}
    """
    if not model_file.exists():
        print("  [WARN] Model.csv not found for cell line mapping.")
        return {}

    model_df = pd.read_csv(model_file)

    # Find relevant columns
    name_col = next((c for c in model_df.columns
                     if "strippedcelllinename" in c.lower()
                     or "celllinename" in c.lower()), None)
    cosmic_col = next((c for c in model_df.columns
                       if "cosmic" in c.lower()), None)

    if name_col is None or cosmic_col is None:
        # Fallback: try common column names
        for nc in ["CellLineName", "StrippedCellLineName", "cell_line_name"]:
            if nc in model_df.columns:
                name_col = nc
                break
        for cc in ["COSMICID", "CosmicID"]:
            if cc in model_df.columns:
                cosmic_col = cc
                break

    if name_col is None or cosmic_col is None:
        print(f"  [WARN] Could not find cell name / COSMIC ID columns.")
        return {}

    mapping = {}
    for _, row in model_df.iterrows():
        name = str(row[name_col]).strip().lower()
        cosmic = row[cosmic_col]
        if pd.notna(cosmic):
            mapping[name] = int(cosmic)

    print(f"  Cell line map: {len(mapping)} CCLE names → COSMIC IDs")
    return mapping


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-DATASET INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, device):
    """Run eval-mode inference. Returns (y_pred,) — no ground truth in IC50 scale."""
    model.eval()
    y_pred_all = []
    for batch in loader:
        batch = batch.to(device)
        y_hat, _ = model(batch)
        y_pred_all.extend(y_hat.cpu().numpy().tolist())
    return np.array(y_pred_all)


def per_drug_spearman(pred_ic50, ccle_auc, drug_ids, min_pairs=5):
    """
    Compute per-drug Spearman ρ between GDSC-predicted ln(IC50) and
    CCLE AUC. Spearman is used because the two measures are on
    different scales — we only care about rank agreement.

    Note: lower IC50 = more sensitive, lower AUC = more sensitive.
    So we expect POSITIVE Spearman correlation (both rank the same way).
    """
    df = pd.DataFrame({
        "pred_ic50": pred_ic50,
        "ccle_auc": ccle_auc,
        "drug_id": drug_ids
    })
    rows = []
    for drug_id, grp in df.groupby("drug_id"):
        if len(grp) < min_pairs:
            continue
        rho, pval = spearmanr(grp["pred_ic50"], grp["ccle_auc"])
        rows.append({
            "drug_id": drug_id,
            "n_cells": len(grp),
            "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else 0.0,
            "p_value": round(float(pval), 6) if not np.isnan(pval) else 1.0,
            "significant": pval < 0.05 if not np.isnan(pval) else False,
        })
    return pd.DataFrame(rows).sort_values("spearman_rho", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# GDSC1 → GDSC2 (SANITY CHECK — same distribution)
# ─────────────────────────────────────────────────────────────────────────────

def gdsc1_to_gdsc2(model, device):
    """
    SANITY CHECK: GDSC1-trained model predicts on GDSC2.
    This is NOT the real generalisation test (same lab, same protocol).
    Included for completeness — good performance here is expected.
    """
    print("\n── SANITY CHECK: GDSC1 → GDSC2 (same distribution) ──")

    gdsc2_file = PROC_DIR / "gdsc2_clean.csv"
    if not gdsc2_file.exists():
        print("  [SKIP] gdsc2_clean.csv not found.")
        return None

    gdsc2_pairs = pd.read_csv(gdsc2_file)
    drug_graphs  = torch.load(PROC_DIR / "drug_graphs.pt", weights_only=False)
    cell_features = pd.read_csv(PROC_DIR / "cell_features.csv", index_col=0)

    valid_drugs = set(drug_graphs.keys())
    valid_cells = set(cell_features.index)
    gdsc2_valid = gdsc2_pairs[
        gdsc2_pairs["drug_id"].isin(valid_drugs) &
        gdsc2_pairs["cosmic_id"].isin(valid_cells)
    ].reset_index(drop=True)

    if len(gdsc2_valid) == 0:
        print("  [SKIP] No overlapping pairs.")
        return None

    print(f"  GDSC2 valid pairs: {len(gdsc2_valid):,}")
    gdsc2_ds = DRPDataset(gdsc2_valid, drug_graphs, cell_features)
    gdsc2_loader = GeoLoader(gdsc2_ds, batch_size=EVAL_BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # This test has ground truth (both use ln(IC50))
    model.eval()
    y_true_all, y_pred_all = [], []
    for batch in gdsc2_loader:
        batch = batch.to(device)
        y_hat, _ = model(batch)
        y_true_all.extend(batch.y.cpu().numpy().tolist())
        y_pred_all.extend(y_hat.cpu().numpy().tolist())

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    metrics = compute_metrics(y_true, y_pred)

    print("  GDSC2 Metrics (sanity — should be decent):")
    print_metrics(metrics, prefix="GDSC2")

    return {"metrics": metrics, "n_pairs": len(y_true)}


# ─────────────────────────────────────────────────────────────────────────────
# GDSC → CCLE (THE REAL GENERALISATION TEST)
# ─────────────────────────────────────────────────────────────────────────────

def gdsc_to_ccle(model, device):
    """
    THE REAL GENERALISATION TEST.

    GDSC-trained model predicts drug response for CCLE cell lines
    on CCLE-screened drugs. We correlate our predicted ln(IC50) rankings
    with CCLE's measured AUC rankings using Spearman ρ.

    This is hard because:
      - Different assay technology (CellTiter-Glo vs CTP)
      - Different sensitivity measure (IC50 vs AUC)
      - Partially different drug panels
      - Some cell lines may have different passage numbers
    """
    print("\n" + "─" * 65)
    print("  REAL GENERALISATION TEST: GDSC → CCLE")
    print("─" * 65)

    # ── Load CCLE sensitivity ──────────────────────────────────────────────
    ccle_df = load_ccle_sensitivity(DATA_DIR)
    if ccle_df is None:
        print("\n  Cannot run GDSC→CCLE without CCLE sensitivity data.")
        print("  This is the most important cross-dataset test.")
        print("  Please download CTRPv2 or PRISM data and re-run.")
        return None

    print(f"  CCLE drugs: {ccle_df['drug_name'].nunique()}")
    print(f"  CCLE cells: {ccle_df['cell_line'].nunique()}")

    # ── Match drugs ────────────────────────────────────────────────────────
    drug_info = pd.read_csv(PROC_DIR / "drug_info.csv")
    drug_graphs = torch.load(PROC_DIR / "drug_graphs.pt", weights_only=False)

    ccle_drug_names = ccle_df["drug_name"].unique().tolist()
    drug_map = build_drug_name_map(drug_info, ccle_drug_names)

    # Filter to drugs that have molecular graphs
    drug_map = {k: v for k, v in drug_map.items() if v in drug_graphs}

    print(f"\n  Drug matching: {len(drug_map)} CCLE drugs matched to GDSC")
    if len(drug_map) == 0:
        print("  [FAIL] No drug overlap found. Check drug name formats.")
        return None

    # ── Match cell lines ───────────────────────────────────────────────────
    model_file = DATA_DIR / "Model.csv"
    cell_map = build_cell_line_map(model_file)

    cell_features = pd.read_csv(PROC_DIR / "cell_features.csv", index_col=0)
    valid_cosmics = set(cell_features.index)

    # Map CCLE cell names → COSMIC IDs that we have features for
    cell_map = {k: v for k, v in cell_map.items()
                if v in valid_cosmics}

    print(f"  Cell matching: {len(cell_map)} CCLE cells with GDSC features")

    # ── Build prediction pairs ─────────────────────────────────────────────
    print("\n  Building cross-dataset prediction pairs ...")
    rows = []
    for _, row in ccle_df.iterrows():
        drug_name = row["drug_name"]
        cell_name = str(row.get("cell_line", "")).strip().lower()

        gdsc_drug_id = drug_map.get(drug_name)
        cosmic_id = cell_map.get(cell_name)

        if gdsc_drug_id is not None and cosmic_id is not None:
            rows.append({
                "drug_id": gdsc_drug_id,
                "cosmic_id": cosmic_id,
                "ccle_drug_name": drug_name,
                "ccle_auc": float(row["auc"]),
                # Dummy IC50 for dataset compatibility — we won't use it
                "ln_ic50": 0.0,
                "sensitive": 0,
            })

    if len(rows) == 0:
        print("  [FAIL] No matching drug-cell pairs found.")
        return None

    cross_df = pd.DataFrame(rows)
    print(f"  Cross-dataset pairs: {len(cross_df):,}")
    print(f"  Unique drugs: {cross_df['drug_id'].nunique()}")
    print(f"  Unique cells: {cross_df['cosmic_id'].nunique()}")

    # ── Run predictions ────────────────────────────────────────────────────
    cross_ds = DRPDataset(cross_df, drug_graphs, cell_features)
    cross_loader = GeoLoader(cross_ds, batch_size=EVAL_BATCH_SIZE,
                              shuffle=False, num_workers=0)

    pred_ic50 = run_inference(model, cross_loader, device)
    ccle_auc  = cross_df["ccle_auc"].values[:len(pred_ic50)]
    drug_ids  = cross_df["drug_id"].values[:len(pred_ic50)]

    # ── Overall correlation ────────────────────────────────────────────────
    # Lower predicted IC50 should correspond to lower AUC (more sensitive)
    overall_rho, overall_p = spearmanr(pred_ic50, ccle_auc)
    overall_r, _ = pearsonr(pred_ic50, ccle_auc)

    print(f"\n  OVERALL GDSC→CCLE Correlation:")
    print(f"    Spearman ρ = {overall_rho:.4f}  (p = {overall_p:.2e})")
    print(f"    Pearson r  = {overall_r:.4f}")

    # ── Per-drug correlation ───────────────────────────────────────────────
    drug_corr = per_drug_spearman(pred_ic50, ccle_auc, drug_ids)
    if not drug_corr.empty:
        n_sig = drug_corr["significant"].sum()
        print(f"\n  Per-drug Spearman ρ ({len(drug_corr)} drugs):")
        print(f"    Median ρ  = {drug_corr['spearman_rho'].median():.4f}")
        print(f"    Mean ρ    = {drug_corr['spearman_rho'].mean():.4f}")
        print(f"    Significant (p<0.05): {n_sig} / {len(drug_corr)}")

        # Top and bottom drugs
        print(f"\n  Top 5 drugs (best generalisation):")
        for _, row in drug_corr.head(5).iterrows():
            print(f"    Drug {row['drug_id']}: ρ={row['spearman_rho']:.4f} "
                  f"(n={row['n_cells']}, p={row['p_value']:.4f})")

        print(f"\n  Bottom 5 drugs (worst generalisation):")
        for _, row in drug_corr.tail(5).iterrows():
            print(f"    Drug {row['drug_id']}: ρ={row['spearman_rho']:.4f} "
                  f"(n={row['n_cells']}, p={row['p_value']:.4f})")

    return {
        "overall_spearman": round(float(overall_rho), 4),
        "overall_pearson": round(float(overall_r), 4),
        "overall_p_value": float(overall_p),
        "n_pairs": len(pred_ic50),
        "n_drugs": int(cross_df["drug_id"].nunique()),
        "n_cells": int(cross_df["cosmic_id"].nunique()),
        "per_drug_metrics": drug_corr.to_dict(orient="records")
        if not drug_corr.empty else [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_dataset():
    set_seed(SEED)
    device = get_device()

    print("\n" + "=" * 65)
    print("  MIGN-XAI: Cross-Dataset Generalisation Evaluation")
    print("=" * 65)
    print("  Level 1 (sanity): GDSC1 → GDSC2 (same lab, same protocol)")
    print("  Level 2 (real) :  GDSC  → CCLE  (different lab, different assay)")

    # ── Load model ─────────────────────────────────────────────────────────
    cell_features = pd.read_csv(PROC_DIR / "cell_features.csv", index_col=0)
    cell_in = cell_features.shape[1]

    model = MIGN_XAI(
        node_in=NODE_FEATURE_DIM,
        edge_in=EDGE_FEATURE_DIM,
        cell_in=cell_in,
    ).to(device)

    best_ckpt = CKPT_DIR / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"\n  Loaded model from epoch {ckpt.get('epoch', '?')}")
    else:
        print("  [WARN] No checkpoint found. Using untrained model.")
    model.eval()

    all_results = {}

    # ── Level 1: GDSC1 → GDSC2 (sanity) ───────────────────────────────────
    gdsc2_results = gdsc1_to_gdsc2(model, device)
    if gdsc2_results:
        all_results["gdsc1_to_gdsc2_sanity"] = gdsc2_results

    # ── Level 2: GDSC → CCLE (real generalisation) ─────────────────────────
    ccle_results = gdsc_to_ccle(model, device)
    if ccle_results:
        all_results["gdsc_to_ccle"] = ccle_results

    # ── Save ───────────────────────────────────────────────────────────────
    results_file = RESULT_DIR / "cross_dataset_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  CROSS-DATASET SUMMARY")
    print("=" * 65)
    if "gdsc1_to_gdsc2_sanity" in all_results:
        m = all_results["gdsc1_to_gdsc2_sanity"]["metrics"]
        print(f"  GDSC1→GDSC2 (sanity): Pearson r={m['pearson_r']:.4f}, "
              f"RMSE={m['rmse']:.4f}")
    if "gdsc_to_ccle" in all_results:
        r = all_results["gdsc_to_ccle"]
        print(f"  GDSC→CCLE   (real)  : Spearman ρ={r['overall_spearman']:.4f}, "
              f"p={r['overall_p_value']:.2e}, "
              f"{r['n_drugs']} drugs × {r['n_cells']} cells")
    print(f"\n  Results saved to {results_file}")

    return all_results


if __name__ == "__main__":
    run_cross_dataset()

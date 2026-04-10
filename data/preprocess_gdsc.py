"""
data/preprocess_gdsc.py
─────────────────────────────────────────────────────────────────────────────
Step 2: Clean and filter GDSC IC50 data.

What this script does:
  1. Loads GDSC1 and GDSC2 Excel files
  2. Applies quality filters (removes poor curve fits)
  3. Extracts and log-transforms IC50 values
  4. Saves clean DataFrames as CSV for downstream use

Output files (saved to PROC_DIR):
  - gdsc1_clean.csv   : ~150,000 drug-cell pairs with ln(IC50)
  - gdsc2_clean.csv   : ~90,000  drug-cell pairs with ln(IC50)
  - drug_info.csv     : Drug names, IDs, targets, SMILES strings

Run:
    python data/preprocess_gdsc.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PROC_DIR, RMSE_THRESHOLD, MIN_CELL_LINES


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN MAPS (GDSC Excel column names are verbose — map to clean names)
# ─────────────────────────────────────────────────────────────────────────────

GDSC_COLS = {
    "DATASET"          : "dataset",
    "DRUG_ID"          : "drug_id",
    "DRUG_NAME"        : "drug_name",
    "PUTATIVE_TARGET"  : "target",
    "PATHWAY_NAME"     : "pathway",
    "COSMIC_ID"        : "cosmic_id",
    "CELL_LINE_NAME"   : "cell_name",
    "TCGA_DESC"        : "cancer_type",
    "LN_IC50"          : "ln_ic50",
    "AUC"              : "auc",
    "RMSE"             : "rmse",
    "Z_SCORE"          : "z_score",
}


def load_gdsc(filepath: Path, dataset_name: str) -> pd.DataFrame:
    """
    Load one GDSC Excel file, rename columns, and return raw DataFrame.

    Parameters
    ----------
    filepath    : Path to .xlsx file
    dataset_name: "GDSC1" or "GDSC2"
    """
    print(f"  Loading {filepath.name} ...")
    if not filepath.exists():
        raise FileNotFoundError(
            f"\n[ERROR] File not found: {filepath}\n"
            f"Please run: python data/download_data.py\n"
        )

    df = pd.read_excel(filepath, engine="openpyxl")
    print(f"  Raw shape: {df.shape}")

    # Keep only columns we care about (gracefully handle missing ones)
    keep = {k: v for k, v in GDSC_COLS.items() if k in df.columns}
    df = df[list(keep.keys())].rename(columns=keep)

    # Ensure COSMIC ID is integer
    df["cosmic_id"] = pd.to_numeric(df["cosmic_id"], errors="coerce")
    df = df.dropna(subset=["cosmic_id"])
    df["cosmic_id"] = df["cosmic_id"].astype(int)

    print(f"  After column selection: {df.shape}")
    return df


def quality_filter(df: pd.DataFrame, rmse_threshold: float) -> pd.DataFrame:
    """
    Remove low-quality drug-cell line screenings.

    Filters applied:
      1. Remove pairs where dose-response curve RMSE > threshold
         (poor pharmacological curve fit = unreliable IC50)
      2. Remove rows with missing ln(IC50)
      3. Remove drugs screened against fewer than MIN_CELL_LINES cell lines
         (insufficient data to learn from)
    """
    n_start = len(df)

    # Filter 1: Poor curve fit
    if "rmse" in df.columns:
        df = df[df["rmse"] < rmse_threshold]
        print(f"  After RMSE filter (< {rmse_threshold}): {len(df)} rows "
              f"(removed {n_start - len(df)})")

    # Filter 2: Missing IC50
    df = df.dropna(subset=["ln_ic50"])
    print(f"  After dropping NaN IC50: {len(df)} rows")

    # Filter 3: Drugs with too few cell lines
    drug_counts = df.groupby("drug_id")["cosmic_id"].nunique()
    valid_drugs = drug_counts[drug_counts >= MIN_CELL_LINES].index
    df = df[df["drug_id"].isin(valid_drugs)]
    print(f"  After min cell line filter (>= {MIN_CELL_LINES}): {len(df)} rows")

    return df.reset_index(drop=True)


def compute_binary_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary sensitivity label.

    For each drug, compute the drug-specific median ln(IC50).
    Cell lines below the median are labelled as 'sensitive' (1),
    above as 'resistant' (0). This drug-specific threshold avoids
    treating all drugs on the same scale.
    """
    drug_medians = df.groupby("drug_id")["ln_ic50"].median()
    df["median_ic50"] = df["drug_id"].map(drug_medians)
    df["sensitive"] = (df["ln_ic50"] < df["median_ic50"]).astype(int)
    df = df.drop(columns=["median_ic50"])
    print(f"  Sensitive pairs: {df['sensitive'].sum()} / {len(df)} "
          f"({df['sensitive'].mean()*100:.1f}%)")
    return df


def extract_drug_info(df: pd.DataFrame, compounds_file: Path) -> pd.DataFrame:
    """
    Build a drug metadata table with SMILES strings.

    The GDSC compounds CSV contains SMILES for each drug. We join it
    to the drug_id/drug_name information from the IC50 table.
    """
    # Get unique drug info from IC50 table
    drug_df = (df[["drug_id", "drug_name", "target", "pathway"]]
               .drop_duplicates("drug_id")
               .reset_index(drop=True))

    # Try to join SMILES from compounds file
    if compounds_file.exists():
        compounds = pd.read_csv(compounds_file)
        # Column names vary by GDSC release — try common ones
        id_col = next((c for c in compounds.columns
                       if "drug_id" in c.lower() or "id" in c.lower()), None)
        smiles_col = next((c for c in compounds.columns
                           if "smiles" in c.lower()), None)

        if id_col and smiles_col:
            smiles_map = compounds.set_index(id_col)[smiles_col].to_dict()
            drug_df["smiles"] = drug_df["drug_id"].map(smiles_map)
            n_with_smiles = drug_df["smiles"].notna().sum()
            print(f"  Drugs with SMILES: {n_with_smiles} / {len(drug_df)}")
        else:
            print("  [WARN] Could not find ID or SMILES columns in compounds file")
            drug_df["smiles"] = None
    else:
        print(f"  [WARN] Compounds file not found: {compounds_file}")
        print("  SMILES will be fetched from PubChem in drug_graph.py")
        drug_df["smiles"] = None

    return drug_df


def describe_dataset(df: pd.DataFrame, name: str):
    """Print descriptive statistics for the cleaned dataset."""
    print(f"\n  ── {name} Summary ──")
    print(f"  Drug-cell pairs  : {len(df):,}")
    print(f"  Unique drugs     : {df['drug_id'].nunique()}")
    print(f"  Unique cell lines: {df['cosmic_id'].nunique()}")
    print(f"  Cancer types     : {df['cancer_type'].nunique() if 'cancer_type' in df.columns else 'N/A'}")
    print(f"  ln(IC50) range   : [{df['ln_ic50'].min():.2f}, {df['ln_ic50'].max():.2f}]")
    print(f"  ln(IC50) mean±std: {df['ln_ic50'].mean():.2f} ± {df['ln_ic50'].std():.2f}")


def preprocess_gdsc():
    print("\n" + "="*65)
    print("  Step 2: GDSC Preprocessing")
    print("="*65 + "\n")

    compounds_file = DATA_DIR / "gdsc_compounds.csv"

    # ── Process GDSC1 ──────────────────────────────────────────────────────
    print("── Processing GDSC1 ──")
    gdsc1_raw = load_gdsc(DATA_DIR / "GDSC1_fitted_dose_response.xlsx", "GDSC1")
    gdsc1     = quality_filter(gdsc1_raw, RMSE_THRESHOLD)
    gdsc1     = compute_binary_label(gdsc1)
    describe_dataset(gdsc1, "GDSC1")

    # ── Process GDSC2 ──────────────────────────────────────────────────────
    print("\n── Processing GDSC2 ──")
    gdsc2_raw = load_gdsc(DATA_DIR / "GDSC2_fitted_dose_response.xlsx", "GDSC2")
    gdsc2     = quality_filter(gdsc2_raw, RMSE_THRESHOLD)
    gdsc2     = compute_binary_label(gdsc2)
    describe_dataset(gdsc2, "GDSC2")

    # ── Build drug info table from GDSC1 (larger drug panel) ──────────────
    print("\n── Building drug metadata table ──")
    drug_info = extract_drug_info(gdsc1, compounds_file)

    # ── Save ──────────────────────────────────────────────────────────────
    gdsc1.to_csv(PROC_DIR / "gdsc1_clean.csv", index=False)
    gdsc2.to_csv(PROC_DIR / "gdsc2_clean.csv", index=False)
    drug_info.to_csv(PROC_DIR / "drug_info.csv", index=False)

    print(f"\n  Saved:")
    print(f"  → {PROC_DIR}/gdsc1_clean.csv   ({len(gdsc1):,} rows)")
    print(f"  → {PROC_DIR}/gdsc2_clean.csv   ({len(gdsc2):,} rows)")
    print(f"  → {PROC_DIR}/drug_info.csv     ({len(drug_info)} drugs)")
    print("\n  Next step: python data/preprocess_omics.py\n")


if __name__ == "__main__":
    preprocess_gdsc()
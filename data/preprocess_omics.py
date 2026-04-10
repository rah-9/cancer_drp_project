"""
data/preprocess_omics.py
─────────────────────────────────────────────────────────────────────────────
Step 3: Process gene expression and somatic mutation data from CCLE/DepMap.

What this script does:
  1. Loads CCLE gene expression (RNA-seq, already log2 TPM+1 normalised)
  2. Filters to LINCS L1000 landmark genes (978 genes)
  3. Z-score normalises each gene across all cell lines
  4. Aggregates genes into 50 MSigDB Hallmark pathway activity scores
  5. Loads somatic mutation data and creates a binary mutation matrix
  6. Joins everything via COSMIC ID to align with GDSC
  7. Saves final feature matrices as CSV and numpy arrays

Output files (saved to PROC_DIR):
  - cell_features.csv    : (n_cells × 1026+) full feature matrix
  - cell_gex.csv         : (n_cells × 978) gene expression only
  - cell_pathways.csv    : (n_cells × n_pathways) pathway activity scores
  - cell_mutations.csv   : (n_cells × n_genes) binary mutation matrix
  - feature_names.json   : ordered list of all feature names

Run:
    python data/preprocess_omics.py
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (DATA_DIR, PROC_DIR, N_LANDMARK_GENES, N_PATHWAY_FEATURES,
                    MIN_PATHWAY_GENES, USE_COSMIC_GENES, MAX_MUT_GENES)


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK GENE SETS (used if internet downloads fail)
# ─────────────────────────────────────────────────────────────────────────────

# A curated subset of well-known oncogenes / tumour suppressors as fallback
FALLBACK_COSMIC_GENES = [
    "EGFR", "KRAS", "BRAF", "PIK3CA", "TP53", "BRCA1", "BRCA2", "PTEN",
    "RB1", "APC", "VHL", "CDKN2A", "CDK4", "CDK6", "MYC", "ERBB2",
    "ALK", "RET", "MET", "FGFR1", "FGFR2", "FGFR3", "NRAS", "HRAS",
    "MAP2K1", "MAP2K2", "AKT1", "AKT2", "AKT3", "MTOR", "TSC1", "TSC2",
    "STK11", "NF1", "NF2", "ATM", "CHEK2", "PALB2", "RAD51", "FANCD2",
    "MDM2", "MDM4", "CCND1", "CCND2", "CCND3", "CCNE1", "CCNE2",
    "BCL2", "BCL6", "MCL1", "BAX", "BAD", "BID", "CASP3", "CASP9",
    "JAK1", "JAK2", "STAT3", "STAT5A", "STAT5B", "SRC", "ABL1", "KIT",
    "PDGFRA", "PDGFRB", "FLT3", "CSF1R", "SMO", "PTCH1", "GLI1",
    "NOTCH1", "NOTCH2", "HES1", "DLL3", "WNT5A", "CTNNB1", "AXIN1",
    "RNF43", "RSPO2", "RSPO3", "IDH1", "IDH2", "TET2", "DNMT3A",
    "EZH2", "KDM6A", "ARID1A", "SMARCA4", "SMARCB1",
]


def load_landmark_genes() -> list:
    """
    Load the 978 LINCS L1000 landmark gene symbols.
    Falls back to reading from a bundled list if file not found.
    """
    landmark_file = DATA_DIR / "l1000_landmark_genes.txt"

    if landmark_file.exists():
        try:
            df = pd.read_csv(landmark_file, sep="\t")
            # Common column name variants
            for col in ["pr_gene_symbol", "gene_symbol", "Symbol", "SYMBOL"]:
                if col in df.columns:
                    genes = df[col].dropna().tolist()
                    print(f"  Loaded {len(genes)} landmark genes from file.")
                    return genes
        except Exception as e:
            print(f"  [WARN] Could not parse landmark file: {e}")

    # Try fetching from GitHub
    try:
        import requests
        url = ("https://raw.githubusercontent.com/cmap/cmapM/master/"
               "resources/lm_gene_info_gs_n978x22.txt")
        resp = requests.get(url, timeout=30)
        lines = resp.text.strip().split("\n")
        # First line is header; gene symbol is first column
        genes = [l.split("\t")[0] for l in lines[1:] if l.strip()]
        print(f"  Fetched {len(genes)} landmark genes from GitHub.")
        return genes
    except Exception:
        pass

    # Hard-coded fallback (300 well-known genes — not full L1000)
    print("  [WARN] Using fallback gene list (not full L1000). "
          "Download l1000_landmark_genes.txt for best results.")
    return FALLBACK_COSMIC_GENES[:300]


def load_expression(gex_file: Path, model_file: Path) -> pd.DataFrame:
    """
    Load CCLE gene expression and add COSMIC ID index.

    Parameters
    ----------
    gex_file   : OmicsExpressionProteinCodingGenesTPMLogp1.csv
    model_file : Model.csv (contains ModelID → COSMIC_ID mapping)

    Returns
    -------
    DataFrame indexed by COSMIC_ID, columns = gene symbols
    """
    print(f"  Loading gene expression from {gex_file.name} ...")
    if not gex_file.exists():
        raise FileNotFoundError(
            f"\n[ERROR] {gex_file} not found.\n"
            "Please download from https://depmap.org/portal/download/all/\n"
        )

    gex = pd.read_csv(gex_file, index_col=0)
    # Column names are formatted as "SYMBOL (ENTREZID)" — strip the ID
    gex.columns = [c.split(" (")[0] if " (" in c else c for c in gex.columns]
    print(f"  Expression shape: {gex.shape}  (cells × genes)")

    # Load model info to get COSMIC IDs
    print(f"  Loading cell line metadata from {model_file.name} ...")
    model = pd.read_csv(model_file)

    # Find COSMIC ID column (varies by DepMap release)
    cosmic_col = next((c for c in model.columns
                       if "cosmic" in c.lower()), None)
    id_col = "ModelID"  # DepMap cell line identifier

    if cosmic_col is None:
        raise ValueError("Could not find COSMIC ID column in Model.csv. "
                         "Check the column names in your downloaded file.")

    model = model[[id_col, cosmic_col]].dropna()
    model[cosmic_col] = pd.to_numeric(model[cosmic_col], errors="coerce")
    model = model.dropna().set_index(id_col)
    model[cosmic_col] = model[cosmic_col].astype(int)

    # Join expression with COSMIC IDs
    gex_joined = gex.join(model[cosmic_col])
    gex_joined = gex_joined.dropna(subset=[cosmic_col])
    gex_joined[cosmic_col] = gex_joined[cosmic_col].astype(int)
    gex_indexed = gex_joined.set_index(cosmic_col)
    gex_indexed.index.name = "cosmic_id"
    print(f"  After joining with COSMIC IDs: {gex_indexed.shape}")
    return gex_indexed


def filter_landmark_genes(gex: pd.DataFrame, landmark_genes: list) -> pd.DataFrame:
    """
    Filter expression to L1000 landmark genes.
    Uses intersection so missing genes are silently skipped.
    """
    available = [g for g in landmark_genes if g in gex.columns]
    pct = 100 * len(available) / len(landmark_genes)
    print(f"  Landmark genes available: {len(available)} / {len(landmark_genes)} ({pct:.1f}%)")
    return gex[available]


def zscore_normalize(gex: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalise each gene across all cell lines.

    Why: Ensures high-expression genes (e.g. housekeeping genes like GAPDH)
    don't dominate early network layers simply due to scale. After Z-scoring,
    every gene contributes equally to initial weight updates.
    """
    scaler = StandardScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(gex),
        index=gex.index,
        columns=gex.columns
    )
    return scaled, scaler


def compute_pathway_features(gex: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate gene expression into pathway activity scores using
    MSigDB Hallmark gene sets (50 pathways, downloaded via gseapy).

    Each pathway score = mean Z-score of all member genes present in gex.
    This adds biological context (pathway-level signal) beyond individual genes.
    """
    try:
        import gseapy
        hallmarks = gseapy.get_library("MSigDB_Hallmark_2020")
        print(f"  Loaded {len(hallmarks)} Hallmark pathways from MSigDB")
    except Exception as e:
        print(f"  [WARN] Could not load MSigDB hallmarks: {e}")
        print("  Skipping pathway features.")
        return pd.DataFrame(index=gex.index)

    pathway_scores = {}
    skipped = 0
    for pathway_name, gene_list in hallmarks.items():
        # Find genes in this pathway that exist in our expression matrix
        overlap = [g for g in gene_list if g in gex.columns]
        if len(overlap) < MIN_PATHWAY_GENES:
            skipped += 1
            continue
        # Pathway activity = mean expression of member genes
        pathway_scores[pathway_name] = gex[overlap].mean(axis=1)

    pathway_df = pd.DataFrame(pathway_scores)
    print(f"  Pathway features: {len(pathway_df.columns)} pathways "
          f"(skipped {skipped} with < {MIN_PATHWAY_GENES} genes)")
    return pathway_df


def load_mutations(mut_file: Path, model_file: Path,
                   cosmic_genes: list = None) -> pd.DataFrame:
    """
    Load somatic mutation data and create binary cell × gene matrix.

    Entry (i,j) = 1 if cell line i has a non-synonymous mutation in gene j.
    Restricted to cancer-relevant genes to reduce noise.
    """
    print(f"  Loading mutations from {mut_file.name} ...")
    if not mut_file.exists():
        print(f"  [WARN] Mutation file not found: {mut_file}")
        print("  Mutation features will be omitted.")
        return pd.DataFrame()

    mut_raw = pd.read_csv(mut_file, low_memory=False)
    print(f"  Raw mutation records: {len(mut_raw):,}")

    # Identify relevant columns (column names vary by DepMap release)
    model_col = next((c for c in mut_raw.columns if "ModelID" in c or "model_id" in c.lower()), None)
    gene_col  = next((c for c in mut_raw.columns if "HugoSymbol" in c or "gene_name" in c.lower()), None)
    type_col  = next((c for c in mut_raw.columns if "VariantType" in c or "variant_type" in c.lower()), None)

    if not all([model_col, gene_col]):
        print("  [WARN] Could not identify required columns in mutation file.")
        return pd.DataFrame()

    # Keep only protein-altering mutations
    DAMAGING_TYPES = {
        "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
        "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
        "Translation_Start_Site", "Nonstop_Mutation",
    }
    if type_col:
        mut_filtered = mut_raw[mut_raw[type_col].isin(DAMAGING_TYPES)]
    else:
        mut_filtered = mut_raw  # use all if type column missing
    print(f"  After variant type filter: {len(mut_filtered):,} records")

    # Restrict to cancer-relevant genes
    if cosmic_genes is not None:
        mut_filtered = mut_filtered[mut_filtered[gene_col].isin(cosmic_genes)]
        print(f"  After COSMIC gene filter: {len(mut_filtered):,} records")

    # Pivot to binary matrix: rows=ModelID, cols=genes, values=1/0
    mut_matrix = mut_filtered.pivot_table(
        index=model_col,
        columns=gene_col,
        values=type_col if type_col else gene_col,
        aggfunc=lambda x: 1,
        fill_value=0
    )
    print(f"  Mutation matrix (before COSMIC join): {mut_matrix.shape}")

    # Map ModelID → COSMIC_ID
    model = pd.read_csv(model_file)
    cosmic_col = next((c for c in model.columns if "cosmic" in c.lower()), None)
    if cosmic_col is None:
        print("  [WARN] No COSMIC ID column in Model.csv. Cannot align mutations.")
        return pd.DataFrame()

    model_map = (model[["ModelID", cosmic_col]]
                 .dropna()
                 .assign(**{cosmic_col: lambda df: df[cosmic_col].astype(int)})
                 .set_index("ModelID")[cosmic_col]
                 .to_dict())

    mut_matrix.index = mut_matrix.index.map(model_map)
    mut_matrix = mut_matrix[mut_matrix.index.notna()]
    mut_matrix.index = mut_matrix.index.astype(int)
    mut_matrix.index.name = "cosmic_id"
    mut_matrix = mut_matrix[~mut_matrix.index.duplicated(keep="first")]

    print(f"  Final mutation matrix: {mut_matrix.shape}  (cells × mutated genes)")
    return mut_matrix


def preprocess_omics():
    print("\n" + "="*65)
    print("  Step 3: Omics Preprocessing")
    print("="*65 + "\n")

    gex_file   = DATA_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
    mut_file   = DATA_DIR / "OmicsSomaticMutations.csv"
    model_file = DATA_DIR / "Model.csv"

    # ── Load landmark genes ────────────────────────────────────────────────
    print("── Loading Landmark Genes ──")
    landmark_genes = load_landmark_genes()

    # ── Gene Expression ────────────────────────────────────────────────────
    print("\n── Gene Expression Preprocessing ──")
    gex_raw       = load_expression(gex_file, model_file)
    gex_landmark  = filter_landmark_genes(gex_raw, landmark_genes)
    gex_scaled, _ = zscore_normalize(gex_landmark)
    print(f"  Final GEx matrix: {gex_scaled.shape}")

    # ── Pathway Features ───────────────────────────────────────────────────
    print("\n── Pathway Activity Scores ──")
    pathway_df = compute_pathway_features(gex_scaled)

    # ── Somatic Mutations ──────────────────────────────────────────────────
    print("\n── Somatic Mutation Matrix ──")
    cosmic_genes = FALLBACK_COSMIC_GENES if USE_COSMIC_GENES else None
    mut_df = load_mutations(mut_file, model_file, cosmic_genes)

    # Truncate mutation features if too many
    if not mut_df.empty and len(mut_df.columns) > MAX_MUT_GENES:
        # Keep most commonly mutated genes
        mut_freq = mut_df.sum(axis=0).sort_values(ascending=False)
        top_genes = mut_freq.head(MAX_MUT_GENES).index
        mut_df = mut_df[top_genes]
        print(f"  Truncated to top {MAX_MUT_GENES} most frequently mutated genes")

    # ── Align all matrices to common cell lines ────────────────────────────
    print("\n── Aligning Cell Lines ──")
    common_cells = set(gex_scaled.index)
    if not pathway_df.empty:
        common_cells &= set(pathway_df.index)
    if not mut_df.empty:
        common_cells &= set(mut_df.index)

    common_cells = sorted(common_cells)
    print(f"  Common cell lines across all modalities: {len(common_cells)}")

    gex_scaled = gex_scaled.loc[common_cells]
    if not pathway_df.empty:
        pathway_df = pathway_df.loc[common_cells]
    if not mut_df.empty:
        mut_df = mut_df.loc[common_cells]

    # ── Build combined feature matrix ──────────────────────────────────────
    print("\n── Building Combined Feature Matrix ──")
    parts = [gex_scaled]
    feature_groups = {"gex": list(gex_scaled.columns)}

    if not pathway_df.empty:
        parts.append(pathway_df)
        feature_groups["pathway"] = list(pathway_df.columns)

    if not mut_df.empty:
        parts.append(mut_df)
        feature_groups["mutation"] = list(mut_df.columns)

    cell_features = pd.concat(parts, axis=1)
    print(f"  Combined feature matrix: {cell_features.shape}")
    print(f"  Breakdown: GEx={len(feature_groups.get('gex',[]))}, "
          f"Pathways={len(feature_groups.get('pathway',[]))}, "
          f"Mutations={len(feature_groups.get('mutation',[]))}")

    # ── Save ──────────────────────────────────────────────────────────────
    cell_features.to_csv(PROC_DIR / "cell_features.csv")
    gex_scaled.to_csv(PROC_DIR / "cell_gex.csv")

    if not pathway_df.empty:
        pathway_df.to_csv(PROC_DIR / "cell_pathways.csv")
    if not mut_df.empty:
        mut_df.to_csv(PROC_DIR / "cell_mutations.csv")

    # Save feature names with group info for later XAI
    with open(PROC_DIR / "feature_names.json", "w") as f:
        json.dump(feature_groups, f, indent=2)

    print(f"\n  Saved to {PROC_DIR}/")
    print("  Next step: python data/drug_graph.py\n")


if __name__ == "__main__":
    preprocess_omics()
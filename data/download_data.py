"""
data/download_data.py
─────────────────────────────────────────────────────────────────────────────
Step 1: Download all required datasets automatically.

Run this first:
    python data/download_data.py

What it downloads:
  - GDSC1 drug sensitivity Excel (IC50 values)
  - GDSC2 drug sensitivity Excel (second drug panel)
  - GDSC screened compounds CSV (contains SMILES strings for drugs)
  - Instructions for manual DepMap downloads (their CDN blocks scripted access)
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, GDSC1_URL, GDSC2_URL, GDSC_COMPOUNDS_URL


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """
    Download a file from a URL with a progress bar.
    Returns True on success, False on failure.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"  [SKIP] {dest_path.name} already exists.")
        return True

    print(f"  [DOWNLOAD] {description or dest_path.name}")
    print(f"             URL: {url}")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_path.name
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

        print(f"  [OK] Saved to {dest_path}")
        return True

    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def download_all():
    print("\n" + "=" * 65)
    print("  MIGN-XAI: Downloading Datasets")
    print("=" * 65 + "\n")

    # ── GDSC datasets ─────────────────────────────────────────────────────
    print("── GDSC (Genomics of Drug Sensitivity in Cancer) ──")
    print("   Source: cancerrxgene.org | No registration required\n")

    download_file(GDSC1_URL,
                  DATA_DIR / "GDSC1_fitted_dose_response.xlsx",
                  "GDSC1 IC50 values (~200 drugs × ~1000 cell lines)")

    download_file(GDSC2_URL,
                  DATA_DIR / "GDSC2_fitted_dose_response.xlsx",
                  "GDSC2 IC50 values (second drug panel)")

    download_file(GDSC_COMPOUNDS_URL,
                  DATA_DIR / "gdsc_compounds.csv",
                  "GDSC drug list with SMILES strings")

    # ── DepMap / CCLE (manual instructions) ───────────────────────────────
    print("\n── DepMap / CCLE (Manual Download Required) ──")
    print("   DepMap blocks automated downloads. Please do this manually:\n")
    print("   1. Go to: https://depmap.org/portal/download/all/")
    print("   2. Search for and download these files:")
    print("      a) OmicsExpressionProteinCodingGenesTPMLogp1.csv")
    print("         → Gene expression (RNA-seq, log2 TPM+1 normalised)")
    print("      b) OmicsSomaticMutations.csv")
    print("         → Somatic mutation calls per gene per cell line")
    print("      c) Model.csv")
    print("         → Cell line metadata (contains COSMIC ID for GDSC joining)")
    print(f"   3. Save all three files to: {DATA_DIR}/\n")

    # ── L1000 Landmark Genes ───────────────────────────────────────────────
    print("── L1000 Landmark Genes ──")
    print("   978 genes that capture 80% of transcriptomic variance.\n")

    landmark_url = (
        "https://raw.githubusercontent.com/cmap/cmapM/master/"
        "resources/lm_gene_info_gs_n978x22.txt"
    )
    success = download_file(
        landmark_url,
        DATA_DIR / "l1000_landmark_genes.txt",
        "LINCS L1000 landmark gene list (978 genes)"
    )
    if not success:
        # Create a note file if download fails
        note = DATA_DIR / "l1000_NOTE.txt"
        note.write_text(
            "Download landmark genes manually from:\n"
            "https://lincsproject.org/LINCS/tools/workflows/find-the-l1000-landmark-genes\n"
            "Save as: l1000_landmark_genes.txt\n"
            "Format: tab-separated with column 'pr_gene_symbol'"
        )
        print(f"  [NOTE] Instructions saved to {note}")

    # ── COSMIC Cancer Gene Census ──────────────────────────────────────────
    print("\n── COSMIC Cancer Gene Census (for mutation filtering) ──")
    print("   Free but requires registration at cancer.sanger.ac.uk")
    print("   OR use the built-in fallback list in preprocess_omics.py\n")

    print("=" * 65)
    print("  DOWNLOAD COMPLETE. Next step:")
    print("  python data/preprocess_gdsc.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    download_all()
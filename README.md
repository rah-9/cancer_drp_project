# MIGN-XAI: Multimodal Interaction Graph-Omics Network with Explainability

## Cancer Drug Response Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![PyG 2.3+](https://img.shields.io/badge/PyG-2.3%2B-green.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A research-grade deep learning pipeline for predicting cancer drug sensitivity (ln(IC50)) from **molecular drug graphs** and **multi-omics cell line profiles**, with built-in explainability for biological discovery.

---

## Table of Contents

- [Motivation](#motivation)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Pipeline](#data-pipeline)
- [Training](#training)
- [Evaluation](#evaluation)
- [Explainability (XAI)](#explainability-xai)
- [Experiments](#experiments)
- [Model Details](#model-details)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

---

## Motivation

**Why this matters**: A cancer patient's tumour has a unique genomic profile. Clinicians must choose from hundreds of approved drugs, but laboratory screening of every drug on every patient's tumour is infeasible. Computational drug response prediction can narrow down candidates *in silico*, enabling precision oncology.

**Why existing methods fall short**:

| Limitation | How MIGN-XAI Addresses It |
|-----------|--------------------------|
| Drug features are flat fingerprints (ECFP) | GIN encoder learns directly from molecular graphs with edge features |  
| Gene expression treated as flat vector | Hierarchical cell encoder with pathway-level aggregation |
| Drug and cell features fused by simple concatenation | Cross-attention fusion enables drug-specific feature routing |
| Models are black boxes | SHAP attribution + attention visualisation + biological validation |
| Evaluated on random splits (data leakage) | Cell-blind and drug-blind split strategies with leak verification |
| No cross-dataset validation | GDSC → CCLE generalisation test (different lab, different assay) |

---

## Architecture

```
                              MIGN-XAI Architecture
 ┌──────────────────────┐
 │   SMILES String      │
 │  "CC(=O)Oc1ccccc1"   │
 └──────────┬───────────┘
            │  RDKit
            ▼
 ┌──────────────────────┐        ┌─────────────────────────────┐
 │  Molecular Graph     │        │  Cell Line Omics Vector     │
 │  (atoms + bonds)     │        │  [GEx | Pathways | Mutations]│
 │  45-dim node feats   │        │  ~1026 features             │
 │  10-dim edge feats   │        └──────────┬──────────────────┘
 └──────────┬───────────┘                   │
            │                               │
            ▼                               ▼
 ┌──────────────────────┐        ┌──────────────────────────────┐
 │  GIN Encoder         │        │  Cell Encoder (Hierarchical) │
 │  3× GINEConv layers  │        │  4-layer MLP with BatchNorm  │
 │  + edge features     │        │  → z_cell (128-d)            │
 │  + global pooling    │        │  → cell_seq (4 × 128-d)      │
 │  → z_drug (256-d)    │        │    (per-layer abstractions)  │
 └──────────┬───────────┘        └───────────┬─────────────────┘
            │                                │
            │         ┌──────────────────────┤
            │         │                      │
            ▼         ▼                      │
 ┌──────────────────────────────┐            │
 │  Cross-Attention Fusion      │            │
 │  Drug queries cell sequence  │            │
 │  4 heads × 4 layers          │            │
 │  → z_fused (128-d)           │            │
 └──────────┬───────────────────┘            │
            │                                │
            ▼                                ▼
 ┌───────────────────────────────────────────────┐
 │  Residual Concatenation                       │
 │  z_final = [z_fused ‖ z_drug ‖ z_cell]       │
 │          = [128    ‖  256   ‖  128 ] = 512-d  │
 └──────────────────────┬────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────┐
 │  Prediction Head             │
 │  512 → 256 → 128 → 1        │
 │  → Predicted ln(IC50)        │
 └──────────────────────────────┘
```

**Key design decisions**:

1. **GIN with edge features (GINEConv)**: Unlike GCN/GAT, GIN achieves maximal expressive power among message-passing GNNs (Xu et al., 2019). Edge features encode bond type, conjugation, ring membership — critical for drug activity.

2. **Hierarchical cell encoder**: The 4-layer MLP produces intermediate representations at each layer. Layer 1 captures raw gene signals, Layer 4 captures pathway-level abstractions. Cross-attention can attend to *any* of these levels depending on the drug.

3. **Cross-attention fusion**: The drug embedding *queries* the cell encoder's multi-level output. This means different drugs "look at" different biological levels — a targeted therapy might attend to specific gene features (Layer 1), while a broad cytotoxic drug attends to pathway-level features (Layer 4).

4. **Residual concatenation**: `z_final = [z_fused, z_drug, z_cell]` ensures the prediction head has access to (a) the interaction signal, (b) the raw drug identity, and (c) the raw cell profile. This consistently outperforms using `z_fused` alone.

---

## Project Structure

```
cancer_drp_project/
├── config.py                        # All hyperparameters, paths, URLs
├── requirements.txt                 # Python dependencies
├── test_integration.py              # 7 automated integration tests
│
├── data/
│   ├── __init__.py
│   ├── download_data.py             # Auto-download GDSC + DepMap data
│   ├── preprocess_gdsc.py           # Clean IC50 data, quality filter
│   ├── preprocess_omics.py          # Gene expression + mutations + pathways
│   ├── drug_graph.py                # SMILES → PyG molecular graphs
│   └── dataset.py                   # PyG Dataset + split strategies + leakage checks
│
├── models/
│   ├── __init__.py
│   ├── gin_encoder.py               # GIN drug encoder (3 GINEConv layers)
│   ├── cell_encoder.py              # Hierarchical MLP cell encoder
│   ├── cross_attention.py           # Multi-head cross-attention fusion
│   └── mign_xai.py                  # Full MIGN-XAI model + MC Dropout
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                   # Pearson, Spearman, RMSE, AUROC, Wilcoxon
│   ├── lds.py                       # Label Distribution Smoothing
│   ├── helpers.py                   # Seeding, logging, checkpoints
│   └── metrics_and_helpers.py       # Unified re-export module
│
├── experiments/
│   ├── __init__.py
│   ├── train.py                     # Main training (Huber + LDS + early stop)
│   ├── evaluate.py                  # Multi-split eval + failure analysis + MC Dropout
│   ├── ablation.py                  # 7-variant ablation study
│   ├── cross_dataset.py             # GDSC → CCLE generalisation
│   └── baselines.py                 # SVM, RF, GraphDRP + Wilcoxon significance
│
├── results/
│   ├── __init__.py
│   └── xai/
│       ├── __init__.py
│       ├── shap_analysis.py         # KernelSHAP + stability checks
│       ├── attention_viz.py         # Cross-attention heatmaps
│       └── biological_val.py        # Precision@K + Fisher's enrichment
│
└── notebooks/
    └── full_pipeline.ipynb          # End-to-end Google Colab notebook
```

---

## Installation

### Option 1: Local (GPU recommended)

```bash
# Clone the repository
git clone https://github.com/rah-9/cancer_drp_project.git
cd cancer_drp_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install PyTorch (CUDA 11.8 example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyG
pip install torch-geometric

# Install remaining dependencies
pip install -r requirements.txt
```

### Option 2: Google Colab

Open `notebooks/full_pipeline.ipynb` in Colab — it handles all installation automatically.

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.10-3.12 |
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB | 8+ GB (RTX 3060+) |
| Disk | 5 GB | 10 GB |
| Training time | ~2h (GPU) | ~30min (A100) |

---

## Data Pipeline

The data pipeline runs in 5 sequential steps:

### Step 1: Download Data
```bash
python data/download_data.py
```

Downloads from public sources:
- **GDSC1/GDSC2**: Drug sensitivity (IC50) data from the Genomics of Drug Sensitivity in Cancer project
- **DepMap/CCLE**: Gene expression (TPM), somatic mutations from the Cancer Dependency Map portal

> **Note**: DepMap data requires manual download from [depmap.org](https://depmap.org/portal/). Place files in `data/raw/`.

### Step 2: Preprocess GDSC
```bash
python data/preprocess_gdsc.py
```

- Loads raw GDSC dose-response data
- Filters by curve fit quality (RMSE < 0.3)
- Removes drugs with < 5 screened cell lines
- Computes ln(IC50) as the continuous target
- Generates binary sensitivity labels (below median = sensitive)
- Outputs: `data/processed/gdsc1_clean.csv`

### Step 3: Preprocess Omics
```bash
python data/preprocess_omics.py
```

Produces a unified cell line feature vector (~ 1026 dimensions):

| Feature Type | Count | Source | Processing |
|-------------|-------|--------|-----------|
| L1000 landmark genes | 978 | DepMap expression | Z-score normalised |
| MSigDB Hallmark pathways | ~48 | Aggregated from L1000 | Mean expression per pathway |
| Binary mutations | variable | DepMap somatic mutations | 1 = mutated, 0 = wildtype |

- Outputs: `data/processed/cell_features.csv`, `data/processed/feature_names.json`

### Step 4: Build Molecular Graphs
```bash
python data/drug_graph.py
```

Converts drug SMILES strings into PyTorch Geometric graph objects using RDKit:

| Feature | Dimension | Encodes |
|---------|-----------|---------|
| Node (atom) features | 45 | Element type (one-hot), degree, formal charge, hybridisation, aromaticity, H count, ring membership |
| Edge (bond) features | 10 | Bond type (one-hot), conjugation, ring membership, stereo |

- Outputs: `data/processed/drug_graphs.pt`

### Step 5: Create Dataset Splits
```python
from data.dataset import load_all_data
splitter = load_all_data("gdsc1")
train_ds, val_ds, test_ds = splitter.get_split("cell_blind")
```

Three split strategies with **automatic data leakage verification**:

| Strategy | What's held out | Measures | Leakage check |
|----------|----------------|----------|---------------|
| `random` | Random pairs | Interpolation (easy, inflated) | Reports overlap % (expected) |
| `cell_blind` | Entire cell lines | Can the model generalise to unseen tumours? | `AssertionError` if any cell leaked |
| `drug_blind` | Entire drugs | Can the model predict for novel compounds? | `AssertionError` if any drug leaked |

> **Important**: Random splitting leads to data leakage — the same cell line appears in both train and test. Always use `cell_blind` or `drug_blind` for rigorous evaluation.

---

## Training

### Basic Training
```bash
python experiments/train.py --split cell_blind --epochs 200 --lr 3e-4 --lds
```

### Training Details

| Component | Choice | Why |
|-----------|--------|-----|
| **Loss** | Huber Loss (delta=1.0) | Robust to IC50 outliers (some drugs have extreme ln(IC50) values) |
| **Optimizer** | AdamW (lr=3e-4, wd=1e-4) | Weight decay prevents overfitting on small cell line counts |
| **Scheduler** | Cosine Annealing (T=100, eta_min=1e-6) | Smooth LR decay, avoids plateau oscillation |
| **Regularisation** | L1 on cell encoder Layer 1 (lambda=1e-4) | Encourages sparse gene selection → better SHAP interpretability |
| **LDS** | Label Distribution Smoothing (100 bins, kernel=5) | Oversamples rare IC50 values (very sensitive cell lines are underrepresented) |
| **Early stopping** | Patience=20 on val Pearson r | Prevents overfitting — stops when validation correlation stops improving |
| **Gradient clipping** | max_norm=1.0 | Prevents exploding gradients from GIN message passing |

### Resource Monitoring

Training automatically logs:
- Parameter count (~3.05M trainable)
- GPU memory usage (after model load + peak during training)
- Wall-clock training time
- GPU device info

### Model Checkpoint

Best model (highest validation Pearson r) is saved to `checkpoints/best_model.pt` containing:
- Model state dict
- Optimizer state
- Epoch number
- Validation metrics

---

## Evaluation

### Run Evaluation
```bash
# All three splits
python experiments/evaluate.py

# Single split
python experiments/evaluate.py --split cell_blind
```

### What Gets Evaluated

**Per-split metrics**:

| Metric | Type | What it measures |
|--------|------|-----------------|
| Pearson r | Regression | Linear correlation (primary metric) |
| Spearman rho | Regression | Rank correlation (robust to outliers) |
| RMSE | Regression | Prediction error in ln(uM) units |
| MAE | Regression | Clinically interpretable error |
| AUROC | Classification | Sensitivity/resistance discrimination |
| AUPR | Classification | Performance on imbalanced classes |
| F1 | Classification | Harmonic mean of precision/recall |

**Per-drug breakdown**: Pearson r and RMSE for each individual drug (drugs with >= 10 test pairs)

**Failure case analysis**:
- Top 10 worst predictions (which drug-cell pairs fail?)
- Error by IC50 range (does model fail on sensitive or resistant cell lines?)
- Drug concentration of failures (are errors clustered in specific drugs?)

**MC Dropout uncertainty estimation** (cell_blind split):
- Mean/median prediction uncertainty (sigma)
- Error-uncertainty Spearman rho (does model know when it's wrong?)
- +/- 1 sigma and +/- 2 sigma calibration coverage

### Outputs
```
results/
├── evaluation_results.json         # All metrics per split
├── eval_cell_blind_predictions.csv # Per-sample predictions
├── eval_cell_blind_per_drug.csv    # Per-drug metrics
├── eval_cell_blind_failures.csv    # Worst predictions analysis
└── plots/
    ├── scatter_random.png
    ├── scatter_cell_blind.png
    └── scatter_drug_blind.png
```

---

## Explainability (XAI)

MIGN-XAI provides three complementary explainability approaches:

### 1. SHAP Gene Attribution
```bash
python results/xai/shap_analysis.py
```

Uses KernelSHAP to identify which cell line features (genes, pathways, mutations) drive each drug's sensitivity prediction.

**Key design**: The drug embedding is *fixed* and cell features are *varied*. This produces drug-specific gene attributions — "which genes make a cell line sensitive to Erlotinib?" — rather than global feature importance.

**Stability verification**: SHAP is run 3 times with different random seeds per drug. Reports:
- Top-10 Jaccard similarity across runs (gene set consistency)
- Rank Spearman rho across runs (full ranking consistency)
- Cross-drug consistency (do EGFR inhibitors identify similar genes?)

### 2. Attention Visualisation
```bash
python results/xai/attention_viz.py
```

Extracts and visualises cross-attention weights:
- Per-drug attention heatmaps (which cell encoder layers does each drug attend to?)
- Per-head specialisation analysis (do different heads learn different patterns?)
- Pathway-class grouped boxplots

### 3. Biological Validation
```bash
python results/xai/biological_val.py
```

Validates SHAP attributions against curated drug-target databases:

| Metric | What it measures |
|--------|-----------------|
| Precision@10 | Are top 10 SHAP genes known targets of this drug? |
| Precision@20 | Are top 20 SHAP genes known targets? |
| Recall@10 | What fraction of known targets appear in top 10? |
| Fisher's p-value | Is overlap significantly better than random? |

Validation drugs with known mechanisms: Erlotinib (EGFR), Vemurafenib (BRAF), Olaparib (PARP), Palbociclib (CDK4/6), Imatinib (BCR-ABL), Gefitinib (EGFR), Crizotinib (ALK), Trametinib (MEK).

---

## Experiments

### Ablation Study
```bash
python experiments/ablation.py
```

7 architectural variants tested on cell-blind split:

| Variant | What's changed | Tests |
|---------|---------------|-------|
| V1: Full MIGN-XAI | Nothing (baseline) | Full model performance |
| V2: Concat Fusion | Replace cross-attention with concatenation | Is cross-attention worth it? |
| V3: GEx Only | Remove mutation features | Do mutations add value? |
| V4: No Edge Features | Remove bond features from GIN | Do bond features matter? |
| V5: No LDS | Remove label distribution smoothing | Does LDS help? |
| V6: Random Split | Use random instead of cell-blind | How much does leakage inflate? |
| V7: Shallow GIN | 1 GIN layer instead of 3 | Is depth important? |

Reports delta-from-full-model (delta Pearson r, delta RMSE) and validates that the full model is the best variant.

### Baseline Comparison
```bash
python experiments/baselines.py
```

| Baseline | Drug Features | Cell Features | Fusion |
|----------|--------------|---------------|--------|
| SVM (LinearSVR) | ECFP fingerprints | Omics vector | Concatenation |
| Random Forest | ECFP fingerprints | Omics vector | Concatenation |
| GraphDRP-GIN | GIN graph | Omics MLP | Concatenation |
| **MIGN-XAI** | **GIN graph + edges** | **Hierarchical MLP** | **Cross-attention** |

All baselines use the **exact same cell-blind split** (verified via MD5 hash).

Improvements are tested for statistical significance using the **Wilcoxon signed-rank test** on per-sample squared errors (p < 0.05 required to claim significance).

### Cross-Dataset Generalisation
```bash
python experiments/cross_dataset.py
```

The most important validation test:

| Level | Test | Difficulty | What it proves |
|-------|------|-----------|---------------|
| Sanity | GDSC1 → GDSC2 | Easy | Same lab, same protocol — should work |
| **Real** | **GDSC → CCLE** | **Hard** | **Different lab (Sanger → Broad), different assay (CellTiter-Glo → CTP)** |

The GDSC → CCLE test uses **Spearman rank correlation** (not Pearson) because the two datasets measure sensitivity on different scales (ln(IC50) vs AUC). What matters is whether the *ranking* of cell line sensitivity is preserved.

Drug name matching uses a 3-strategy approach (exact → salt-form stripping → normalised matching) with **overlap bias warnings** when too few drugs match.

---

## Model Details

### Parameter Count

| Component | Parameters | Description |
|-----------|-----------|-------------|
| Drug Encoder (GIN) | 742,144 | 3 GINEConv layers + global pooling + output projection |
| Cell Encoder (MLP) | 1,974,272 | 4-layer hierarchical MLP |
| Cross-Attention Fusion | 165,632 | 4-head multi-head attention |
| Prediction Head | 165,121 | 3-layer MLP (512 → 256 → 128 → 1) |
| **Total** | **3,047,169** | ~3.05M trainable parameters |

### Uncertainty Estimation

MIGN-XAI supports **Monte Carlo Dropout** for uncertainty estimation:

```python
model.eval()
mean, std, all_preds = model.mc_dropout_predict(batch, n_forward=30)
# mean: (B,) predicted ln(IC50)
# std:  (B,) epistemic uncertainty
# "ln(IC50) = 3.2 ± 0.8"
```

This runs T stochastic forward passes with dropout enabled (BatchNorm stays in eval mode), giving a confidence interval on each prediction. The evaluation pipeline reports:
- **Error-uncertainty correlation**: Does the model know when it's unsure?
- **Calibration**: What fraction of true values fall within the predicted confidence interval?

### Input Specifications

| Input | Shape | Description |
|-------|-------|-------------|
| `batch.x` | (total_atoms, 45) | Atom features |
| `batch.edge_index` | (2, total_bonds) | Bond connectivity |
| `batch.edge_attr` | (total_bonds, 10) | Bond features |
| `batch.batch` | (total_atoms,) | Atom-to-molecule mapping |
| `batch.cell` | (B, ~1026) | Cell line omics features |
| `batch.y` | (B, 1) | Target ln(IC50) |

---

## Reproducibility

### Seed Control

`set_seed(42)` locks **all 7 sources of randomness**:

| Source | Method |
|--------|--------|
| Python `random` | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch GPU | `torch.cuda.manual_seed_all(42)` |
| cuDNN auto-tuner | `torch.backends.cudnn.benchmark = False` |
| cuDNN algorithms | `torch.backends.cudnn.deterministic = True` |
| PyTorch ops | `torch.use_deterministic_algorithms(True, warn_only=True)` |

### Data Split Verification

Every split is verified for data leakage using `_verify_no_leakage()`:
- **Cell-blind**: `AssertionError` if any cell line appears in both train and test
- **Drug-blind**: `AssertionError` if any drug appears in both train and test
- **Random**: Reports overlap percentage (expected leakage)

### Baseline Fairness

Baselines print a **split hash** (MD5 of cell/drug ID arrays) so you can verify they used the identical split as MIGN-XAI.

### Integration Tests
```bash
python test_integration.py
```

Runs 7 automated tests:

| Test | What it verifies |
|------|-----------------|
| 1 | Model forward pass produces correct output shapes |
| 2 | MC Dropout produces non-negative uncertainty estimates |
| 3 | Seed reproducibility across torch + numpy |
| 4 | Wilcoxon test correctly identifies the better model |
| 5 | Metrics computation is numerically correct |
| 6 | LDS weights have valid range |
| 7 | SHAP prediction pathway works end-to-end |

---

## Quick Start (Full Pipeline)

```bash
# 1. Download data
python data/download_data.py

# 2. Preprocess
python data/preprocess_gdsc.py
python data/preprocess_omics.py
python data/drug_graph.py

# 3. Train
python experiments/train.py --split cell_blind --epochs 200 --lr 3e-4 --lds

# 4. Evaluate
python experiments/evaluate.py

# 5. Ablation + Baselines
python experiments/ablation.py
python experiments/baselines.py

# 6. Cross-dataset generalisation
python experiments/cross_dataset.py

# 7. Explainability
python results/xai/shap_analysis.py
python results/xai/attention_viz.py
python results/xai/biological_val.py
```

Or use the **Colab notebook**: `notebooks/full_pipeline.ipynb`

---

## Key References

| Reference | Relevance |
|-----------|-----------|
| Xu et al. (2019) "How Powerful are Graph Neural Networks?" | GIN architecture theory |
| Nguyen et al. (2021) "GraphDRP" | Baseline GNN for drug response |
| Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation" | MC Dropout uncertainty |
| Yang et al. (2021) "Delving into Deep Imbalanced Regression" | Label Distribution Smoothing |
| Lundberg & Lee (2017) "SHAP" | Gene attribution explainability |
| Barretina et al. (2012) "CCLE" | Cell line omics data |
| Yang et al. (2013) "GDSC" | Drug sensitivity screening |
| Subramanian et al. (2017) "L1000" | Landmark gene selection |
| Liberzon et al. (2015) "MSigDB Hallmarks" | Pathway aggregation |

---

## License

This project is for research and educational purposes. All datasets used are publicly available.

---

## Acknowledgements

- **GDSC** (Wellcome Sanger Institute & Massachusetts General Hospital)
- **DepMap / CCLE** (Broad Institute)
- **MSigDB** (UC San Diego & Broad Institute)
- **RDKit** (Open-source cheminformatics)
- **PyTorch Geometric** (TU Dortmund)

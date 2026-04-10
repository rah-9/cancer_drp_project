"""
data/dataset.py
─────────────────────────────────────────────────────────────────────────────
Step 5: PyTorch Geometric Dataset class + three split strategies.

Classes:
  DRPDataset       — Core dataset: maps (drug_id, cosmic_id) pairs to
                     PyG Data objects combining drug graphs + cell features
  DataSplitter     — Creates train/val/test splits using three strategies:
                       1. random     — simple random 80/10/10
                       2. cell_blind — held-out cell lines (clinical relevance)
                       3. drug_blind — held-out drugs (repurposing relevance)

Usage:
    from data.dataset import DRPDataset, DataSplitter
    splitter = DataSplitter(gdsc1_df, drug_graphs, cell_features)
    train_ds, val_ds, test_ds = splitter.get_split("cell_blind")
"""

import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch_geometric.data import Dataset, Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (PROC_DIR, SEED, TEST_CELL_FRACTION, TEST_DRUG_FRACTION)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class DRPDataset(Dataset):
    """
    PyTorch Geometric Dataset for Drug Response Prediction.

    Each sample is a (drug, cell_line) pair. It returns a PyG Data
    object containing:
      - Drug molecular graph (nodes, edges, edge features from GIN encoder)
      - Cell line omics vector (gene expression + pathways + mutations)
      - Target: ln(IC50) for regression
      - Label: 0/1 for binary classification (resistant/sensitive)

    Parameters
    ----------
    pairs_df      : DataFrame with columns [drug_id, cosmic_id, ln_ic50, sensitive]
    drug_graphs   : dict {drug_id: PyG Data object}
    cell_features : DataFrame indexed by cosmic_id
    transform     : Optional PyG transform
    """

    def __init__(self, pairs_df: pd.DataFrame,
                 drug_graphs: dict,
                 cell_features: pd.DataFrame,
                 transform=None):
        super().__init__(transform=transform)
        self.pairs        = pairs_df.reset_index(drop=True)
        self.drug_graphs  = drug_graphs
        self.cell_features = cell_features
        # Pre-convert cell features to tensors for speed
        self._cell_tensor_cache = {}

    def len(self) -> int:
        return len(self.pairs)

    def get(self, idx: int) -> Data:
        """
        Retrieve a single (drug, cell) pair as a PyG Data object.

        The drug graph and cell features are combined into one Data object:
          data.x           — atom node features (from drug graph)
          data.edge_index  — bond connectivity (from drug graph)
          data.edge_attr   — bond features (from drug graph)
          data.cell        — cell line omics vector (concatenated)
          data.y           — target ln(IC50) (for regression)
          data.label       — binary sensitivity label (0=resistant, 1=sensitive)
          data.drug_id     — drug identifier
          data.cosmic_id   — cell line identifier
        """
        row = self.pairs.iloc[idx]
        drug_id   = row["drug_id"]
        cosmic_id = int(row["cosmic_id"])

        # ── Drug graph ──────────────────────────────────────────────────────
        drug_data = self.drug_graphs[drug_id].clone()

        # ── Cell features ───────────────────────────────────────────────────
        if cosmic_id not in self._cell_tensor_cache:
            feat_vec = self.cell_features.loc[cosmic_id].values.astype(np.float32)
            # Store as (1, n_features) so PyG batches to (B, n_features)
            self._cell_tensor_cache[cosmic_id] = torch.tensor(
                feat_vec, dtype=torch.float).unsqueeze(0)
        drug_data.cell = self._cell_tensor_cache[cosmic_id]

        # ── Targets ─────────────────────────────────────────────────────────
        drug_data.y         = torch.tensor([float(row["ln_ic50"])], dtype=torch.float)
        drug_data.label     = torch.tensor([int(row["sensitive"])], dtype=torch.long)
        drug_data.drug_id   = drug_id
        drug_data.cosmic_id = cosmic_id

        return drug_data


# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

class DataSplitter:
    """
    Creates train/validation/test splits using three strategies.

    Strategy selection rationale:
    ─────────────────────────────
    random    : Simplest baseline. Cell lines appear in both train and test.
                Inflates performance (data leakage). Only for sanity checking.

    cell_blind: 10% of unique cell lines reserved exclusively for test.
                No cell line in test appears in train. Simulates predicting
                drug response for a new patient's tumour profile.
                → Most clinically relevant evaluation.

    drug_blind: 10% of unique drugs reserved exclusively for test.
                No test drug appears in training. Simulates predicting
                response to a novel/repurposed compound.
                → Most relevant for drug repurposing.
    """

    def __init__(self,
                 pairs_df: pd.DataFrame,
                 drug_graphs: dict,
                 cell_features: pd.DataFrame):
        """
        Parameters
        ----------
        pairs_df      : Full drug-cell pair DataFrame from GDSC
        drug_graphs   : Dict of drug molecular graphs
        cell_features : Cell line feature matrix (indexed by cosmic_id)
        """
        # Filter pairs to those where we have BOTH a drug graph
        # and cell line features
        valid_drugs = set(drug_graphs.keys())
        valid_cells = set(cell_features.index)

        self.pairs = pairs_df[
            pairs_df["drug_id"].isin(valid_drugs) &
            pairs_df["cosmic_id"].isin(valid_cells)
        ].reset_index(drop=True)

        self.drug_graphs   = drug_graphs
        self.cell_features = cell_features

        n_original = len(pairs_df)
        n_filtered = len(self.pairs)
        print(f"  DataSplitter: {n_filtered:,} pairs available "
              f"({n_original - n_filtered:,} dropped due to missing graphs/features)")
        print(f"  Unique drugs: {self.pairs['drug_id'].nunique()}, "
              f"Cell lines: {self.pairs['cosmic_id'].nunique()}")

    def get_split(self,
                  strategy: str = "cell_blind",
                  val_fraction: float = 0.1,
                  test_fraction: float = 0.1,
                  seed: int = SEED):
        """
        Create train/val/test datasets for the given strategy.

        Parameters
        ----------
        strategy     : "random", "cell_blind", or "drug_blind"
        val_fraction : Fraction of data for validation
        test_fraction: Fraction of data for test
        seed         : Random seed for reproducibility

        Returns
        -------
        train_ds, val_ds, test_ds : DRPDataset objects
        """
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"\n  Creating {strategy} split ...")

        if strategy == "random":
            train_pairs, val_pairs, test_pairs = self._random_split(
                val_fraction, test_fraction, seed)

        elif strategy == "cell_blind":
            train_pairs, val_pairs, test_pairs = self._cell_blind_split(
                test_fraction, val_fraction, seed)

        elif strategy == "drug_blind":
            train_pairs, val_pairs, test_pairs = self._drug_blind_split(
                test_fraction, val_fraction, seed)

        else:
            raise ValueError(f"Unknown strategy: {strategy}. "
                             "Choose from: random, cell_blind, drug_blind")

        self._print_split_stats(train_pairs, val_pairs, test_pairs)

        # ── Data leakage verification ──────────────────────────────────────
        self._verify_no_leakage(strategy, train_pairs, val_pairs, test_pairs)

        train_ds = DRPDataset(train_pairs, self.drug_graphs, self.cell_features)
        val_ds   = DRPDataset(val_pairs,   self.drug_graphs, self.cell_features)
        test_ds  = DRPDataset(test_pairs,  self.drug_graphs, self.cell_features)

        return train_ds, val_ds, test_ds

    # ── Split implementations ─────────────────────────────────────────────

    def _random_split(self, val_frac, test_frac, seed):
        """
        Simple random split of all (drug, cell) pairs.
        WARNING: Pairs from the same cell line or drug appear in
        both train and test — this leaks information.
        Use only for sanity checking.
        """
        train_val, test = train_test_split(
            self.pairs, test_size=test_frac, random_state=seed)
        train, val = train_test_split(
            train_val, test_size=val_frac / (1 - test_frac), random_state=seed)
        return train, val, test

    def _cell_blind_split(self, test_frac, val_frac, seed):
        """
        Hold out a fraction of cell lines entirely.
        No test cell line appears in training data.
        Within training data, further split off a validation set
        (also cell-blind relative to training).
        """
        unique_cells = self.pairs["cosmic_id"].unique()
        np.random.seed(seed)

        n_test = max(1, int(len(unique_cells) * test_frac))
        n_val  = max(1, int(len(unique_cells) * val_frac))

        perm       = np.random.permutation(unique_cells)
        test_cells = set(perm[:n_test])
        val_cells  = set(perm[n_test:n_test + n_val])
        train_cells = set(perm[n_test + n_val:])

        train = self.pairs[self.pairs["cosmic_id"].isin(train_cells)]
        val   = self.pairs[self.pairs["cosmic_id"].isin(val_cells)]
        test  = self.pairs[self.pairs["cosmic_id"].isin(test_cells)]
        return train, val, test

    def _drug_blind_split(self, test_frac, val_frac, seed):
        """
        Hold out a fraction of drugs entirely.
        No test drug appears in training data.
        """
        unique_drugs = self.pairs["drug_id"].unique()
        np.random.seed(seed)

        n_test = max(1, int(len(unique_drugs) * test_frac))
        n_val  = max(1, int(len(unique_drugs) * val_frac))

        perm       = np.random.permutation(unique_drugs)
        test_drugs = set(perm[:n_test])
        val_drugs  = set(perm[n_test:n_test + n_val])
        train_drugs = set(perm[n_test + n_val:])

        train = self.pairs[self.pairs["drug_id"].isin(train_drugs)]
        val   = self.pairs[self.pairs["drug_id"].isin(val_drugs)]
        test  = self.pairs[self.pairs["drug_id"].isin(test_drugs)]
        return train, val, test

    def _print_split_stats(self, train, val, test):
        total = len(train) + len(val) + len(test)
        print(f"  Train: {len(train):6,} pairs ({100*len(train)/total:.1f}%) | "
              f"{train['drug_id'].nunique()} drugs | {train['cosmic_id'].nunique()} cells")
        print(f"  Val:   {len(val):6,} pairs ({100*len(val)/total:.1f}%) | "
              f"{val['drug_id'].nunique()} drugs | {val['cosmic_id'].nunique()} cells")
        print(f"  Test:  {len(test):6,} pairs ({100*len(test)/total:.1f}%) | "
              f"{test['drug_id'].nunique()} drugs | {test['cosmic_id'].nunique()} cells")

    def _verify_no_leakage(self, strategy, train, val, test):
        """
        Verify there is no data leakage between splits.

        For each split strategy, checks the appropriate constraint:
          - cell_blind: no cell line in test appears in train
          - drug_blind: no drug in test appears in train
          - random:     warns if cell/drug overlap exists (expected but noted)

        Raises AssertionError for blind strategies if leakage detected.
        """
        train_cells = set(train["cosmic_id"].unique())
        val_cells   = set(val["cosmic_id"].unique())
        test_cells  = set(test["cosmic_id"].unique())

        train_drugs = set(train["drug_id"].unique())
        val_drugs   = set(val["drug_id"].unique())
        test_drugs  = set(test["drug_id"].unique())

        if strategy == "cell_blind":
            # HARD CHECK: zero cell overlap
            leak_test  = train_cells & test_cells
            leak_val   = train_cells & val_cells
            assert len(leak_test) == 0, (
                f"DATA LEAKAGE: {len(leak_test)} cell lines in both "
                f"train and test: {list(leak_test)[:5]}..."
            )
            assert len(leak_val) == 0, (
                f"DATA LEAKAGE: {len(leak_val)} cell lines in both "
                f"train and val: {list(leak_val)[:5]}..."
            )
            print("  [LEAKAGE CHECK] cell_blind: PASSED "
                  "(0 cell overlap between train/val/test)")

        elif strategy == "drug_blind":
            # HARD CHECK: zero drug overlap
            leak_test  = train_drugs & test_drugs
            leak_val   = train_drugs & val_drugs
            assert len(leak_test) == 0, (
                f"DATA LEAKAGE: {len(leak_test)} drugs in both "
                f"train and test: {list(leak_test)[:5]}..."
            )
            assert len(leak_val) == 0, (
                f"DATA LEAKAGE: {len(leak_val)} drugs in both "
                f"train and val: {list(leak_val)[:5]}..."
            )
            print("  [LEAKAGE CHECK] drug_blind: PASSED "
                  "(0 drug overlap between train/val/test)")

        elif strategy == "random":
            # SOFT CHECK: report overlap (expected — this is why random is weak)
            cell_overlap = train_cells & test_cells
            drug_overlap = train_drugs & test_drugs
            print(f"  [LEAKAGE CHECK] random: WARNING (as expected)")
            print(f"    Cell overlap: {len(cell_overlap)} / "
                  f"{len(test_cells)} test cells also in train "
                  f"({100*len(cell_overlap)/max(len(test_cells),1):.0f}%)")
            print(f"    Drug overlap: {len(drug_overlap)} / "
                  f"{len(test_drugs)} test drugs also in train "
                  f"({100*len(drug_overlap)/max(len(test_drugs),1):.0f}%)")
            print(f"    -> Use cell_blind or drug_blind for rigorous evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Load everything from PROC_DIR
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data(gdsc_version: str = "gdsc1"):
    """
    Convenience function: load processed data and return a DataSplitter.

    Parameters
    ----------
    gdsc_version : "gdsc1" (default) or "gdsc2"

    Returns
    -------
    splitter : DataSplitter ready to create splits
    """
    print(f"\n  Loading processed data from {PROC_DIR} ...")

    pairs_file = PROC_DIR / f"{gdsc_version}_clean.csv"
    if not pairs_file.exists():
        raise FileNotFoundError(
            f"[ERROR] {pairs_file} not found. Run preprocess_gdsc.py first.")

    pairs_df = pd.read_csv(pairs_file)
    print(f"  Pairs: {len(pairs_df):,}")

    graphs_file = PROC_DIR / "drug_graphs.pt"
    if not graphs_file.exists():
        raise FileNotFoundError(
            "[ERROR] drug_graphs.pt not found. Run drug_graph.py first.")
    drug_graphs = torch.load(graphs_file, weights_only=False)
    print(f"  Drug graphs: {len(drug_graphs)}")

    features_file = PROC_DIR / "cell_features.csv"
    if not features_file.exists():
        raise FileNotFoundError(
            "[ERROR] cell_features.csv not found. Run preprocess_omics.py first.")
    cell_features = pd.read_csv(features_file, index_col=0)
    print(f"  Cell features: {cell_features.shape}")

    splitter = DataSplitter(pairs_df, drug_graphs, cell_features)
    return splitter


if __name__ == "__main__":
    # Test dataset creation
    try:
        splitter = load_all_data("gdsc1")
        for strategy in ["random", "cell_blind", "drug_blind"]:
            train_ds, val_ds, test_ds = splitter.get_split(strategy)
            sample = train_ds[0]
            print(f"\n  Sample data object ({strategy}):")
            print(f"    x (atoms):       {sample.x.shape}")
            print(f"    edge_index:      {sample.edge_index.shape}")
            print(f"    edge_attr:       {sample.edge_attr.shape}")
            print(f"    cell features:   {sample.cell.shape}")
            print(f"    target y:        {sample.y.item():.4f}")
            print(f"    label:           {sample.label.item()}")
    except FileNotFoundError as e:
        print(f"\n  {e}")
        print("  Run the preprocessing steps first.")
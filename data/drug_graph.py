"""
data/drug_graph.py
─────────────────────────────────────────────────────────────────────────────
Step 4: Convert drug SMILES strings to molecular graphs for GNN processing.

What this script does:
  1. Loads drug metadata (with SMILES from GDSC compounds file)
  2. For drugs missing SMILES, fetches them from PubChem via name
  3. Converts each SMILES to a PyTorch Geometric Data object:
       - Nodes (atoms): 45-dimensional feature vector
       - Edges (bonds): 10-dimensional feature vector
  4. Saves all drug graphs as a dictionary {drug_id: Data}

Output:
  - PROC_DIR/drug_graphs.pt    : Dict of PyG Data objects
  - PROC_DIR/drug_smiles.csv   : Drug SMILES lookup table

Run:
    python data/drug_graph.py
"""

import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PROC_DIR, ATOM_SYMBOLS, NODE_FEATURE_DIM, EDGE_FEATURE_DIM


# ─────────────────────────────────────────────────────────────────────────────
# ATOM FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def one_hot(value, categories: list) -> list:
    """One-hot encode a value against a list of categories."""
    return [int(value == c) for c in categories]


def atom_features(atom) -> list:
    """
    Build a 45-dimensional feature vector for a single atom.

    Features:
      [0:10]  Atom symbol one-hot  (10 categories incl. 'other')
      [10:21] Degree one-hot       (0 through 10)
      [21:26] Total H count        (0 through 4)
      [26:33] Implicit valence     (0 through 6)
      [33]    Is aromatic          (binary)
      [34:40] Ring size membership (rings of size 3-8)
      [40:43] Chirality            (unspecified, CW, CCW)
      [43:45] Formal charge        (sign + magnitude, 2 features)
    """
    from rdkit.Chem.rdchem import ChiralType

    feat = []

    # 1. Atom symbol (10-dim)
    sym = atom.GetSymbol() if atom.GetSymbol() in ATOM_SYMBOLS[:-1] else "other"
    feat += one_hot(sym, ATOM_SYMBOLS)                          # 10

    # 2. Degree (11-dim): number of directly bonded atoms
    feat += one_hot(min(atom.GetDegree(), 10), list(range(11))) # 11

    # 3. Total hydrogen count (5-dim)
    feat += one_hot(min(atom.GetTotalNumHs(), 4), list(range(5))) # 5

    # 4. Implicit valence (7-dim)
    feat += one_hot(min(atom.GetImplicitValence(), 6), list(range(7))) # 7

    # 5. Aromaticity (1-dim)
    feat += [int(atom.GetIsAromatic())]                         # 1

    # 6. Ring membership (6-dim): is atom in a ring of size 3–8?
    feat += [int(atom.IsInRingOfSize(r)) for r in range(3, 9)] # 6

    # 7. Chirality (3-dim)
    chiral = atom.GetChiralTag()
    feat += [
        int(chiral == ChiralType.CHI_UNSPECIFIED),
        int(chiral == ChiralType.CHI_TETRAHEDRAL_CW),
        int(chiral == ChiralType.CHI_TETRAHEDRAL_CCW),
    ]                                                           # 3

    # 8. Formal charge (2-dim): sign (negative=0, positive=1) + capped magnitude
    fc = atom.GetFormalCharge()
    feat += [int(fc >= 0), min(abs(fc), 2)]                    # 2

    # Total: 10+11+5+7+1+6+3+2 = 45
    assert len(feat) == 45, f"Expected 45 atom features, got {len(feat)}"
    return feat


def bond_features(bond) -> list:
    """
    Build a 10-dimensional feature vector for a single bond.

    Features:
      [0:4]  Bond type         (single, double, triple, aromatic)
      [4]    Is conjugated     (binary)
      [5]    Is in ring        (binary)
      [6]    Has stereo config (binary)
      [7:10] Stereo type       (E, Z, none — for double bonds)
    """
    from rdkit.Chem.rdchem import BondType, BondStereo

    bt = bond.GetBondType()
    stereo = bond.GetStereo()

    feat = [
        int(bt == BondType.SINGLE),
        int(bt == BondType.DOUBLE),
        int(bt == BondType.TRIPLE),
        int(bt == BondType.AROMATIC),
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        int(stereo != BondStereo.STEREONONE),
        int(stereo == BondStereo.STEREOE),
        int(stereo == BondStereo.STEREOZ),
        0,  # padding to reach 10
    ]

    assert len(feat) == 10, f"Expected 10 bond features, got {len(feat)}"
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# SMILES → GRAPH CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def smiles_to_graph(smiles: str, drug_id=None):
    """
    Convert a SMILES string to a PyTorch Geometric Data object.

    Parameters
    ----------
    smiles   : SMILES string for the drug molecule
    drug_id  : Optional identifier stored in the graph object

    Returns
    -------
    torch_geometric.data.Data with:
      .x           : Node (atom) features  — shape (n_atoms, 45)
      .edge_index  : COO format edge list  — shape (2, n_edges*2)
      .edge_attr   : Bond features         — shape (n_edges*2, 10)
      .drug_id     : Drug identifier (stored for bookkeeping)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from torch_geometric.data import Data

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Sanitize molecule (compute aromaticity, valence, etc.)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        # ── Node features ──────────────────────────────────────────────────
        node_feats = [atom_features(a) for a in mol.GetAtoms()]
        if len(node_feats) == 0:
            return None
        x = torch.tensor(node_feats, dtype=torch.float)  # (n_atoms, 45)

        # ── Edge features (bidirectional) ──────────────────────────────────
        edge_idx, edge_attr = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = bond_features(bond)
            # Add both directions for undirected graph
            edge_idx  += [[i, j], [j, i]]
            edge_attr += [bf,      bf]

        if len(edge_idx) == 0:
            # Single-atom molecule — no bonds
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr  = torch.zeros((0, 10), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_idx,  dtype=torch.long).t().contiguous()
            edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.drug_id = drug_id
        data.smiles  = smiles
        data.n_atoms = x.shape[0]
        return data

    except Exception as e:
        return None


def fetch_smiles_from_pubchem(drug_name: str) -> str | None:
    """
    Fetch SMILES string for a drug by name from PubChem.
    Used as fallback when SMILES is not in the GDSC compounds file.
    """
    try:
        import pubchempy as pcp
        compounds = pcp.get_compounds(drug_name, "name")
        if compounds:
            return compounds[0].isomeric_smiles
    except Exception:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_drug_graphs():
    print("\n" + "="*65)
    print("  Step 4: SMILES → Molecular Graphs")
    print("="*65 + "\n")

    drug_info_file = PROC_DIR / "drug_info.csv"
    if not drug_info_file.exists():
        raise FileNotFoundError(
            "[ERROR] drug_info.csv not found. Run preprocess_gdsc.py first."
        )

    drug_info = pd.read_csv(drug_info_file)
    print(f"  Loaded {len(drug_info)} drugs from drug_info.csv")

    # ── Resolve SMILES ──────────────────────────────────────────────────────
    # Step A: Use SMILES from GDSC if available
    smiles_available = drug_info.dropna(subset=["smiles"]) if "smiles" in drug_info.columns else pd.DataFrame()
    smiles_missing = drug_info[~drug_info.index.isin(smiles_available.index)]
    print(f"  SMILES from GDSC compounds: {len(smiles_available)}")
    print(f"  Drugs without SMILES: {len(smiles_missing)} (will fetch from PubChem)")

    # Step B: Fetch missing SMILES from PubChem
    fetched = {}
    if len(smiles_missing) > 0:
        print("  Fetching missing SMILES from PubChem ...")
        for _, row in tqdm(smiles_missing.iterrows(), total=len(smiles_missing)):
            smiles = fetch_smiles_from_pubchem(row["drug_name"])
            if smiles:
                fetched[row["drug_id"]] = smiles
        print(f"  Fetched {len(fetched)} SMILES from PubChem")

    # Combine
    smiles_dict = {}
    if "smiles" in drug_info.columns:
        for _, row in drug_info.dropna(subset=["smiles"]).iterrows():
            smiles_dict[row["drug_id"]] = row["smiles"]
    smiles_dict.update(fetched)

    # ── Convert SMILES to graphs ────────────────────────────────────────────
    print(f"\n  Converting {len(smiles_dict)} SMILES to molecular graphs ...")
    drug_graphs = {}
    failed = []

    for drug_id, smiles in tqdm(smiles_dict.items()):
        graph = smiles_to_graph(str(smiles), drug_id=drug_id)
        if graph is not None:
            drug_graphs[drug_id] = graph
        else:
            failed.append(drug_id)

    print(f"  Successfully converted: {len(drug_graphs)}")
    print(f"  Failed (invalid SMILES): {len(failed)}")
    if failed:
        print(f"  Failed drug IDs: {failed[:10]}{'...' if len(failed)>10 else ''}")

    # ── Print graph statistics ─────────────────────────────────────────────
    if drug_graphs:
        n_atoms_list = [g.n_atoms for g in drug_graphs.values()]
        n_edges_list = [g.edge_index.shape[1]//2 for g in drug_graphs.values()]
        print(f"\n  Graph statistics:")
        print(f"  Atoms per molecule: {np.mean(n_atoms_list):.1f} ± {np.std(n_atoms_list):.1f} "
              f"(min={min(n_atoms_list)}, max={max(n_atoms_list)})")
        print(f"  Bonds per molecule: {np.mean(n_edges_list):.1f} ± {np.std(n_edges_list):.1f}")
        print(f"  Node feature dim : {drug_graphs[list(drug_graphs.keys())[0]].x.shape[1]}")
        print(f"  Edge feature dim : {drug_graphs[list(drug_graphs.keys())[0]].edge_attr.shape[1]}")

    # ── Save ──────────────────────────────────────────────────────────────
    torch.save(drug_graphs, PROC_DIR / "drug_graphs.pt")

    # Also save SMILES CSV for reference
    smiles_df = pd.DataFrame(
        [(did, drug_info[drug_info["drug_id"]==did]["drug_name"].values[0]
          if did in drug_info["drug_id"].values else "unknown",
          smi)
         for did, smi in smiles_dict.items()],
        columns=["drug_id", "drug_name", "smiles"]
    )
    smiles_df.to_csv(PROC_DIR / "drug_smiles.csv", index=False)

    print(f"\n  Saved {len(drug_graphs)} drug graphs to {PROC_DIR}/drug_graphs.pt")
    print("  Next step: python data/dataset.py\n")
    return drug_graphs


if __name__ == "__main__":
    process_drug_graphs()
"""
models/gin_encoder.py
─────────────────────────────────────────────────────────────────────────────
Drug Encoder: Graph Isomorphism Network (GIN) with edge features.

Why GIN over GCN or GAT?
─────────────────────────
- GCN averages neighbour features: mathematically equivalent to a low-pass
  filter that smooths out structural differences between nearby atoms.
  Two different drug scaffolds with similar local environments become
  indistinguishable after a few GCN layers.

- GAT learns attention weights between atom pairs but its discriminative
  power is still bounded below that of GIN.

- GIN (Xu et al., ICLR 2019) is provably as expressive as the Weisfeiler-
  Lehman graph isomorphism test — the most powerful message-passing GNN
  for distinguishing non-isomorphic graphs. Subtle scaffold differences
  between drugs that have dramatically different binding profiles are
  captured by GIN where GCN and GAT fail.

Architecture:
  Input: Molecular graph G = (V, E)
    V = atoms (45-dim node features)
    E = bonds (10-dim edge features)
  3 GINEConv layers (GIN + edge features) with:
    - BatchNorm after each layer
    - Residual connections (from layer 2 onward)
    - Dropout for regularisation
  Readout: GlobalMeanPool + GlobalMaxPool concatenated
  Output: 512-dim drug embedding (2 × hidden_dim)
  Linear projection to out_dim (default 256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
from torch_geometric.nn import BatchNorm as GraphBN
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (GIN_HIDDEN_DIM, GIN_OUT_DIM, GIN_N_LAYERS,
                    GIN_DROPOUT, NODE_FEATURE_DIM, EDGE_FEATURE_DIM)


class GINLayer(nn.Module):
    """
    Single GIN layer with edge features (GINEConv).

    GINEConv update rule for atom i at layer l:
      h_i^(l) = MLP^(l)( (1 + ε) · h_i^(l-1)  +  Σ_{j∈N(i)} ReLU(h_j^(l-1) + e_ij) )

    where e_ij is the edge feature between atoms i and j.
    The learnable ε allows the model to weight self-features vs. neighbourhood.
    """

    def __init__(self, in_dim: int, hidden_dim: int, edge_dim: int,
                 dropout: float = 0.1):
        super().__init__()

        # The MLP inside GINEConv — two linear layers
        mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(mlp, edge_dim=edge_dim)
        self.bn   = GraphBN(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        """
        Parameters
        ----------
        x          : Atom features       (n_atoms, in_dim)
        edge_index : Bond connectivity   (2, n_bonds*2)
        edge_attr  : Bond features       (n_bonds*2, edge_dim)

        Returns
        -------
        Updated atom features            (n_atoms, hidden_dim)
        """
        h = self.conv(x, edge_index, edge_attr)
        h = self.bn(h)
        h = F.relu(h)
        h = self.drop(h)
        return h


class DrugGINEncoder(nn.Module):
    """
    3-layer GIN encoder that maps a molecular graph to a fixed-size
    drug embedding vector.

    Parameters
    ----------
    node_in  : Input atom feature dimension (default: 45)
    edge_in  : Input bond feature dimension (default: 10)
    hidden   : Hidden dimension inside GIN (default: 256)
    out_dim  : Output embedding dimension  (default: 256)
    n_layers : Number of GIN layers        (default: 3)
    dropout  : Dropout probability          (default: 0.1)

    Output
    ------
    z_drug : Drug embedding (batch_size, out_dim)
    """

    def __init__(self,
                 node_in:  int = NODE_FEATURE_DIM,
                 edge_in:  int = EDGE_FEATURE_DIM,
                 hidden:   int = GIN_HIDDEN_DIM,
                 out_dim:  int = GIN_OUT_DIM,
                 n_layers: int = GIN_N_LAYERS,
                 dropout:  float = GIN_DROPOUT):
        super().__init__()

        self.n_layers = n_layers
        self.hidden   = hidden

        # Input projection: map raw atom features to model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )

        # Edge feature projection: map bond features to hidden dim
        # (GINEConv requires edge_dim == in_dim for its internal add)
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_in, hidden),
            nn.ReLU(),
        )

        # Stack of GIN layers — all operate in 'hidden' dimension
        self.layers = nn.ModuleList([
            GINLayer(hidden, hidden, hidden, dropout)
            for _ in range(n_layers)
        ])

        # Output projection: 2*hidden (mean+max pool) → out_dim
        self.output_proj = nn.Sequential(
            nn.Linear(2 * hidden, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        """
        Parameters
        ----------
        x          : Atom node features   (total_atoms, node_in)
        edge_index : Edge connectivity    (2, total_edges)
        edge_attr  : Edge features        (total_edges, edge_in)
        batch      : Batch vector         (total_atoms,) — maps each
                     atom to its molecule in the batch

        Returns
        -------
        z_drug : Drug embeddings          (batch_size, out_dim)
        """
        # Project input features to model dimension
        h = self.input_proj(x)                           # (atoms, hidden)
        e = self.edge_proj(edge_attr)                    # (edges, hidden)

        # Message passing with residual connections
        for i, layer in enumerate(self.layers):
            h_new = layer(h, edge_index, e)
            # Residual: add input to output (same shape from layer 1+)
            h = h + h_new  # Skip connection for gradient flow

        # Graph-level readout
        # Why both mean and max?
        # - Mean captures "average molecular environment" (overall polarity etc.)
        # - Max captures "most pharmacophore-relevant feature" (key binding atoms)
        h_mean = global_mean_pool(h, batch)              # (batch, hidden)
        h_max  = global_max_pool(h, batch)               # (batch, hidden)
        h_pool = torch.cat([h_mean, h_max], dim=-1)      # (batch, 2*hidden)

        z_drug = self.output_proj(h_pool)                # (batch, out_dim)
        return z_drug

    def get_atom_embeddings(self, x, edge_index, edge_attr):
        """
        Return per-atom embeddings (for GNNExplainer XAI).
        Does NOT apply graph pooling.
        """
        h = self.input_proj(x)
        e = self.edge_proj(edge_attr)
        for layer in self.layers:
            h = h + layer(h, edge_index, e)
        return h  # (n_atoms, hidden)
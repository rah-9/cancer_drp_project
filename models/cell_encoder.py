"""
models/cell_encoder.py
─────────────────────────────────────────────────────────────────────────────
Cell Line Encoder: Sparse regularised MLP for omics features.

Architecture:
  Input:  Combined omics vector (GEx + Pathways + Mutations) ~1026-dim
  Layers: 4-layer MLP with progressively shrinking widths
          input → 1024 → 512 → 256 → 128
  Output:
    z_cell    : Final 128-dim cell line embedding (for fusion)
    cell_seq  : All 4 intermediate outputs stacked (B × 4 × 128)
                Used by cross-attention module

Key design decisions:
─────────────────────
1. L1 Regularisation on Layer 1
   Applied in the training loop (not inside this module). L1 on the
   first layer encourages sparse gene selection — the network learns
   to identify a small subset of highly predictive genes, which later
   facilitates SHAP attribution (fewer active features = clearer signal).

2. Intermediate Outputs as Sequence
   The cross-attention module needs a "sequence" to attend over.
   By saving outputs at each depth level, the drug encoder can learn
   which level of biological abstraction is most relevant:
   - Layer 1 (1024→512→256→128): raw gene-level signal
   - Layer 2: gene-pair interactions
   - Layer 3: pathway-level patterns
   - Layer 4: high-level cancer phenotype signal
   Different drugs attend to different levels — e.g., targeted therapies
   may focus on specific gene-level alterations while cytotoxic drugs
   may attend to high-level proliferation phenotype features.

3. BatchNorm before Dropout
   BN normalises activations before the stochastic dropout mask,
   preventing dead neurons from both over- and under-activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CELL_HIDDEN_DIMS, CELL_DROPOUT


class CellEncoderBlock(nn.Module):
    """Single MLP block: Linear → BatchNorm → ReLU → Dropout."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn     = nn.BatchNorm1d(out_dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.relu(self.bn(self.linear(x))))


class CellLineEncoder(nn.Module):
    """
    4-layer sparse MLP for omics feature encoding.

    Parameters
    ----------
    in_dim       : Input feature dimension (GEx + pathway + mutation)
    hidden_dims  : List of hidden layer widths [1024, 512, 256, 128]
    dropout      : Dropout probability (default 0.3)

    Attributes
    ----------
    out_dim      : Final embedding dimension (= hidden_dims[-1])
    n_layers     : Number of MLP layers
    """

    def __init__(self,
                 in_dim:      int,
                 hidden_dims: list = CELL_HIDDEN_DIMS,
                 dropout:     float = CELL_DROPOUT):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.out_dim     = hidden_dims[-1]
        self.n_layers    = len(hidden_dims)

        # Build layers: each reduces dimension progressively
        dims = [in_dim] + hidden_dims
        self.blocks = nn.ModuleList([
            CellEncoderBlock(dims[i], dims[i+1], dropout)
            for i in range(len(dims) - 1)
        ])

        # Projection layers to standardise all intermediate outputs
        # to the final hidden dimension (for cross-attention stacking)
        # Layer 0 output: hidden_dims[0] = 1024 → project to hidden_dims[-1] = 128
        # Layer 1 output: hidden_dims[1] = 512  → project to 128
        # etc.
        self.projectors = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[-1])
            if hidden_dims[i] != hidden_dims[-1]
            else nn.Identity()
            for i in range(len(hidden_dims))
        ])

    def forward(self, x):
        """
        Parameters
        ----------
        x : Omics feature vector (batch_size, in_dim)

        Returns
        -------
        z_cell   : Final embedding            (batch_size, out_dim=128)
        cell_seq : All intermediates stacked  (batch_size, n_layers, out_dim)
                   — for cross-attention module
        """
        h = x
        intermediates = []

        for i, block in enumerate(self.blocks):
            h = block(h)
            # Project intermediate to standard dim for cross-attention
            proj = self.projectors[i](h)
            intermediates.append(proj)

        z_cell   = intermediates[-1]                           # (B, 128)
        cell_seq = torch.stack(intermediates, dim=1)           # (B, n_layers, 128)

        return z_cell, cell_seq

    def get_layer1_weights(self):
        """Return Layer 1 weights for L1 regularisation in training loop."""
        return list(self.blocks[0].linear.parameters())
"""
models/cross_attention.py
─────────────────────────────────────────────────────────────────────────────
Cross-Attention Fusion Module: Drug queries over Cell Line sequence.

This is the architectural innovation of MIGN-XAI.

Why cross-attention instead of concatenation?
─────────────────────────────────────────────
Concatenation: [z_drug | z_cell] → the model sees all gene features
at once with no drug-specific routing. Every gene is equally "visible"
to every drug, regardless of whether that gene is mechanistically
relevant to the drug's target.

Cross-attention: the drug embedding acts as a Query that selectively
retrieves relevant information from the cell line representation.
Formally:
  α = softmax( Q_drug · K_cell^T / √d_k )
  context = α · V_cell

The attention weight α_i represents how much the drug focuses on
abstraction level i of the cell line encoding:
- α_1 high → drug is sensitive to raw gene-level features
  (e.g., BRCA1 point mutations for PARP inhibitors)
- α_3/α_4 high → drug is sensitive to high-level pathway phenotype
  (e.g., proliferation signature for anti-mitotics)

This drug-specific attention is biologically meaningful because:
  - Erlotinib's efficacy depends on EGFR exon 19/21 deletions (gene level)
  - Paclitaxel's efficacy depends on the overall proliferation state (pathway)
  - The cross-attention learns this distinction end-to-end.

Architecture:
  Query : Drug embedding            (B, d_drug=256) → projected to (B, 1, d)
  Keys  : Cell intermediate stack   (B, L=4, d_cell=128)
  Values: Cell intermediate stack   (B, L=4, d_cell=128)

  Multi-head attention with 4 heads:
    Each head attends to different biological aspects
    Head outputs are concatenated then projected back to d

  Output: Fused context vector      (B, d=128)
          Attention weights          (B, n_heads, 1, L) — saved for XAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GIN_OUT_DIM, CELL_HIDDEN_DIMS, ATTN_HEADS, ATTN_DIM


class CrossAttentionFusion(nn.Module):
    """
    Drug-to-cell cross-attention fusion module.

    Parameters
    ----------
    d_drug   : Drug embedding dimension  (default: 256 from GIN)
    d_cell   : Cell embedding dimension  (default: 128, all intermediates)
    n_heads  : Number of attention heads (default: 4)

    Outputs
    -------
    z_fused  : Drug-conditioned cell context  (B, d_cell)
    attn_w   : Attention weights             (B, n_heads, 1, L) for XAI
    """

    def __init__(self,
                 d_drug:  int = GIN_OUT_DIM,
                 d_cell:  int = CELL_HIDDEN_DIMS[-1],
                 n_heads: int = ATTN_HEADS):
        super().__init__()

        self.d_cell  = d_cell
        self.n_heads = n_heads

        # Project drug embedding to attention query space
        self.q_proj = nn.Sequential(
            nn.Linear(d_drug, d_cell),
            nn.LayerNorm(d_cell),
        )

        # Multi-head attention: drug query → cell keys and values
        self.mha = nn.MultiheadAttention(
            embed_dim=d_cell,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,  # (batch, seq, dim) convention
        )

        # Post-attention normalisation and projection
        self.layer_norm = nn.LayerNorm(d_cell)
        self.ff = nn.Sequential(
            nn.Linear(d_cell, d_cell * 2),
            nn.ReLU(),
            nn.Linear(d_cell * 2, d_cell),
        )
        self.ff_norm = nn.LayerNorm(d_cell)

    def forward(self, z_drug, cell_seq):
        """
        Parameters
        ----------
        z_drug   : Drug embedding from GIN     (B, d_drug)
        cell_seq : Cell line encoding stack    (B, L, d_cell)
                   L = number of MLP layers (=4)

        Returns
        -------
        z_fused  : Fused representation        (B, d_cell)
        attn_w   : Attention weights           (B, n_heads, 1, L)
        """
        B = z_drug.shape[0]

        # Project drug to query: (B, d_drug) → (B, 1, d_cell)
        # The [None] dimension is the "query sequence length" of 1
        q = self.q_proj(z_drug).unsqueeze(1)              # (B, 1, d_cell)

        # Keys and Values from cell sequence: (B, L, d_cell)
        k = v = cell_seq

        # Multi-head cross-attention
        # need_weights=True for XAI visualisation
        attn_out, attn_w = self.mha(
            query=q,
            key=k,
            value=v,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights
        )
        # attn_out: (B, 1, d_cell)
        # attn_w:   (B, n_heads, 1, L)

        attn_out = attn_out.squeeze(1)                     # (B, d_cell)

        # Residual + layer norm (transformer-style)
        z = self.layer_norm(attn_out + q.squeeze(1))

        # Feed-forward with residual (second transformer sub-layer)
        z_fused = self.ff_norm(z + self.ff(z))             # (B, d_cell)

        return z_fused, attn_w
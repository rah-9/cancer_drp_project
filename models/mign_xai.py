"""
models/mign_xai.py
─────────────────────────────────────────────────────────────────────────────
MIGN-XAI: Multimodal Interaction Graph-Omics Network with Explainability

Full model combining:
  1. DrugGINEncoder    — GIN with edge features for molecular graphs
  2. CellLineEncoder   — Sparse MLP for omics feature encoding
  3. CrossAttentionFusion — Drug queries over cell line sequence
  4. Prediction Head   — FC layers → ln(IC50) regression

Final representation before prediction:
  z_final = concat(z_fused, z_drug, z_cell)
           = concat(128, 256, 128) = 512-dim

Complete forward pass:
  Drug graph → GIN → z_drug (256)  ─────────────────────┐
  Cell omics → MLP → z_cell (128) + cell_seq (4×128) ──→ CrossAttn → z_fused (128)
                                                          └→ concat → z_final (512) → Head → ŷ

The concatenation of all three representations (fused + original drug + original cell)
ensures the prediction head has access to:
- The cross-modal interaction signal (z_fused)
- The unconditional drug representation (z_drug)
- The unconditional cell representation (z_cell)
This residual-style concatenation consistently outperforms using z_fused alone.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (GIN_OUT_DIM, CELL_HIDDEN_DIMS, ATTN_DIM,
                    HEAD_HIDDEN, HEAD_DROPOUT,
                    NODE_FEATURE_DIM, EDGE_FEATURE_DIM, ATTN_HEADS)
from models.gin_encoder     import DrugGINEncoder
from models.cell_encoder    import CellLineEncoder
from models.cross_attention import CrossAttentionFusion


class PredictionHead(nn.Module):
    """
    Regression head: takes fused representation and predicts ln(IC50).

    input_dim = GIN_OUT_DIM + ATTN_DIM + CELL_HIDDEN_DIMS[-1]
              =    256      +    128    +        128
              =    512
    """

    def __init__(self, input_dim: int, hidden_dim: int = HEAD_HIDDEN,
                 dropout: float = HEAD_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim // 2, 1),   # scalar regression output
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)        # (batch_size,)


class MIGN_XAI(nn.Module):
    """
    Multimodal Interaction Graph-Omics Network with Explainability.

    Parameters
    ----------
    node_in  : Atom feature dimension (default 45)
    edge_in  : Bond feature dimension (default 10)
    cell_in  : Cell omics feature dimension (GEx + pathway + mut)

    Key outputs
    -----------
    y_hat   : Predicted ln(IC50) — (batch_size,)
    attn_w  : Cross-attention weights — (B, n_heads, 1, L) for XAI
    """

    def __init__(self,
                 node_in: int = NODE_FEATURE_DIM,
                 edge_in: int = EDGE_FEATURE_DIM,
                 cell_in: int = None):
        super().__init__()

        if cell_in is None:
            raise ValueError("cell_in must be provided. "
                             "Set it to the dimension of your cell feature vector.")

        # ── Encoders ────────────────────────────────────────────────────────
        self.drug_encoder = DrugGINEncoder(
            node_in  = node_in,
            edge_in  = edge_in,
            hidden   = GIN_OUT_DIM,
            out_dim  = GIN_OUT_DIM,
            n_layers = 3,
        )

        self.cell_encoder = CellLineEncoder(
            in_dim      = cell_in,
            hidden_dims = CELL_HIDDEN_DIMS,
        )

        # ── Fusion ──────────────────────────────────────────────────────────
        self.fusion = CrossAttentionFusion(
            d_drug  = GIN_OUT_DIM,
            d_cell  = CELL_HIDDEN_DIMS[-1],
            n_heads = ATTN_HEADS,
        )

        # ── Prediction Head ─────────────────────────────────────────────────
        fused_dim = CELL_HIDDEN_DIMS[-1]          # 128 (from cross-attn)
        z_dim     = GIN_OUT_DIM + CELL_HIDDEN_DIMS[-1] + fused_dim  # 256+128+128=512
        self.head = PredictionHead(
            input_dim  = z_dim,
            hidden_dim = HEAD_HIDDEN,
        )

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialisation for all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        """
        Full forward pass.

        Parameters
        ----------
        batch : PyTorch Geometric Batch object containing:
          batch.x          : Atom features        (total_atoms, node_in)
          batch.edge_index : Bond connectivity     (2, total_bonds)
          batch.edge_attr  : Bond features         (total_bonds, edge_in)
          batch.batch      : Atom-to-molecule map  (total_atoms,)
          batch.cell       : Cell omics features   (B, cell_in)

        Returns
        -------
        y_hat   : Predicted ln(IC50)              (B,)
        attn_w  : Cross-attention weights         (B, n_heads, 1, L)
        """
        # ── Drug branch: molecular graph → embedding ─────────────────────
        z_drug = self.drug_encoder(
            x          = batch.x,
            edge_index = batch.edge_index,
            edge_attr  = batch.edge_attr,
            batch      = batch.batch,
        )  # (B, 256)

        # ── Cell branch: omics vector → embedding + sequence ─────────────
        z_cell, cell_seq = self.cell_encoder(batch.cell)
        # z_cell:   (B, 128)
        # cell_seq: (B, 4, 128)

        # ── Fusion: drug queries over cell sequence ───────────────────────
        z_fused, attn_w = self.fusion(z_drug, cell_seq)
        # z_fused: (B, 128)
        # attn_w:  (B, n_heads, 1, 4)

        # ── Concatenate all representations ───────────────────────────────
        z_final = torch.cat([z_fused, z_drug, z_cell], dim=-1)  # (B, 512)

        # ── Predict ln(IC50) ──────────────────────────────────────────────
        y_hat = self.head(z_final)  # (B,)

        return y_hat, attn_w

    def get_drug_embedding(self, batch):
        """Return drug embeddings only (for SHAP fixed-drug analysis)."""
        return self.drug_encoder(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch)

    def predict_from_embeddings(self, z_drug, cell_features_batch):
        """
        Predict IC50 given pre-computed drug embedding and raw cell features.
        Used in SHAP wrapper where drug is fixed and cell features vary.

        Parameters
        ----------
        z_drug             : Pre-computed drug embedding  (1, 256)
        cell_features_batch: Cell omics numpy array       (N, cell_in)

        Returns
        -------
        Predicted ln(IC50) array   (N,)
        """
        cell_tensor = torch.tensor(
            cell_features_batch, dtype=torch.float).to(z_drug.device)

        z_cell, cell_seq = self.cell_encoder(cell_tensor)

        # Expand drug embedding to match batch size
        B = cell_tensor.shape[0]
        z_drug_exp = z_drug.expand(B, -1)

        z_fused, _ = self.fusion(z_drug_exp, cell_seq)
        z_final = torch.cat([z_fused, z_drug_exp, z_cell], dim=-1)
        y_hat = self.head(z_final)
        return y_hat.detach().cpu().numpy()

    def mc_dropout_predict(self, batch, n_forward: int = 30):
        """
        Monte Carlo Dropout uncertainty estimation.

        Runs T stochastic forward passes with dropout enabled at inference
        time. The variance across passes estimates epistemic uncertainty
        (model uncertainty due to limited training data).

        Why MC Dropout?
        ───────────────
        A point prediction of ln(IC50) = 3.2 is useless to a clinician
        without knowing whether the model is confident. MC Dropout gives
        "ln(IC50) = 3.2 ± 0.8" which tells you this prediction is noisy.

        Parameters
        ----------
        batch      : PyG Batch object
        n_forward  : Number of stochastic forward passes (default 30)

        Returns
        -------
        mean    : (B,) mean predicted ln(IC50)
        std     : (B,) standard deviation (uncertainty)
        all_preds: (T, B) all stochastic predictions
        """
        # Enable dropout during inference
        def enable_dropout(module):
            if isinstance(module, torch.nn.Dropout):
                module.train()

        self.eval()  # BatchNorm stays in eval mode
        self.apply(enable_dropout)  # Only dropout goes to train mode

        preds = []
        with torch.no_grad():
            for _ in range(n_forward):
                y_hat, _ = self.forward(batch)
                preds.append(y_hat.cpu().numpy())

        self.eval()  # Reset everything to eval

        all_preds = np.stack(preds, axis=0)  # (T, B)
        mean = all_preds.mean(axis=0)        # (B,)
        std  = all_preds.std(axis=0)         # (B,)

        return mean, std, all_preds

    def count_parameters(self):
        """Count and print model parameters by component."""
        def count(module):
            return sum(p.numel() for p in module.parameters())

        total      = count(self)
        drug_params = count(self.drug_encoder)
        cell_params = count(self.cell_encoder)
        attn_params = count(self.fusion)
        head_params = count(self.head)

        print(f"\n  MIGN-XAI Parameter Count:")
        print(f"  Drug Encoder (GIN)    : {drug_params:>10,}")
        print(f"  Cell Encoder (MLP)    : {cell_params:>10,}")
        print(f"  Cross-Attention Fusion: {attn_params:>10,}")
        print(f"  Prediction Head       : {head_params:>10,}")
        print(f"  ─────────────────────────────────")
        print(f"  Total                 : {total:>10,}")
        return total


def build_model(cell_in: int, device: torch.device = None) -> MIGN_XAI:
    """
    Build and return MIGN_XAI model on the specified device.

    Parameters
    ----------
    cell_in : Dimension of cell line feature vector
    device  : Target device (auto-detected if None)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MIGN_XAI(
        node_in = NODE_FEATURE_DIM,
        edge_in = EDGE_FEATURE_DIM,
        cell_in = cell_in,
    ).to(device)

    model.count_parameters()
    print(f"  Device: {device}\n")
    return model
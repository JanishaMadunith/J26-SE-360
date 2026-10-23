"""Graph Neural Network (LightGCN) Ranking Layer for M4.

Learns high-order message-passing representations across:
User Context Nodes <---> Nutrient Nodes <---> Food Item Nodes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_WEIGHTS_PATH = MODELS_DIR / "gnn_embeddings.pt"


class LightGCN(nn.Module):
    """Light Graph Convolution Network for Recommender Systems."""

    def __init__(self, num_nodes: int, embedding_dim: int = 32, num_layers: int = 2):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Node initial embeddings (E_0)
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, norm_adj: torch.sparse_coo_tensor) -> torch.Tensor:
        """Graph Convolution: message passing across layers."""
        x = self.embedding.weight
        all_embeddings = [x]

        for _ in range(self.num_layers):
            x = torch.sparse.mm(norm_adj, x)
            all_embeddings.append(x)

        # Average multi-hop representations
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)
        return final_embeddings

    def score(self, final_emb: torch.Tensor, u_idx: int, f_idx: int) -> float:
        """Computes dot product similarity between user context and food."""
        u_vec = final_emb[u_idx]
        f_vec = final_emb[f_idx]
        return float(torch.dot(u_vec, f_vec).item())

    def cosine_score(self, final_emb: torch.Tensor, u_idx: int, f_idx: int) -> float:
        """Computes normalized cosine similarity between context node and candidate node."""
        u_vec = final_emb[u_idx]
        f_vec = final_emb[f_idx]
        u_norm = torch.norm(u_vec)
        f_norm = torch.norm(f_vec)
        if u_norm == 0.0 or f_norm == 0.0:
            return 0.0
        sim = torch.dot(u_vec, f_vec) / (u_norm * f_norm)
        return float(sim.clamp(-1.0, 1.0).item())


class GNNRecommenderRanker:
    """Inference ranker that uses trained GNN embeddings to rank KG candidate meals."""

    def __init__(self, weights_path: Path = MODEL_WEIGHTS_PATH):
        self.weights_path = weights_path
        self.model: LightGCN | None = None
        self.node_to_idx: dict[str, int] = {}
        self.idx_to_node: dict[int, str] = {}
        self.final_embeddings: torch.Tensor | None = None
        self.is_loaded = False
        self._load_if_available()

    def _load_if_available(self) -> bool:
        if not self.weights_path.exists():
            return False

        try:
            checkpoint = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            self.node_to_idx = checkpoint["node_to_idx"]
            self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
            emb_dim = checkpoint["embedding_dim"]
            num_layers = checkpoint["num_layers"]
            num_nodes = len(self.node_to_idx)

            self.model = LightGCN(num_nodes=num_nodes, embedding_dim=emb_dim, num_layers=num_layers)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            norm_adj = checkpoint["norm_adj"]
            with torch.no_grad():
                self.final_embeddings = self.model(norm_adj)

            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Warning: Could not load GNN checkpoint ({e}). Falling back to KG path scoring.")
            self.is_loaded = False
            return False

    def rank_candidates(self, context_state_id: str, candidate_food_ids: list[str]) -> list[tuple[str, float]]:
        """Scores candidate foods using high-order GNN embeddings."""
        if not self.is_loaded or self.final_embeddings is None:
            # Fallback if model not trained yet
            return [(fid, 1.0) for fid in candidate_food_ids]

        c_idx = self.node_to_idx.get(context_state_id)
        if c_idx is None:
            return [(fid, 1.0) for fid in candidate_food_ids]

        scored = []
        for fid in candidate_food_ids:
            f_idx = self.node_to_idx.get(fid)
            if f_idx is not None:
                sim = self.model.score(self.final_embeddings, c_idx, f_idx)
            else:
                sim = 0.0
            scored.append((fid, sim))

        return sorted(scored, key=lambda x: x[1], reverse=True)

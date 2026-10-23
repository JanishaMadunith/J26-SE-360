"""Trains LightGCN model on the Expanded Knowledge Graph (Baseline + Food.com).

Saves expanded embeddings to models/gnn_embeddings_expanded.pt
and writes specifications to models/model_card_expanded.json.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import networkx as nx
import torch
import torch.nn.functional as F

import sys
src_dir = str(Path(__file__).resolve().parents[2])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from m4_food.graph.gnn_ranker import LightGCN
from m4_food.graph.knowledge_base_expanded import create_expanded_knowledge_graph
from m4_food.graph.schema import NodeType, RelationType
from m4_food.graph.train_gnn import build_graph_tensors

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_WEIGHTS_PATH = MODELS_DIR / "gnn_embeddings_expanded.pt"
MODEL_CARD_PATH = MODELS_DIR / "model_card_expanded.json"


def train_expanded_lightgcn(
    epochs: int = 60,
    embedding_dim: int = 32,
    num_layers: int = 2,
    lr: float = 0.01,
    reg_weight: float = 1e-4,
) -> dict:
    print("=" * 80)
    print("TRAINING GRAPH NEURAL NETWORK (LightGCN) ON EXPANDED MULTI-SOURCE GRAPH")
    print("=" * 80)

    g = create_expanded_knowledge_graph()
    node_to_idx, norm_adj, positive_pairs = build_graph_tensors(g)
    num_nodes = len(node_to_idx)
    num_edges = norm_adj._nnz()
    print(f"Expanded Graph: {num_nodes} nodes, {num_edges} sparse edges.")
    print(f"Positive Training Triples: {len(positive_pairs)}")

    model = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.perf_counter()
    model.train()

    # BPR Loss Loop
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        final_embeddings = model(norm_adj)

        # Sample positive and negative nodes
        u_list, pos_list, neg_list = [], [], []
        for u, pos in positive_pairs:
            neg = random.randint(0, num_nodes - 1)
            while neg == pos:
                neg = random.randint(0, num_nodes - 1)
            u_list.append(u)
            pos_list.append(pos)
            neg_list.append(neg)

        u_tensor = torch.tensor(u_list, dtype=torch.long)
        pos_tensor = torch.tensor(pos_list, dtype=torch.long)
        neg_tensor = torch.tensor(neg_list, dtype=torch.long)

        u_emb = final_embeddings[u_tensor]
        pos_emb = final_embeddings[pos_tensor]
        neg_emb = final_embeddings[neg_tensor]

        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)

        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        l2_reg = (
            model.embedding(u_tensor).norm(2).pow(2)
            + model.embedding(pos_tensor).norm(2).pow(2)
            + model.embedding(neg_tensor).norm(2).pow(2)
        ) / len(u_list)

        total_loss = bpr_loss + reg_weight * l2_reg
        total_loss.backward()
        optimizer.step()

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs:02d} | BPR Loss: {bpr_loss.item():.4f} | Total Loss: {total_loss.item():.4f}")

    elapsed = time.perf_counter() - start_time
    print(f"Training completed in {elapsed:.3f} seconds.")

    # Save learned embeddings and node mapping
    model.eval()
    with torch.no_grad():
        final_emb = model(norm_adj).cpu()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "node_to_idx": node_to_idx,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
        "norm_adj": norm_adj,
    }
    torch.save(checkpoint, MODEL_WEIGHTS_PATH)
    print(f"Saved expanded GNN weights to: {MODEL_WEIGHTS_PATH}")

    model_card = {
        "model_name": "m4-lightgcn-expanded-multi-source-ranker",
        "version": "1.0.0-expanded",
        "architecture": "LightGCN (2-layer graph convolution)",
        "embedding_dim": embedding_dim,
        "loss_function": "Bayesian Personalized Ranking (BPR)",
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "positive_triples": len(positive_pairs),
        "total_training_samples": len(positive_pairs) * epochs,
        "training_time_sec": round(elapsed, 3),
        "data_sources": [
            "USDA FoodData Central SR Legacy",
            "Food.com Recipes (RAW_recipes.csv)",
            "Medical Research Institute (MRI) Sri Lanka",
            "PubMed Neuro-nutritional Clinical Literature",
        ],
        "target_task": "Predicts multi-source context-aware food and task compatibility ranking",
    }
    with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)
    print(f"Saved expanded model card to: {MODEL_CARD_PATH}")

    return model_card


if __name__ == "__main__":
    train_expanded_lightgcn()

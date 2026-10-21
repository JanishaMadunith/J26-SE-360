"""GNN Training Pipeline: Trains LightGCN on Knowledge Graph and User Interaction Triples.

Saves model weights and embeddings to models/gnn_embeddings.pt.
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
from m4_food.graph.knowledge_base import create_default_knowledge_graph
from m4_food.graph.schema import NodeType, RelationType

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"
MODEL_WEIGHTS_PATH = MODELS_DIR / "gnn_embeddings.pt"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"


def build_graph_tensors(g: nx.DiGraph) -> tuple[dict[str, int], torch.sparse_coo_tensor, list[tuple[int, int]]]:
    """Extracts node mapping, normalized symmetric adjacency matrix, and positive pairs."""
    nodes = list(g.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(nodes)}
    num_nodes = len(nodes)

    edge_list = []
    positive_pairs = []

    for u, v, data in g.edges(data=True):
        u_idx = node_to_idx[u]
        v_idx = node_to_idx[v]
        # Undirected edge for symmetric message passing
        edge_list.append((u_idx, v_idx))
        edge_list.append((v_idx, u_idx))

        # Check if positive contextual pair (State <-> Nutrient or Food <-> Nutrient)
        rel = data.get("relation")
        if rel in (RelationType.ALLEVIATES, RelationType.BOOSTS, RelationType.REGULATES, RelationType.CONTAINS):
            positive_pairs.append((u_idx, v_idx))

    # Add self loops
    for i in range(num_nodes):
        edge_list.append((i, i))

    edge_list = list(set(edge_list))
    src = torch.tensor([e[0] for e in edge_list], dtype=torch.long)
    dst = torch.tensor([e[1] for e in edge_list], dtype=torch.long)

    # Compute Degree Matrix D
    degrees = torch.zeros(num_nodes, dtype=torch.float32)
    for s in src:
        degrees[s] += 1.0

    # Symmetric Normalization: D^{-1/2} A D^{-1/2}
    deg_inv_sqrt = torch.pow(degrees, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    values = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    indices = torch.stack([src, dst], dim=0)

    norm_adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()
    return node_to_idx, norm_adj, positive_pairs


def train_lightgcn(
    epochs: int = 60,
    embedding_dim: int = 32,
    num_layers: int = 2,
    lr: float = 0.01,
    reg_weight: float = 1e-4,
) -> dict:
    """Trains the GNN using Bayesian Personalized Ranking (BPR) loss."""
    print("=" * 80)
    print("TRAINING GRAPH NEURAL NETWORK (LightGCN) FOR CONTEXT-AWARE RANKING")
    print("=" * 80)

    g = create_default_knowledge_graph()
    node_to_idx, norm_adj, positive_pairs = build_graph_tensors(g)
    num_nodes = len(node_to_idx)
    print(f"Graph constructed: {num_nodes} nodes, {norm_adj._nnz()} sparse edges.")

    model = LightGCN(num_nodes=num_nodes, embedding_dim=embedding_dim, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_time = time.perf_counter()
    model.train()

    # BPR Loss Loop
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        final_embeddings = model(norm_adj)

        # Sample positive pairs and random negative nodes
        u_list = []
        pos_list = []
        neg_list = []

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

        # BPR Loss: -ln(sigmoid(pos_score - neg_score))
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        # L2 Regularization on initial embeddings
        l2_reg = (
            model.embedding(u_tensor).norm(2).pow(2)
            + model.embedding(pos_tensor).norm(2).pow(2)
            + model.embedding(neg_tensor).norm(2).pow(2)
        ) * reg_weight / len(u_list)

        total_loss = bpr_loss + l2_reg
        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - BPR Loss: {bpr_loss.item():.4f} - Total Loss: {total_loss.item():.4f}")

    train_duration = time.perf_counter() - start_time
    print(f"\nTraining completed in {train_duration:.2f} seconds.")

    # Save Checkpoint
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "node_to_idx": node_to_idx,
        "norm_adj": norm_adj,
        "embedding_dim": embedding_dim,
        "num_layers": num_layers,
    }
    torch.save(checkpoint, MODEL_WEIGHTS_PATH)
    print(f"Model checkpoint saved to: {MODEL_WEIGHTS_PATH}")

    # Save Model Card
    card = {
        "model_name": "m4-lightgcn-knowledge-graph-ranker",
        "version": "0.3.0",
        "architecture": "LightGCN (2-layer graph convolution)",
        "embedding_dim": embedding_dim,
        "loss_function": "Bayesian Personalized Ranking (BPR)",
        "num_nodes": num_nodes,
        "training_time_sec": round(train_duration, 2),
        "target_task": "Predicts user context-food-nutrient compatibility ranking",
    }
    with open(MODEL_CARD_PATH, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"Model card saved to: {MODEL_CARD_PATH}")

    return card


if __name__ == "__main__":
    train_lightgcn()

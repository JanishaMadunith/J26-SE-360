"""Expanded Knowledge Base: Integrates Baseline Foods + Food.com Extracted Global Foods.

Builds an expanded heterogeneous NetworkX graph and produces an interactive PyVis visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
import networkx as nx

import sys
src_dir = str(Path(__file__).resolve().parents[2])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from m4_food.graph.knowledge_base import create_default_knowledge_graph
from m4_food.graph.schema import KGNode, NodeType, RelationType

EXPANDED_FOODS_JSON = Path(__file__).resolve().parent / "data" / "expanded_foods.json"
EXPANDED_HTML = Path(__file__).resolve().parent / "knowledge_graph_expanded.html"


def create_expanded_knowledge_graph() -> nx.DiGraph:
    """Builds the expanded graph combining baseline + Food.com foods."""
    # 1. Start with the baseline validated graph (72 nodes, 197 edges)
    g = create_default_knowledge_graph()

    if not EXPANDED_FOODS_JSON.exists():
        print(f"Warning: {EXPANDED_FOODS_JSON} does not exist.")
        return g

    with open(EXPANDED_FOODS_JSON, "r", encoding="utf-8") as f:
        expanded_list = json.load(f)

    print(f"Integrating {len(expanded_list)} global recipes from Food.com into Knowledge Graph...")

    for item in expanded_list:
        name = item["name"]
        fid = f"food:{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        if g.has_node(fid):
            continue

        props = {
            "calories": item["calories"],
            "portion": item["portion"],
            "source": item.get("source", "Food.com"),
            "ingredients": item.get("ingredients_sample", []),
        }

        node_obj = KGNode(
            id=fid,
            name=name,
            node_type=NodeType.FOOD,
            properties=props,
        )
        g.add_node(fid, node=node_obj)

        # Connect nutrients (CONTAINS)
        for nut in item.get("nutrients", []):
            nid = f"nutrient:{nut}"
            if not g.has_node(nid):
                g.add_node(nid, node=KGNode(id=nid, name=nut.capitalize(), node_type=NodeType.NUTRIENT, properties={}))
            g.add_edge(fid, nid, relation=RelationType.CONTAINS, weight=1.0)

        # Connect allergens (HAS_ALLERGEN)
        for alg in item.get("allergens", []):
            aid = f"allergen:{alg}"
            if not g.has_node(aid):
                g.add_node(aid, node=KGNode(id=aid, name=alg.capitalize(), node_type=NodeType.ALLERGEN, properties={}))
            g.add_edge(fid, aid, relation=RelationType.HAS_ALLERGEN, weight=1.0)

        # Connect diets (COMPLIES_WITH)
        diets = item.get("diet_tags", {})
        if diets.get("vegan"):
            g.add_edge(fid, "diet:vegan", relation=RelationType.COMPLIES_WITH, weight=1.0)
        if diets.get("vegetarian"):
            g.add_edge(fid, "diet:vegetarian", relation=RelationType.COMPLIES_WITH, weight=1.0)
        if diets.get("halal"):
            g.add_edge(fid, "diet:halal", relation=RelationType.COMPLIES_WITH, weight=1.0)

    print(f"Expanded Knowledge Graph built: {len(g.nodes())} total nodes, {len(g.edges())} total edges.")
    return g


def visualize_expanded_graph() -> Path:
    from pyvis.network import Network
    from m4_food.graph.visualize_graph import NODE_COLORS, EDGE_COLORS

    g = create_expanded_knowledge_graph()
    net = Network(height="900px", width="100%", bgcolor="#14141F", font_color="#ECEFF1", directed=True)

    net.set_options("""
    {
      "nodes": {
        "font": { "size": 13, "face": "Segoe UI, Arial, sans-serif" },
        "borderWidth": 2
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 120 }
      }
    }
    """)

    for nid, data in g.nodes(data=True):
        node_obj = data.get("node")
        if not node_obj:
            continue
        style = NODE_COLORS.get(node_obj.node_type, {"color": "#9E9E9E", "shape": "dot", "size": 16})
        props = node_obj.properties

        lines = [f"<b>{node_obj.name}</b>", f"<i>Type: {node_obj.node_type.value}</i>"]
        if "calories" in props:
            lines.append(f"Calories: {props['calories']} kcal")
        if "portion" in props:
            lines.append(f"Portion: {props['portion']}")
        if "source" in props:
            lines.append(f"Source: {props['source']}")
        if "ingredients" in props and props["ingredients"]:
            lines.append(f"Ingredients: {', '.join(props['ingredients'])}")

        net.add_node(
            nid,
            label=node_obj.name,
            title="<br>".join(lines),
            color=style["color"],
            shape=style["shape"],
            size=style["size"],
        )

    for u, v, data in g.edges(data=True):
        rel = data.get("relation")
        weight = float(data.get("weight", 1.0))
        edge_color = EDGE_COLORS.get(rel, "#B0BEC5")
        net.add_edge(u, v, title=f"<b>{rel.value if hasattr(rel, 'value') else str(rel)}</b>", color=edge_color, width=1.0 + weight)

    net.save_graph(str(EXPANDED_HTML))
    print(f"Saved interactive expanded HTML visualization at:\n  --> {EXPANDED_HTML}")
    return EXPANDED_HTML


if __name__ == "__main__":
    visualize_expanded_graph()

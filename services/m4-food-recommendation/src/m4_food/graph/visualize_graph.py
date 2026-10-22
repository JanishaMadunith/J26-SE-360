"""Interactive Knowledge Graph Visualizer for M4.

Generates an interactive, physics-enabled HTML visualization (via PyVis)
with color-coded nodes, hover tooltips, and relationship filters.

Run:
    python services/m4-food-recommendation/src/m4_food/graph/visualize_graph.py
"""

from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

# Ensure package on path
src_dir = str(Path(__file__).resolve().parents[2])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from pyvis.network import Network

from m4_food.graph.knowledge_base import create_default_knowledge_graph
from m4_food.graph.schema import NodeType, RelationType

OUTPUT_HTML = Path(__file__).resolve().parent / "knowledge_graph.html"

# Aesthetic color palette for presentation
NODE_COLORS = {
    NodeType.AFFECTIVE_STATE: {"color": "#E64A19", "shape": "dot", "size": 32},       # Deep Orange-Red
    NodeType.COGNITIVE_STATE: {"color": "#F57C00", "shape": "dot", "size": 30},       # Amber-Orange
    NodeType.NUTRIENT: {"color": "#2E7D32", "shape": "diamond", "size": 24},          # Forest Green
    NodeType.FOOD: {"color": "#1976D2", "shape": "box", "size": 20},                  # Professional Blue
    NodeType.WELLNESS_TASK: {"color": "#7B1FA2", "shape": "ellipse", "size": 22},      # Royal Purple
    NodeType.HEALTH_CONDITION: {"color": "#C2185B", "shape": "triangle", "size": 26},  # Crimson
    NodeType.ALLERGEN: {"color": "#D32F2F", "shape": "triangleDown", "size": 22},      # Red Warning
    NodeType.DIETARY_PREFERENCE: {"color": "#388E3C", "shape": "star", "size": 22},    # Leaf Green
}

EDGE_COLORS = {
    RelationType.ALLEVIATES: "#4CAF50",         # Green = beneficial
    RelationType.BOOSTS: "#8BC34A",             # Light Green = energizing
    RelationType.REGULATES: "#00BCD4",          # Cyan = stabilizing
    RelationType.CONTAINS: "#90CAF9",           # Soft Blue = structural
    RelationType.CONTRAINDICATED_FOR: "#FF1744", # Bright Red = safety risk
    RelationType.HAS_ALLERGEN: "#FF5252",        # Coral Red = allergen
    RelationType.COMPLIES_WITH: "#AED581",       # Pale Green = compliant
}


def build_interactive_graph(open_in_browser: bool = True) -> Path:
    print("=" * 70)
    print("BUILDING INTERACTIVE KNOWLEDGE GRAPH VISUALIZATION")
    print("=" * 70)

    g = create_default_knowledge_graph()
    net = Network(height="850px", width="100%", bgcolor="#1A1A24", font_color="#ECEFF1", directed=True)

    # Physics configuration for smooth presentation exploration
    net.set_options("""
    {
      "nodes": {
        "font": {
          "size": 14,
          "face": "Segoe UI, Arial, sans-serif"
        },
        "borderWidth": 2
      },
      "edges": {
        "smooth": {
          "type": "continuous",
          "roundness": 0.3
        },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.8 }
        }
      },
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -60,
          "centralGravity": 0.015,
          "springLength": 120,
          "springConstant": 0.06
        },
        "solver": "forceAtlas2Based",
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # 1. Add Nodes
    for nid, data in g.nodes(data=True):
        node_obj = data.get("node")
        if not node_obj:
            continue

        style = NODE_COLORS.get(node_obj.node_type, {"color": "#9E9E9E", "shape": "dot", "size": 18})
        props = node_obj.properties

        # Rich HTML tooltip
        tooltip_lines = [
            f"<b>{node_obj.name}</b>",
            f"<i>Type: {node_obj.node_type.value}</i>",
        ]
        if "calories" in props:
            tooltip_lines.append(f"Calories: {props['calories']} kcal")
        if "portion" in props:
            tooltip_lines.append(f"Portion: {props['portion']}")
        if "duration_minutes" in props:
            tooltip_lines.append(f"Duration: {props['duration_minutes']} min")
        if "usda_nutrients" in props:
            usda = props["usda_nutrients"]
            tooltip_lines.append("<br><b>USDA Lab Data:</b>")
            for k, v in usda.items():
                tooltip_lines.append(f"• {k.replace('_', ' ').capitalize()}: {v}")
        if "bio_rationale" in props:
            tooltip_lines.append(f"<br>Note: {props['bio_rationale']}")

        tooltip_html = "<br>".join(tooltip_lines)

        net.add_node(
            nid,
            label=node_obj.name,
            title=tooltip_html,
            color=style["color"],
            shape=style["shape"],
            size=style["size"],
        )

    # 2. Add Edges
    for u, v, data in g.edges(data=True):
        rel = data.get("relation")
        weight = float(data.get("weight", 1.0))
        exp = data.get("explanation", "")
        edge_color = EDGE_COLORS.get(rel, "#B0BEC5")

        tooltip = f"<b>{rel.value if hasattr(rel, 'value') else str(rel)}</b><br>Weight: {weight}"
        if exp:
            tooltip += f"<br>Mechanism: {exp}"

        width = 1.0 + (weight * 1.5)
        dashes = rel in (RelationType.CONTRAINDICATED_FOR, RelationType.HAS_ALLERGEN)

        net.add_edge(
            u,
            v,
            title=tooltip,
            color=edge_color,
            width=width,
            dashes=dashes,
        )

    print(f"Total Nodes: {len(g.nodes())} | Total Edges: {len(g.edges())}")
    net.save_graph(str(OUTPUT_HTML))
    print(f"Successfully generated interactive HTML at:\n  --> {OUTPUT_HTML}\n")

    if open_in_browser:
        print("Opening in your default web browser...")
        webbrowser.open(OUTPUT_HTML.as_uri())

    return OUTPUT_HTML


if __name__ == "__main__":
    build_interactive_graph(open_in_browser=True)

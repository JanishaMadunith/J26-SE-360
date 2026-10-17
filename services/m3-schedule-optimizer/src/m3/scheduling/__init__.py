"""Schedule optimisers for Module 3.

* heuristic.py — energy-aware greedy scheduler (serving baseline, v0.1.0)
* cpsat.py     — planned OR-Tools CP-SAT optimiser (design notes only)
"""

from .heuristic import MODEL, completion_probability, schedule  # noqa: F401

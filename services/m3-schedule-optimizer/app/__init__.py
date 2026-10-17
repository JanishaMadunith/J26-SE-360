"""Serving layer for m3-schedule-optimizer.

Adds ../src to sys.path so the research package `m3` imports the same way in tests, uvicorn and
Docker, without installing it.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

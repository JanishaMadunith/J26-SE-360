"""Load the Open University Learning Analytics Dataset (OULAD) from DATA_DIR/oulad."""

from pathlib import Path

import pandas as pd

from .config import DATA_DIR

KEYS = ["code_module", "code_presentation", "id_student"]


def _find(base: Path, name: str) -> Path:
    for cand in (base / "raw" / name, base / "processed" / f"{name}.gz", base / "raw" / f"{name}.gz"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"{name} not found under {base}/raw or {base}/processed")


def load_oulad(data_dir: Path = DATA_DIR) -> dict:
    base = Path(data_dir) / "oulad"
    t = {}
    for name in ["assessments", "courses", "studentAssessment", "studentInfo", "studentRegistration", "vle"]:
        t[name] = pd.read_csv(_find(base, f"{name}.csv"))
    t["studentVle"] = pd.read_csv(
        _find(base, "studentVle.csv"),
        dtype={
            "code_module": "category",
            "code_presentation": "category",
            "id_student": "int32",
            "id_site": "int32",
            "date": "int16",
            "sum_click": "int32",
        },
    )
    return t

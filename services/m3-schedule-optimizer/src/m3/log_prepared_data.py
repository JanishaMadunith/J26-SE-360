"""Show the prepared data of every dataset in MLflow (experiment "M3 - Prepared data").

One run per dataset (OULAD, MS-LaTTE, PMData, ATUS, StudentLife) with:
  * Datasets   - the prepared sample registered as an MLflow dataset (schema + profile, "Datasets used" column)
  * Metrics    - the size numbers (prepared rows, features, students, ...)
  * Artifacts  - prepared_sample.csv (opens as a table), preparation_steps.md, stats.json and the extra tables
  * Description - where the raw data comes from and what it is used for

Reads models/prepared_preview.json (built by m3.build_prepared_preview), so it runs in seconds.

Usage (from services/m3-schedule-optimizer):
    PYTHONPATH=src python -m m3.log_prepared_data
"""

import argparse
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from .config import MODELS_DIR, MODULE_ROOT

EXPERIMENT = "M3 - Prepared data"
SOURCES = {
    "OULAD": "https://analyse.kmi.open.ac.uk/open_dataset",
    "MS-LaTTE": "https://github.com/microsoft/MS-LaTTE",
    "PMData": "https://datasets.simula.no/pmdata/",
    "ATUS": "https://www.bls.gov/tus/data.htm",
    "StudentLife": "https://studentlife.cs.dartmouth.edu/dataset.html",
}
TARGETS = {"OULAD": "y", "MS-LaTTE": "label (most votes)"}


def frame(t: dict) -> pd.DataFrame:
    return pd.DataFrame(t["rows"], columns=t["columns"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", default=f"sqlite:///{MODULE_ROOT / 'mlflow.db'}")
    a = ap.parse_args()
    import mlflow

    preview = json.loads((MODELS_DIR / "prepared_preview.json").read_text())
    mlflow.set_tracking_uri(a.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    for d in preview["datasets"]:
        key = d["name"].split(" ")[0]
        sample = frame(d["sample"])
        with mlflow.start_run(run_name=f"prepared/{key}"), tempfile.TemporaryDirectory() as tmp:
            steps = "\n".join(f"{i}. {s}" for i, s in enumerate(d["steps"], 1))
            mlflow.set_tags(
                {
                    "kind": "prepared data",
                    "dataset": key,
                    "used_for": d["used_for"],
                    "mlflow.note.content": f"**{d['name']}**\n\nUsed for: {d['used_for']}\n\n"
                    f"Raw data: {d['raw']}\n\nPreparation steps:\n\n{steps}",
                }
            )
            ds = mlflow.data.from_pandas(
                sample.astype(str),
                source=SOURCES.get(key, "local"),
                name=f"{key} prepared (sample of {len(sample)} rows)",
                targets=TARGETS.get(key),
            )
            mlflow.log_input(ds, context="prepared")
            metrics, params = {}, {}
            for k, v in d["stats"].items():
                name = re.sub(r"[^A-Za-z0-9_\-. /]", "", k).strip().replace(" ", "_")
                (metrics if isinstance(v, (int, float)) else params)[name] = v
            mlflow.log_metrics(metrics)
            if params:
                mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
            out = Path(tmp)
            sample.to_csv(out / "prepared_sample.csv", index=False)
            (out / "preparation_steps.md").write_text(
                f"# {d['name']}\n\n**Used for:** {d['used_for']}\n\n**Raw data:** {d['raw']}\n\n## Steps\n\n{steps}\n"
            )
            (out / "stats.json").write_text(json.dumps(d["stats"], indent=1, ensure_ascii=False))
            if "per_horizon" in d:
                pd.DataFrame(d["per_horizon"]).T.rename_axis("asked").to_csv(out / "cases_per_horizon.csv")
            if "example_day" in d:
                frame(d["example_day"]["hourly"]).to_csv(out / "example_day_hourly_module2_format.csv", index=False)
            if "one_row_all_features" in d:
                (out / "one_case_all_features.json").write_text(json.dumps(d["one_row_all_features"], indent=1))
            mlflow.log_artifacts(str(out))
        print(f"logged prepared/{key}: {len(sample)} sample rows, {len(metrics)} numbers")


if __name__ == "__main__":
    main()

"""Export JSON Schemas for every contract into contracts/schemas/ (used by the Flutter app and docs).

python contracts/export_schemas.py          # write
python contracts/export_schemas.py --check  # CI: fail if schemas are out of date
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
import wellness_contracts as wc  # noqa: E402

OUT = Path(__file__).parent / "schemas"
EXPORT = {
    "sensor_batch": wc.SensorBatch,
    "sensor_batch_ack": wc.SensorBatchAck,
    "task_create": wc.TaskCreate,
    "task_update": wc.TaskUpdate,
    "task": wc.Task,
    "m1_affective_state_request": wc.AffectiveStateRequest,
    "m1_affective_state_result": wc.AffectiveStateResult,
    "m2_wellness_forecast_request": wc.WellnessForecastRequest,
    "m2_wellness_forecast": wc.WellnessForecast,
    "m3_schedule_request": wc.ScheduleRequest,
    "m3_schedule_result": wc.ScheduleResult,
    "m4_recommendation_request": wc.RecommendationRequest,
    "m4_recommendations": wc.Recommendations,
}


def main(check=False):
    OUT.mkdir(exist_ok=True)
    stale = []
    for name, model in EXPORT.items():
        text = json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n"
        f = OUT / f"{name}.schema.json"
        if check:
            if not f.exists() or f.read_text() != text:
                stale.append(f.name)
        else:
            f.write_text(text)
    if check and stale:
        print("Out-of-date schemas (run `make contracts`):", ", ".join(stale))
        sys.exit(1)
    print(("checked " if check else "wrote ") + f"{len(EXPORT)} schemas in {OUT}")


if __name__ == "__main__":
    main("--check" in sys.argv)

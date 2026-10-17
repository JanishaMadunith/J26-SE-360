"""Module 3 predictor.

BASELINE (v0.1.0): energy-aware greedy scheduler with a transparent logistic completion model
(src/m3/scheduling/heuristic.py).

NEXT: (1) swap the hand-set completion model for the trained one (src/m3/train_oulad.py → app features),
(2) add the OR-Tools CP-SAT optimiser (src/m3/scheduling/cpsat.py) behind the same interface and keep the
heuristic as the evaluation baseline.
"""

from m3.scheduling.heuristic import MODEL, schedule
from wellness_contracts import ModelInfo, ScheduleRequest, ScheduleResult


class Predictor:
    info: ModelInfo = MODEL

    def predict(self, req: ScheduleRequest) -> ScheduleResult:
        return schedule(req)

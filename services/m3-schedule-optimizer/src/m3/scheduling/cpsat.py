"""PLANNED (November): constraint-based optimiser with Google OR-Tools CP-SAT.

Not implemented yet — `ortools` is deliberately not a dependency until this module is built.
It will implement the same interface as heuristic.schedule(ScheduleRequest) -> ScheduleResult so the
service can switch with a setting, and the heuristic remains the comparison baseline.

Model sketch
------------
Decision variables (per flexible task session s, on the 15-minute grid from heuristic.build_grid):
  * start_s ∈ allowed cells (inside day windows, not busy, end <= deadline)  — IntVar
  * presence_s ∈ {0,1}                                                         — task may be dropped if infeasible
  * interval_s = OptionalIntervalVar(start_s, duration_s, presence_s)

Hard constraints
  * NoOverlap(all intervals + fixed tasks + busy blocks)
  * sessions of one task in order with a >= 15-minute gap; all-or-nothing presence per task
  * end_s <= deadline_s; start_s >= as_of
  * daily cap on scheduled flexible minutes, derived from forecast burnout risk (wellness-conditioned capacity)

Objective (maximise)
  Σ presence_s · priority_s · log p(complete | slot)   — p from the trained completion model, precomputed
                                                         per (session, start cell) as an integer table
  − λ1 · lateness/slack penalties − λ2 · fragmentation − λ3 · moves vs the current schedule (stability when
  regenerating, so the plan does not jump around)

Regeneration trigger (proactive): re-solve when a task is missed, a new task is added, or the latest
M2 forecast predicts overload (mean burnout risk > 0.7 or EDF demand > supply), warm-starting from the
current plan with AddHint.

Evaluation against the heuristic: expected completed priority-weighted tasks, deadline misses, overload
hours, schedule stability, and solve time (target < 2 s for a 7-day horizon).
"""

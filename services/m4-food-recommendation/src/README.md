# src/m4_food — research & training code (Module 4)

| File | Purpose |
|---|---|
| `foods.py` | **Done (baseline).** ~30 Sri Lankan and general foods with calories and tags. |
| `recommender.py` | **Done (baseline).** Meal windows around the M3 schedule, diet filters, context scoring, suggested activities. |
| `knowledge_base.py` | *To add.* Load a full nutrition database (Sri Lankan food composition tables, USDA FoodData Central). |
| `ranker.py` | *To add.* Learned ranking from user feedback (accepted / skipped meals). |
| `evaluate.py` | *To add.* Offline ranking metrics + user-satisfaction study analysis. |

Boundary with Module 3: M4 recommends **what** (meals, suggested activities); M3 alone decides **when**.
Suggested activities reach M3 only if the user accepts them (they become normal tasks).
Data never goes in the repo — read it from `DATA_DIR` (the `07-datasets` folder next to the repo).

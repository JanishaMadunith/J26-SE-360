# src/m1_affect — research & training code (Module 1)

Everything that is not request handling lives here, so the same code is used for training and serving.

| File | Purpose |
|---|---|
| `features.py` | **Done (baseline).** Extracts heart rate, HRV, typing rhythm, phone-use and activity features from raw signals. Reuse it when building training data. |
| `datasets.py` | *To add.* Load pilot data / public datasets (e.g. K-EmoPhone, SWELL-KW) into feature tables. |
| `train.py` | *To add.* Train the classifier; save the model to `../models/` with a `model_card.json`. |
| `evaluate.py` | *To add.* Subject-wise cross-validation, confusion matrix, per-class F1. |

Data never goes in the repo — read it from `DATA_DIR` (the `07-datasets` folder next to the repo).

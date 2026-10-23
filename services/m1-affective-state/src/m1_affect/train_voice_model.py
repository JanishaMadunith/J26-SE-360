"""
Voice Model Training
"""
from pathlib import Path

import joblib
import librosa
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Change this to your actual RAVDESS speech dataset folder.
DATASET_DIR = Path("/Users/rohanjayatilake/Desktop/affective-state-research-prototype/research/ravdess-project/data")

MODEL_PATH = Path("ravdess_mfcc_random_forest_5class.pkl")

# Keep this order identical to the Streamlit app.
CLASSES = ["Neutral", "Calm", "Happy", "Sad", "Angry"]

# RAVDESS emotion code -> class index in CLASSES.
EMOTION_MAP = {
    "01": 0,  # Neutral
    "02": 1,  # Calm
    "03": 2,  # Happy
    "04": 3,  # Sad
    "05": 4,  # Angry
}

SAMPLE_RATE = 22050
N_MFCC = 40


def extract_features(audio_path):
    audio, sr = librosa.load(
        str(audio_path),
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio.astype(np.float32),
        sr=sr,
        n_mfcc=N_MFCC
    )

    # 40 MFCC means + 40 MFCC standard deviations = 80 features.
    return np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])


X = []
y = []
skipped = 0

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR.resolve()}\n"
        "Update DATASET_DIR to your actual RAVDESS speech dataset path."
    )

for audio_path in DATASET_DIR.rglob("*.wav"):
    # RAVDESS filenames contain seven hyphen-separated fields.
    # The third field is the emotion code.
    parts = audio_path.stem.split("-")

    if len(parts) != 7:
        skipped += 1
        continue

    emotion_code = parts[2]

    # Exclude Fearful, Disgust, Surprised, and any other labels.
    if emotion_code not in EMOTION_MAP:
        skipped += 1
        continue

    try:
        features = extract_features(audio_path)

        if features.shape != (80,):
            print(f"Skipping unexpected feature shape: {audio_path}")
            skipped += 1
            continue

        X.append(features)
        y.append(EMOTION_MAP[emotion_code])

    except Exception as exc:
        print(f"Could not process {audio_path}: {exc}")
        skipped += 1


if not X:
    raise RuntimeError(
        "No usable WAV files found. Check DATASET_DIR and the dataset structure."
    )

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

print("Feature matrix shape:", X.shape)
print("Labels found:", sorted(np.unique(y).tolist()))
print("Class counts:")

for index, label in enumerate(CLASSES):
    print(f"  {label}: {int(np.sum(y == index))}")

if len(np.unique(y)) != 5:
    raise ValueError(
        "Training data must contain all five target classes. "
        "Check the dataset path and emotion codes."
    )

# Split the data while preserving class proportions.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Confirm that exactly five encoded labels were learned.
expected_classes = np.array([0, 1, 2, 3, 4])

if not np.array_equal(model.classes_, expected_classes):
    raise RuntimeError(
        f"Unexpected model classes: {model.classes_}"
    )

predictions = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=expected_classes,
        target_names=CLASSES,
        zero_division=0
    )
)

joblib.dump(model, MODEL_PATH)

print(f"\nSaved five-class model to: {MODEL_PATH.resolve()}")
print("Number of classes:", len(model.classes_))
print("Model classes:", model.classes_)

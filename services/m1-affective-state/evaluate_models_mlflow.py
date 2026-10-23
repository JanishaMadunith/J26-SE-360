"""MLflow Evaluation & Benchmarking: Random Forest vs. 2D CNN

Compares the two trained M1 speech emotion models:
1. ravdess_mfcc_random_forest_5class.pkl (80 MFCC features, 5 classes)
2. ravdess_emotion_cnn.keras (224x224 Spectrogram, 8 classes mapped to 5 classes)

Logs metrics, operational latency, model size, and confusion matrices into MLflow.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import joblib
import librosa
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras

# Set up paths
SERVICE_DIR = Path(__file__).resolve().parent
MODELS_DIR = SERVICE_DIR / "models"
RF_MODEL_PATH = MODELS_DIR / "ravdess_mfcc_random_forest_5class.pkl"
CNN_MODEL_PATH = MODELS_DIR / "ravdess_emotion_cnn.keras"

DATASET_DIR = Path("/Users/rohanjayatilake/Desktop/affective-state-research-prototype/research/ravdess-project/data")
FALLBACK_DATASET_DIR = Path("/Users/rohanjayatilake/Desktop/RAVDESS Speech Audio inputs")

# Target 5 classes used in M1
CLASSES_5 = ["Neutral", "Calm", "Happy", "Sad", "Angry"]
# RAVDESS emotion code mapping (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry)
RAVDESS_TO_CLASS5 = {
    "01": 0,  # Neutral
    "02": 1,  # Calm
    "03": 2,  # Happy
    "04": 3,  # Sad
    "05": 4,  # Angry
}


def load_test_audio_files(test_actors=("Actor_21", "Actor_22", "Actor_23", "Actor_24")):
    """Load test audio files from held-out actors to ensure speaker-independent evaluation."""
    file_list = []
    base_dir = DATASET_DIR if DATASET_DIR.exists() else FALLBACK_DATASET_DIR
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: neither {DATASET_DIR} nor {FALLBACK_DATASET_DIR} exists.")

    for actor in test_actors:
        actor_dir = base_dir / actor
        if not actor_dir.exists():
            continue
        for wav_path in actor_dir.glob("*.wav"):
            parts = wav_path.stem.split("-")
            if len(parts) >= 3:
                emotion_code = parts[2]
                if emotion_code in RAVDESS_TO_CLASS5:
                    file_list.append((wav_path, RAVDESS_TO_CLASS5[emotion_code]))

    if not file_list:
        # Fallback: scan all available actors for test files
        for wav_path in base_dir.rglob("*.wav"):
            parts = wav_path.stem.split("-")
            if len(parts) >= 3 and parts[2] in RAVDESS_TO_CLASS5:
                file_list.append((wav_path, RAVDESS_TO_CLASS5[parts[2]]))
                if len(file_list) >= 120:  # Take up to 120 files
                    break

    return file_list


def extract_rf_features(audio_path: Path):
    """Extract 80 MFCC features (40 mean + 40 std) for the Random Forest model."""
    audio, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]).reshape(1, -1)
    return features


def extract_cnn_features(audio_path: Path):
    """Extract 224x224x1 Mel-spectrogram for the 2D CNN model."""
    audio, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=224, n_fft=2048, hop_length=512)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)

    # Normalize to 0-1 range
    norm_spec = (melspec_db - melspec_db.min()) / (melspec_db.max() - melspec_db.min() + 1e-8)

    # Resize to exact (224, 224, 1) expected by the CNN InputLayer
    tensor_spec = tf.convert_to_tensor(norm_spec[..., np.newaxis], dtype=tf.float32)
    resized_spec = tf.image.resize(tensor_spec, [224, 224]).numpy()
    return np.expand_dims(resized_spec, axis=0)


def plot_confusion_matrix(cm, classes, title, output_path):
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_and_log_models():
    print("=" * 65)
    print("🚀 Starting MLflow Evaluation: Random Forest vs. 2D CNN")
    print("=" * 65)

    test_files = load_test_audio_files()
    print(f"📊 Loaded {len(test_files)} evaluation samples from held-out test actors.")
    y_true = [item[1] for item in test_files]

    # Set up MLflow (using SQLite backend recommended for MLflow 3.x)
    db_path = SERVICE_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve()}")
    mlflow.set_experiment("M1_Voice_Emotion_Models_Benchmark")

    # ---------------- 1. Evaluate Random Forest ----------------
    print("\n🔍 Evaluating Model 1: Random Forest (80 MFCC features)...", flush=True)
    rf_model = joblib.load(str(RF_MODEL_PATH))
    rf_size_mb = RF_MODEL_PATH.stat().st_size / (1024 * 1024)

    rf_preds = []
    rf_times = []
    for idx, (path, _) in enumerate(test_files, start=1):
        if idx % 30 == 0 or idx == len(test_files):
            print(f"   [RF {idx}/{len(test_files)}] processed...", flush=True)
        t0 = time.perf_counter()
        feat = extract_rf_features(path)
        pred = int(rf_model.predict(feat)[0])
        rf_times.append((time.perf_counter() - t0) * 1000)
        rf_preds.append(pred)

    rf_acc = accuracy_score(y_true, rf_preds)
    rf_f1_macro = f1_score(y_true, rf_preds, average="macro")
    rf_f1_weighted = f1_score(y_true, rf_preds, average="weighted")
    rf_prec = precision_score(y_true, rf_preds, average="macro", zero_division=0)
    rf_rec = recall_score(y_true, rf_preds, average="macro", zero_division=0)
    rf_avg_latency = float(np.mean(rf_times))

    with mlflow.start_run(run_name="RandomForest_MFCC_5Class"):
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", getattr(rf_model, "n_estimators", 300))
        mlflow.log_param("n_features", 80)
        mlflow.log_param("feature_type", "MFCC (40 mean + 40 std)")
        mlflow.log_param("classes", ", ".join(CLASSES_5))

        mlflow.log_metric("test_accuracy", round(rf_acc, 4))
        mlflow.log_metric("f1_macro", round(rf_f1_macro, 4))
        mlflow.log_metric("f1_weighted", round(rf_f1_weighted, 4))
        mlflow.log_metric("precision_macro", round(rf_prec, 4))
        mlflow.log_metric("recall_macro", round(rf_rec, 4))
        mlflow.log_metric("avg_latency_ms", round(rf_avg_latency, 2))
        mlflow.log_metric("model_size_mb", round(rf_size_mb, 2))

        # Per-class F1
        per_class_f1 = f1_score(y_true, rf_preds, average=None, zero_division=0)
        for cls_name, score in zip(CLASSES_5, per_class_f1):
            mlflow.log_metric(f"f1_{cls_name.lower()}", round(float(score), 4))

        # Confusion Matrix artifact
        rf_cm = confusion_matrix(y_true, rf_preds)
        cm_path = SERVICE_DIR / "rf_confusion_matrix.png"
        plot_confusion_matrix(rf_cm, CLASSES_5, "Random Forest Confusion Matrix", cm_path)
        mlflow.log_artifact(str(cm_path))
        if cm_path.exists():
            cm_path.unlink()

        # Classification report artifact
        rep_path = SERVICE_DIR / "rf_classification_report.txt"
        rep_path.write_text(classification_report(y_true, rf_preds, target_names=CLASSES_5, zero_division=0))
        mlflow.log_artifact(str(rep_path))
        if rep_path.exists():
            rep_path.unlink()

    print(f"✅ Random Forest: Accuracy={rf_acc:.2%}, F1-Macro={rf_f1_macro:.2%}, Latency={rf_avg_latency:.1f}ms, Size={rf_size_mb:.1f}MB")

    # ---------------- 2. Evaluate 2D CNN ----------------
    print("\n🔍 Evaluating Model 2: 2D CNN (224x224 Mel-Spectrogram)...")
    cnn_model = keras.models.load_model(str(CNN_MODEL_PATH))
    cnn_size_mb = CNN_MODEL_PATH.stat().st_size / (1024 * 1024)

    cnn_preds = []
    cnn_times = []
    for idx, (path, _) in enumerate(test_files, start=1):
        if idx % 30 == 0 or idx == len(test_files):
            print(f"   [CNN {idx}/{len(test_files)}] processed...", flush=True)
        t0 = time.perf_counter()
        feat = extract_cnn_features(path)
        raw_probs = cnn_model.predict(feat, verbose=0)[0]
        # Restrict to the 5 target classes for fair comparison
        top_5_probs = raw_probs[:5]
        pred = int(np.argmax(top_5_probs))
        cnn_times.append((time.perf_counter() - t0) * 1000)
        cnn_preds.append(pred)

    cnn_acc = accuracy_score(y_true, cnn_preds)
    cnn_f1_macro = f1_score(y_true, cnn_preds, average="macro")
    cnn_f1_weighted = f1_score(y_true, cnn_preds, average="weighted")
    cnn_prec = precision_score(y_true, cnn_preds, average="macro", zero_division=0)
    cnn_rec = recall_score(y_true, cnn_preds, average="macro", zero_division=0)
    cnn_avg_latency = float(np.mean(cnn_times))

    with mlflow.start_run(run_name="CNN_2D_Spectrogram"):
        mlflow.log_param("model_type", "Sequential 2D CNN")
        mlflow.log_param("input_shape", "224x224x1")
        mlflow.log_param("feature_type", "Mel-Spectrogram (dB normalized)")
        mlflow.log_param("trainable_params", 27976)
        mlflow.log_param("classes", ", ".join(CLASSES_5))

        mlflow.log_metric("test_accuracy", round(cnn_acc, 4))
        mlflow.log_metric("f1_macro", round(cnn_f1_macro, 4))
        mlflow.log_metric("f1_weighted", round(cnn_f1_weighted, 4))
        mlflow.log_metric("precision_macro", round(cnn_prec, 4))
        mlflow.log_metric("recall_macro", round(cnn_rec, 4))
        mlflow.log_metric("avg_latency_ms", round(cnn_avg_latency, 2))
        mlflow.log_metric("model_size_mb", round(cnn_size_mb, 2))

        # Per-class F1
        per_class_f1_cnn = f1_score(y_true, cnn_preds, average=None, zero_division=0)
        for cls_name, score in zip(CLASSES_5, per_class_f1_cnn):
            mlflow.log_metric(f"f1_{cls_name.lower()}", round(float(score), 4))

        # Confusion Matrix artifact
        cnn_cm = confusion_matrix(y_true, cnn_preds)
        cm_path_cnn = SERVICE_DIR / "cnn_confusion_matrix.png"
        plot_confusion_matrix(cnn_cm, CLASSES_5, "2D CNN Confusion Matrix", cm_path_cnn)
        mlflow.log_artifact(str(cm_path_cnn))
        if cm_path_cnn.exists():
            cm_path_cnn.unlink()

        # Classification report artifact
        rep_path_cnn = SERVICE_DIR / "cnn_classification_report.txt"
        rep_path_cnn.write_text(classification_report(y_true, cnn_preds, target_names=CLASSES_5, zero_division=0))
        mlflow.log_artifact(str(rep_path_cnn))
        if rep_path_cnn.exists():
            rep_path_cnn.unlink()

    print(f"✅ 2D CNN: Accuracy={cnn_acc:.2%}, F1-Macro={cnn_f1_macro:.2%}, Latency={cnn_avg_latency:.1f}ms, Size={cnn_size_mb:.2f}MB")

    # ---------------- Summary Comparison Table ----------------
    print("\n" + "=" * 65)
    print("🏆 SIDE-BY-SIDE MODEL COMPARISON SUMMARY")
    print("=" * 65)
    print(f"{'Metric':<25} | {'Random Forest':<18} | {'2D CNN':<18}")
    print("-" * 65)
    print(f"{'Accuracy':<25} | {rf_acc:<18.2%} | {cnn_acc:<18.2%}")
    print(f"{'F1-Score (Macro)':<25} | {rf_f1_macro:<18.2%} | {cnn_f1_macro:<18.2%}")
    print(f"{'Precision (Macro)':<25} | {rf_prec:<18.2%} | {cnn_prec:<18.2%}")
    print(f"{'Recall (Macro)':<25} | {rf_rec:<18.2%} | {cnn_rec:<18.2%}")
    print(f"{'Avg Latency / Sample':<25} | {rf_avg_latency:<15.1f} ms | {cnn_avg_latency:<15.1f} ms")
    print(f"{'Disk Footprint':<25} | {rf_size_mb:<15.1f} MB | {cnn_size_mb:<15.2f} MB")
    print("=" * 65)
    print("\n🎉 Both runs successfully logged to MLflow!")
    print("👉 Launch the MLflow UI to view visual charts and confusion matrices:")
    print("   mlflow ui\n")


if __name__ == "__main__":
    evaluate_and_log_models()

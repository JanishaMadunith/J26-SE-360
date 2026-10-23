"""
Model Traning Accuracy test
"""
import os
import joblib
import numpy as np
import librosa

# Model trained using your five-class RAVDESS dataset
MODEL_PATH = "ravdess_mfcc_random_forest_5class.pkl"

EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry"]

# Load the trained model
model = joblib.load(MODEL_PATH)

print("Five-class voice emotion model loaded!")
print("Model classes:", model.classes_)

# Enter the path to a RAVDESS .wav file
audio_path = input("\nEnter the path to a .wav audio file: ").strip()
audio_path = audio_path.strip('"').strip("'")

if not os.path.isfile(audio_path):
    raise FileNotFoundError(f"Audio file not found: {audio_path}")

# Use the same audio preprocessing as training
audio, sr = librosa.load(audio_path, sr=22050, mono=True)

# Extract the same 80 features: 40 MFCC means + 40 standard deviations
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

features = np.concatenate([
    np.mean(mfcc, axis=1),
    np.std(mfcc, axis=1)
]).reshape(1, -1)

# Predict emotion
prediction = int(model.predict(features)[0])
probabilities = model.predict_proba(features)[0]

print("\n--- Prediction Results ---")
print("Predicted emotion:", EMOTIONS[prediction])

print("\nClass probabilities:")
for class_id, probability in zip(model.classes_, probabilities):
    print(f"{EMOTIONS[int(class_id)]:<10}: {probability * 100:.2f}%")

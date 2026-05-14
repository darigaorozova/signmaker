"""
asl_classifier.py

Optional module — trains a custom Random Forest classifier on top of
MediaPipe hand landmarks.

In the current MVP the gesture recognition is done entirely in the browser
via MediaPipe Tasks JS (GestureRecognizer). This Python classifier is here
for future expansion: if you want to train on YOUR OWN custom gestures
(e.g. MEASURE, IRON, SEW) that MediaPipe doesn't know natively, you can:

  1. Run:  python calibrate_asl.py   (records gesture samples via webcam)
  2. The trained model is saved to:  asl_model.pkl
  3. app.py auto-loads it and uses it as a SECONDARY classifier on top of
     MediaPipe (custom gestures override built-in ones).

Dependencies (only needed for training / custom gestures):
    pip install mediapipe scikit-learn opencv-python
"""

import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path("asl_model.pkl")


def landmarks_to_features(landmarks_data: dict) -> np.ndarray:
    """
    Converts a {hand_0: [...21 points...], hand_1: [...]} dict
    into a flat numpy feature vector of length 126 (2 hands × 21 pts × 3 coords).
    Missing hands are zero-padded.
    """
    features = []
    for hand_idx in range(2):
        hand = landmarks_data.get(f"hand_{hand_idx}")
        if hand:
            for pt in hand:
                features.extend([pt.get("x", 0), pt.get("y", 0), pt.get("z", 0)])
        else:
            features.extend([0.0] * 63)   # 21 * 3
    return np.array(features, dtype=np.float32)


def train_asl_classifier(training_data: dict):
    """
    training_data = {
        "MEASURE": [ landmarks_dict_1, landmarks_dict_2, ... ],
        "CONNECT": [ ... ],
        ...
    }
    Trains a Random Forest and saves it to MODEL_PATH.
    Returns the trained model.
    """
    from sklearn.ensemble import RandomForestClassifier

    X, y = [], []
    for gesture, samples in training_data.items():
        for lm in samples:
            X.append(landmarks_to_features(lm))
            y.append(gesture)

    if not X:
        raise ValueError("No training samples provided!")

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved → {MODEL_PATH}")
    print(f"   Classes : {list(model.classes_)}")
    print(f"   Samples : {len(X)}")
    return model


def load_asl_classifier():
    """Loads and returns the saved model, or None if not found."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict_gesture(model, landmarks_data: dict) -> tuple[str, float]:
    """Returns (gesture_name, confidence) for given landmarks."""
    if model is None:
        return "UNKNOWN", 0.0
    feat = landmarks_to_features(landmarks_data).reshape(1, -1)
    pred  = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    return pred, float(np.max(proba))

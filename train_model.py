"""Train separate static and dynamic Random Forest classifiers."""
import csv
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import STATIC_FEATURE_COUNT, DYNAMIC_FEATURE_COUNT, DYNAMIC_FRAMES

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")


def load_csv(path, expected_features):
    if not os.path.exists(path):
        print(f"Dataset not found: {path}")
        return None, None
    labels, features = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(f"{path} must have a 'label' column.")
        feature_columns = [c for c in reader.fieldnames if c != "label"]
        if len(feature_columns) != expected_features:
            raise ValueError(
                f"{path} has {len(feature_columns)} features; expected {expected_features}. "
                "The dataset uses a different features.py schema. Back it up and "
                "recollect the affected dataset with the current files."
            )
        for row in reader:
            try:
                label = row["label"].strip()
                values = [float(row[c]) for c in feature_columns]
                if label and np.isfinite(values).all():
                    labels.append(label)
                    features.append(values)
            except (ValueError, TypeError):
                print(f"Skipping malformed row in {path}")
    if not features:
        return None, None
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def train_one(path, expected_features, output_path, title):
    X, y = load_csv(path, expected_features)
    if X is None:
        print(f"\nSkipping {title}: no usable data.")
        return

    classes, counts = np.unique(y, return_counts=True)
    print(f"\n{title}: {len(y)} samples, {len(classes)} classes")
    for cls, count in zip(classes, counts):
        print(f"  {cls}: {count} samples")
    if len(classes) < 2:
        print(f"Need at least 2 different labels to train {title}.")
        return
    if counts.min() < 2:
        print("Each class needs at least 2 samples for a stratified split.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            random_state=42, n_jobs=-1, min_samples_leaf=2
        )
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("\nHeld-out report (random row split; near-duplicate frames can overestimate accuracy):")
    print(classification_report(y_test, predictions, zero_division=0))
    labels = list(model.classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print("Confusion matrix (rows=true, columns=predicted), labels:", labels)
    print(matrix)

    confusions = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i != j and matrix[i, j] > 0:
                confusions.append((int(matrix[i, j]), true_label, predicted_label))
    if confusions:
        confusions.sort(reverse=True)
        print("Most common confusions:")
        for count, true_label, predicted_label in confusions[:8]:
            print(f"  {true_label} predicted as {predicted_label}: {count}")
    else:
        print("No cross-class errors in this split. Test fresh sessions too.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    bundle = {
        "model": model,
        "classes": labels,
        "feature_count": expected_features,
        "kind": title,
        "dynamic_frames": DYNAMIC_FRAMES if title == "dynamic" else None,
    }
    joblib.dump(bundle, output_path)
    print(f"Saved model: {output_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_one(
        os.path.join(DATA_DIR, "static_samples.csv"),
        STATIC_FEATURE_COUNT,
        os.path.join(MODEL_DIR, "static_model.joblib"),
        "static",
    )
    train_one(
        os.path.join(DATA_DIR, "dynamic_samples.csv"),
        DYNAMIC_FRAMES * DYNAMIC_FEATURE_COUNT,
        os.path.join(MODEL_DIR, "dynamic_model.joblib"),
        "dynamic",
    )


if __name__ == "__main__":
    main()

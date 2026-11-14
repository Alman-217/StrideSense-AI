
from pathlib import Path
import re
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from features import extract_features

DATA_DIR = Path("data/daphnet")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "stridesense_rf.joblib"

SAMPLE_RATE = 64
WINDOW_SIZE = 128       # 2 seconds
STRIDE = 32             # 0.5 seconds

def load_dataset():

    X = []
    y = []
    groups = []

    files = sorted(DATA_DIR.rglob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt recordings found in {DATA_DIR.resolve()}"
        )

    print(f"Found {len(files)} recording files.")

    for file in files:

        print("Processing:", file.name)

        try:
            data = np.loadtxt(file)
        except Exception as error:
            print("Could not load file:", error)
            continue

        if data.ndim != 2 or data.shape[1] != 11:
            print("Skipping file with unexpected format:", file)
            continue

        shank = data[:, 1:4]
        labels = data[:, 10].astype(int)

        magnitude = np.linalg.norm(shank, axis=1)

        subject_match = re.search(
            r"(S\d+)",
            file.stem,
            re.IGNORECASE
        )

        if not subject_match:
            print("Could not identify participant:", file.name)
            continue

        subject_id = subject_match.group(1).upper()

        for start in range(
            0,
            len(magnitude) - WINDOW_SIZE + 1,
            STRIDE
        ):

            end = start + WINDOW_SIZE

            signal_window = magnitude[start:end]
            label_window = labels[start:end]

            valid = np.isin(label_window, [1, 2])

            if np.mean(valid) < 0.95:
                continue

            valid_labels = label_window[valid]

            freeze_fraction = np.mean(valid_labels == 2)

            if freeze_fraction >= 0.80:
                target = 1

            elif freeze_fraction <= 0.20:
                target = 0

            else:
                continue

            feature_dict = extract_features(
                signal_window,
                SAMPLE_RATE
            )

            X.append(feature_dict)
            y.append(target)
            groups.append(subject_id)

    if not X:
        raise ValueError(
            "No valid windows generated. Check the dataset files."
        )

    feature_names = list(X[0].keys())

    X = np.asarray([
        [row[name] for name in feature_names]
        for row in X
    ], dtype=np.float64)

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    return X, y, groups, feature_names

def main():

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    X, y, groups, feature_names = load_dataset()

    participants = np.unique(groups)

    print("\nDATASET SUMMARY")
    print("----------------------------")
    print("Total windows:", len(X))
    print("Participants:", len(participants))
    print("Participants:", participants.tolist())

    class_counts = np.bincount(y, minlength=2)

    print("Non-freezing windows:", class_counts[0])
    print("Freezing windows:", class_counts[1])

    if len(participants) < 5:
        raise ValueError(
            "At least five participants are required for "
            "five-fold grouped cross-validation."
        )

    if len(np.unique(y)) != 2:
        raise ValueError("Both target classes are required.")

    model_config = {
        "n_estimators": 300,
        "max_depth": 14,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": 42,
        "n_jobs": -1
    }

    model_cv = GroupKFold(n_splits=5)

    out_of_fold_predictions = np.zeros(len(y), dtype=int)

    print("\nPARTICIPANT-LEVEL CROSS-VALIDATION")
    print("----------------------------")

    for fold, (train_idx, test_idx) in enumerate(
        model_cv.split(X, y, groups),
        start=1
    ):

        fold_model = RandomForestClassifier(**model_config)

        fold_model.fit(
            X[train_idx],
            y[train_idx]
        )

        predictions = fold_model.predict(X[test_idx])

        out_of_fold_predictions[test_idx] = predictions

        fold_f1 = f1_score(
            y[test_idx],
            predictions,
            zero_division=0
        )

        fold_balanced = balanced_accuracy_score(
            y[test_idx],
            predictions
        )

        test_participants = np.unique(groups[test_idx])

        print(f"\nFold {fold}")
        print("Held-out participants:", test_participants.tolist())
        print("F1:", round(fold_f1, 4))
        print("Balanced accuracy:", round(fold_balanced, 4))

    precision = precision_score(
        y,
        out_of_fold_predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        out_of_fold_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        out_of_fold_predictions,
        zero_division=0
    )

    balanced = balanced_accuracy_score(
        y,
        out_of_fold_predictions
    )

    matrix = confusion_matrix(
        y,
        out_of_fold_predictions,
        labels=[0, 1]
    )

    print("\nOVERALL OUT-OF-FOLD RESULTS")
    print("----------------------------")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("Balanced accuracy:", round(balanced, 4))

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(
        classification_report(
            y,
            out_of_fold_predictions,
            target_names=[
                "Non-freezing",
                "Freezing"
            ],
            zero_division=0
        )
    )

    final_model = RandomForestClassifier(**model_config)

    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "feature_names": feature_names,
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "input_unit": "mg",
        "target_mapping": {
            "1": "non-freezing",
            "2": "freezing"
        },
        "participants": participants.tolist(),
        "evaluation": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "balanced_accuracy": float(balanced)
        }
    }

    joblib.dump(bundle, MODEL_PATH)

    print("\nFINAL MODEL SAVED")
    print(MODEL_PATH.resolve())


if __name__ == "__main__":
    main()
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# =========================
# ŚCIEŻKI
# =========================
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\61x73x61_T100_subjectwise")
INDEX_PATH = DATASET_DIR / "subject_index.csv"

# =========================
# PARAMETRY
# =========================
MAX_SUBJECTS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42


def build_features_from_subject(npy_path: Path) -> np.ndarray:
    """
    Ładuje jednego subjecta o shape (61,73,61,100)
    i buduje lekkie cechy:
    - mean po czasie
    - std po czasie
    """
    x = np.load(npy_path)  # float32, shape (61,73,61,100)

    x_mean = x.mean(axis=3)   # (61,73,61)
    x_std = x.std(axis=3)     # (61,73,61)

    feat = np.concatenate([
        x_mean.ravel(),
        x_std.ravel()
    ]).astype(np.float32)

    return feat


def main():
    print("=== FULL-RES SUBJECTWISE BENCHMARK ===")
    print("Dataset:", DATASET_DIR)
    print("Index  :", INDEX_PATH)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono: {INDEX_PATH}")

    df = pd.read_csv(INDEX_PATH)
    print("\nLiczba wszystkich subjectów w indexie:", len(df))
    print("Rozkład klas:", Counter(df["label_binary"].tolist()))

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(df), size=MAX_SUBJECTS, replace=False)
    df_small = df.iloc[idx].reset_index(drop=True)

    print(f"\n=== PODZBIÓR {MAX_SUBJECTS} ===")
    print("Rozkład klas:", Counter(df_small["label_binary"].tolist()))

    X_list = []
    y_list = []

    for i, row in df_small.iterrows():
        npy_path = Path(row["saved_npy"])
        y = int(row["label_binary"])

        feat = build_features_from_subject(npy_path)

        X_list.append(feat)
        y_list.append(y)

        if i == 0:
            print("\nShape cech jednego subjecta:", feat.shape)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print("\n=== CECHY ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Klasy:", Counter(y.tolist()))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n=== SPLIT ===")
    print("X_train:", X_train.shape, Counter(y_train.tolist()))
    print("X_test :", X_test.shape, Counter(y_test.tolist()))

    clf = LogisticRegression(
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    probs_test = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    print("\n=== WYNIKI TEST ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, probs_test))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("\n=== PODGLĄD PRAWDOPODOBIEŃSTW ===")
    print("min prob:", float(probs_test.min()))
    print("max prob:", float(probs_test.max()))
    print("mean prob:", float(probs_test.mean()))
    print("probs_test:", probs_test)


if __name__ == "__main__":
    main()
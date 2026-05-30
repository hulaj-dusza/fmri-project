from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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
MAX_SUBJECTS = 400
N_SPLITS = 5
RANDOM_STATE = 42


def build_features_from_subject(npy_path: Path) -> np.ndarray:
    x = np.load(npy_path)  # shape (61,73,61,100)

    x_mean = x.mean(axis=3)
    x_std = x.std(axis=3)

    feat = np.concatenate([
        x_mean.ravel(),
        x_std.ravel()
    ]).astype(np.float32)

    return feat


def main():
    print("=== FULL-RES SUBJECTWISE CV ===")
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

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = []
    auc_scores = []

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== FOLD {fold}/{N_SPLITS} ===")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print("Train:", X_train.shape, Counter(y_train.tolist()))
        print("Test :", X_test.shape, Counter(y_test.tolist()))

        clf = LogisticRegression(
            solver="liblinear",
            max_iter=2000,
            random_state=RANDOM_STATE
        )
        clf.fit(X_train, y_train)

        probs_test = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, probs_test)

        acc_scores.append(acc)
        auc_scores.append(auc)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        print("Accuracy:", round(acc, 4))
        print("ROC AUC:", round(auc, 4))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

    print("\n=== PODSUMOWANIE CV ===")
    print("Accuracy foldy:", np.round(acc_scores, 4))
    print("ROC AUC foldy:", np.round(auc_scores, 4))
    print("Średnie Accuracy:", round(float(np.mean(acc_scores)), 4))
    print("Std Accuracy:", round(float(np.std(acc_scores)), 4))
    print("Średnie ROC AUC:", round(float(np.mean(auc_scores)), 4))
    print("Std ROC AUC:", round(float(np.std(auc_scores)), 4))

    print("\n=== RAPORT ZBIORCZY ZE WSZYSTKICH FOLDÓW ===")
    print(confusion_matrix(all_y_true, all_y_pred))
    print(classification_report(all_y_true, all_y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
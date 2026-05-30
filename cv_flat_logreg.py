from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# ===== ścieżki =====
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\32x32x32_T100")
X_PATH = DATASET_DIR / "X.npy"
Y_PATH = DATASET_DIR / "y.npy"

# ===== parametry =====
MAX_SUBJECTS = 200
N_SPLITS = 5
RANDOM_STATE = 42


def main():
    print("=== WCZYTYWANIE DANYCH ===")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH)

    print("Pełny X shape:", X.shape)
    print("Pełny y shape:", y.shape)
    print("Pełne klasy:", Counter(y.tolist()))

    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y), size=MAX_SUBJECTS, replace=False)

    X_small = np.asarray(X[idx], dtype=np.float32)
    y_small = y[idx]

    print("\n=== PODZBIÓR ===")
    print("X_small shape:", X_small.shape)
    print("y_small shape:", y_small.shape)
    print("Klasy:", Counter(y_small.tolist()))

    X_flat = X_small.reshape(X_small.shape[0], -1)
    print("\nPo spłaszczeniu:", X_flat.shape)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = []
    auc_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_flat, y_small), start=1):
        print(f"\n=== FOLD {fold}/{N_SPLITS} ===")

        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y_small[train_idx], y_small[test_idx]

        print("Train:", X_train.shape, Counter(y_train.tolist()))
        print("Test :", X_test.shape, Counter(y_test.tolist()))

        clf = LogisticRegression(
            penalty="l2",
            C=1.0,
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

    # raport na ostatnim foldzie opcjonalnie
    print("\nUwaga: classification report nie jest tu agregowany po wszystkich foldach.")
    print("Jeśli chcesz, w kolejnym kroku mogę dodać pełne zbiorcze podsumowanie predykcji ze wszystkich foldów.")


if __name__ == "__main__":
    main()
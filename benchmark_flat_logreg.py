from pathlib import Path
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# ===== ścieżki =====
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\32x32x32_T100")
X_PATH = DATASET_DIR / "X.npy"
Y_PATH = DATASET_DIR / "y.npy"

# ===== parametry =====
MAX_SUBJECTS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42

def main():
    print("=== Wczytywanie datasetu ===")
    X = np.load(X_PATH, mmap_mode="r")
    y = np.load(Y_PATH)

    print("Pełny X shape:", X.shape)
    print("Pełny y shape:", y.shape)
    print("Pełne klasy:", Counter(y.tolist()))

    # losowy podzbiór
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y), size=MAX_SUBJECTS, replace=False)

    X_small = np.asarray(X[idx], dtype=np.float32)
    y_small = y[idx]

    print("\n=== PODZBIÓR TESTOWY ===")
    print("X_small shape:", X_small.shape)
    print("y_small shape:", y_small.shape)
    print("Klasy w podzbiorze:", Counter(y_small.tolist()))

    # flatten
    X_small_flat = X_small.reshape(X_small.shape[0], -1)

    print("\n=== PO SPŁASZCZENIU ===")
    print("X_small_flat shape:", X_small_flat.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X_small_flat,
        y_small,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_small
    )

    print("\n=== SPLIT ===")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train:", Counter(y_train.tolist()))
    print("y_test:", Counter(y_test.tolist()))

    print("\n=== TRENING LOGISTIC REGRESSION ===")
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
    print("probs_test[:10]:", probs_test[:10])

if __name__ == "__main__":
    main()
    
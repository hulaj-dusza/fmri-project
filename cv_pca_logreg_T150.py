from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

# ===== ścieżki =====
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\32x32x32_T150")
X_PATH = DATASET_DIR / "X.npy"
Y_PATH = DATASET_DIR / "y.npy"

# ===== parametry =====
MAX_SUBJECTS = 627
N_SPLITS = 5
RANDOM_STATE = 42

N_COMPONENTS = 50
BATCH_SIZE = 128

def fit_incremental_pca(X_train_flat, n_components=50, batch_size=32):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    n_samples = X_train_flat.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        ipca.partial_fit(X_train_flat[start:end])

    return ipca


def transform_in_batches(ipca, X_flat, batch_size=32):
    parts = []
    n_samples = X_flat.shape[0]
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        parts.append(ipca.transform(X_flat[start:end]))
    return np.vstack(parts)


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

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_flat, y_small), start=1):
        print(f"\n=== FOLD {fold}/{N_SPLITS} ===")

        X_train_flat = X_flat[train_idx]
        X_test_flat = X_flat[test_idx]
        y_train = y_small[train_idx]
        y_test = y_small[test_idx]

        print("Train:", X_train_flat.shape, Counter(y_train.tolist()))
        print("Test :", X_test_flat.shape, Counter(y_test.tolist()))

        # scaler
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_test_scaled = scaler.transform(X_test_flat)

        # PCA
        print("Fitting IncrementalPCA...")
        ipca = fit_incremental_pca(
            X_train_scaled,
            n_components=N_COMPONENTS,
            batch_size=BATCH_SIZE
        )

        X_train_pca = transform_in_batches(ipca, X_train_scaled, batch_size=BATCH_SIZE)
        X_test_pca = transform_in_batches(ipca, X_test_scaled, batch_size=BATCH_SIZE)

        print("Po PCA train:", X_train_pca.shape)
        print("Po PCA test :", X_test_pca.shape)

        # logistic regression
        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE
        )
        clf.fit(X_train_pca, y_train)

        probs_test = clf.predict_proba(X_test_pca)[:, 1]
        y_pred = clf.predict(X_test_pca)

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
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from scipy.special import expit

# ===== ścieżki =====
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\32x32x32_T100")
X_PATH = DATASET_DIR / "X.npy"
Y_PATH = DATASET_DIR / "y.npy"

# ===== parametry =====
MAX_SUBJECTS = 200
N_SPLITS = 5
RANDOM_STATE = 42

RANK = 3
N_ITER = 20
LR = 1e-2
SEED = 123


def cp_to_tensor_4d(A, B, C, D):
    p1, R = A.shape
    p2, _ = B.shape
    p3, _ = C.shape
    p4, _ = D.shape

    B_tensor = np.zeros((p1, p2, p3, p4), dtype=np.float32)
    for r in range(R):
        B_tensor += np.einsum("i,j,k,l->ijkl", A[:, r], B[:, r], C[:, r], D[:, r])
    return B_tensor


def predict_logits_cp(X, A, B, C, D, b0):
    B_tensor = cp_to_tensor_4d(A, B, C, D)
    logits = np.einsum("nijkl,ijkl->n", X, B_tensor) + b0
    return logits


def fit_cp_logistic_simple(X, y, rank=3, n_iter=20, lr=1e-2, seed=123, l2=0.0, verbose=False):
    rng = np.random.default_rng(seed)

    n, p1, p2, p3, p4 = X.shape
    R = rank

    # większa inicjalizacja niż na początku eksperymentów
    A = (rng.standard_normal((p1, R)) * 0.1).astype(np.float32)
    B = (rng.standard_normal((p2, R)) * 0.1).astype(np.float32)
    C = (rng.standard_normal((p3, R)) * 0.1).astype(np.float32)
    D = (rng.standard_normal((p4, R)) * 0.1).astype(np.float32)

    p = float(y.mean())
    p = min(max(p, 1e-6), 1 - 1e-6)
    b0 = float(np.log(p / (1 - p)))

    history = []

    for it in range(n_iter):
        logits = predict_logits_cp(X, A, B, C, D, b0)
        probs = expit(logits)

        eps = 1e-8
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        reg = 0.5 * l2 * (
            np.sum(A ** 2) + np.sum(B ** 2) + np.sum(C ** 2) + np.sum(D ** 2)
        )
        total_loss = loss + reg
        history.append(float(total_loss))

        residual = probs - y
        grad_full = np.einsum("n,nijkl->ijkl", residual, X) / n

        grad_A = np.zeros_like(A)
        grad_B = np.zeros_like(B)
        grad_C = np.zeros_like(C)
        grad_D = np.zeros_like(D)

        for r in range(R):
            a = A[:, r]
            b = B[:, r]
            c = C[:, r]
            d = D[:, r]

            grad_A[:, r] = np.einsum("ijkl,j,k,l->i", grad_full, b, c, d) + l2 * a
            grad_B[:, r] = np.einsum("ijkl,i,k,l->j", grad_full, a, c, d) + l2 * b
            grad_C[:, r] = np.einsum("ijkl,i,j,l->k", grad_full, a, b, d) + l2 * c
            grad_D[:, r] = np.einsum("ijkl,i,j,k->l", grad_full, a, b, c) + l2 * d

        grad_b0 = float(residual.mean())

        A -= lr * grad_A
        B -= lr * grad_B
        C -= lr * grad_C
        D -= lr * grad_D
        b0 -= lr * grad_b0

        if verbose:
            print(f"Iteracja {it+1}/{n_iter}, loss={total_loss:.6f}")

    return A, B, C, D, b0, history


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

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    acc_scores = []
    auc_scores = []

    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_small, y_small), start=1):
        print(f"\n=== FOLD {fold}/{N_SPLITS} ===")

        X_train = X_small[train_idx]
        X_test = X_small[test_idx]
        y_train = y_small[train_idx]
        y_test = y_small[test_idx]

        print("Train:", X_train.shape, Counter(y_train.tolist()))
        print("Test :", X_test.shape, Counter(y_test.tolist()))

        A, B, C, D, b0, history = fit_cp_logistic_simple(
            X_train,
            y_train,
            rank=RANK,
            n_iter=N_ITER,
            lr=LR,
            seed=SEED + fold,   # lekko zmieniamy seed między foldami
            l2=0.0,
            verbose=False
        )

        print("loss start:", round(history[0], 6))
        print("loss end:", round(history[-1], 6))
        print("loss diff:", round(history[-1] - history[0], 6))

        logits_test = predict_logits_cp(X_test, A, B, C, D, b0)
        probs_test = expit(logits_test)
        y_pred = (probs_test >= 0.5).astype(int)

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
import os
import re
import gc
import time
import glob
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix
)


# =========================================================
# ŚCIEŻKI
# =========================================================

DATA_DIR = r"C:\Users\HP\Searches\fmri_datasets\full_voxel\61x73x61_T100_subjectwise\subjects"

# Jeżeli masz subject_index.csv, kod użyje go automatycznie.
INDEX_CSV = os.path.join(DATA_DIR, "subject_index.csv")

# Plik phenotypic ABIDE — potrzebny, jeśli subject_index.csv nie ma etykiet.
PHENO_CSV = r"C:\Users\HP\Searches\abide_data\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv"

OUT_DIR = "results_cp_logistic_block_glm_subjectwise"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(
    OUT_DIR,
    
    "results_cp_logistic_block_glm_61x73x61_T100_subjectwise.csv"
)


# =========================================================
# PARAMETRY EKSPERYMENTU
# =========================================================

RANKS = [1, 2, 3]

TEST_SIZE = 0.2
RANDOM_STATE = 42

N_EPOCHS = 5

# W sklearn większe C = słabsza regularyzacja.
# Mniejsze C = mocniejsza regularyzacja.
C_L2 = 1.0

NORMALIZE_SUBJECT = True

USE_SMOOTHING = True
SMOOTH_STRENGTH = 0.10

EPS = 1e-8
EXPECTED_SHAPE = (61, 73, 61, 100)


# =========================================================
# WCZYTYWANIE LISTY SUBJECTÓW
# =========================================================

def find_npy_files(data_dir):
    """
    Szuka plików .npy:
    1. w DATA_DIR/subjects
    2. bezpośrednio w DATA_DIR
    """
    candidates = []

    subjects_dir = os.path.join(data_dir, "subjects")

    if os.path.isdir(subjects_dir):
        candidates.extend(glob.glob(os.path.join(subjects_dir, "*.npy")))

    candidates.extend(glob.glob(os.path.join(data_dir, "*.npy")))

    candidates = sorted(list(set(candidates)))

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"Nie znaleziono plików .npy w: {data_dir}"
        )

    return candidates


def extract_sub_id_from_filename(path):
    """
    Próbuje wyciągnąć SUB_ID z nazwy pliku.

    Przykład:
    0730_USM_0050438.npy -> 50438

    W ABIDE SUB_ID w CSV często jest liczbą typu 50438,
    a w nazwie pliku bywa z zerami wiodącymi.
    """
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]

    numbers = re.findall(r"\d+", stem)

    if len(numbers) == 0:
        return None

    # zwykle ostatnia liczba to SUB_ID
    sub_id_raw = numbers[-1]

    try:
        return int(sub_id_raw)
    except ValueError:
        return None


def remap_dx_group_to_label(dx):
    """
    ABIDE DX_GROUP:
    1 = ASD
    2 = CONTROL

    Nasze etykiety:
    1 = ASD
    0 = CONTROL
    """
    if dx == 1:
        return 1
    if dx == 2:
        return 0

    raise ValueError(f"Nieznana wartość DX_GROUP: {dx}")


def load_labels_from_phenotypic(paths, pheno_csv):
    """
    Dopasowuje etykiety na podstawie SUB_ID wyciągniętego z nazwy pliku.
    """
    pheno = pd.read_csv(pheno_csv)

    if "SUB_ID" not in pheno.columns:
        raise ValueError("W pliku phenotypic brakuje kolumny SUB_ID.")

    if "DX_GROUP" not in pheno.columns:
        raise ValueError("W pliku phenotypic brakuje kolumny DX_GROUP.")

    pheno = pheno.copy()
    pheno["SUB_ID_INT"] = pheno["SUB_ID"].astype(int)

    sub_to_dx = dict(zip(pheno["SUB_ID_INT"], pheno["DX_GROUP"]))

    labels = []
    kept_paths = []
    missing = []

    for path in paths:
        sub_id = extract_sub_id_from_filename(path)

        if sub_id is None or sub_id not in sub_to_dx:
            missing.append(path)
            continue

        dx = int(sub_to_dx[sub_id])
        y = remap_dx_group_to_label(dx)

        kept_paths.append(path)
        labels.append(y)

    print(f"Dopasowano etykiety dla {len(kept_paths)} plików.")
    print(f"Brak etykiety dla {len(missing)} plików.")

    if len(kept_paths) == 0:
        raise RuntimeError(
            "Nie udało się dopasować żadnych etykiet. "
            "Sprawdź nazwy plików subjectów i kolumnę SUB_ID w phenotypic CSV."
        )

    return kept_paths, np.asarray(labels, dtype=int)


def auto_find_column(df, candidates):
    """
    Znajduje pierwszą pasującą kolumnę z listy kandydatów.
    """
    cols_lower = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    return None


def load_subject_table(data_dir, index_csv, pheno_csv):
    """
    Zwraca:
    paths: lista ścieżek do plików .npy
    y: etykiety 0/1

    Działa w dwóch wariantach:

    Wariant A:
    subject_index.csv ma kolumny z path oraz y/label/DX_GROUP.

    Wariant B:
    Nie ma etykiet w subject_index.csv albo nie ma indexu.
    Wtedy kod bierze pliki .npy z folderu i dopasowuje etykiety z PHENO_CSV.
    """

    # -----------------------------------------------------
    # Wariant A: subject_index.csv istnieje
    # -----------------------------------------------------
    if os.path.exists(index_csv):
        print("Znaleziono subject_index.csv:", index_csv)

        df = pd.read_csv(index_csv)
        print("Kolumny subject_index.csv:", list(df.columns))

        path_col = auto_find_column(
            df,
            ["path", "filepath", "file_path", "npy_path", "subject_path", "filename", "file"]
        )

        label_col = auto_find_column(
            df,
            ["y", "label", "target", "dx", "DX_GROUP", "diagnosis"]
        )

        if path_col is not None:
            raw_paths = df[path_col].astype(str).tolist()

            paths = []
            for p in raw_paths:
                if os.path.isabs(p):
                    paths.append(p)
                else:
                    paths.append(os.path.join(data_dir, p))

            # Jeżeli ścieżki nie istnieją, spróbuj dodać folder subjects
            fixed_paths = []
            for p in paths:
                if os.path.exists(p):
                    fixed_paths.append(p)
                else:
                    alt = os.path.join(data_dir, "subjects", os.path.basename(p))
                    fixed_paths.append(alt)

            paths = fixed_paths

            existing_mask = np.array([os.path.exists(p) for p in paths])

            if existing_mask.sum() == 0:
                print("Nie znaleziono plików ze ścieżek w subject_index.csv.")
            else:
                paths = [p for p, ok in zip(paths, existing_mask) if ok]
                df_existing = df.loc[existing_mask].copy()

                if label_col is not None:
                    raw_y = df_existing[label_col].values

                    # Jeśli kolumna to DX_GROUP z ABIDE: 1=ASD, 2=CONTROL
                    if set(np.unique(raw_y).tolist()) == {1, 2}:
                        y = np.array([remap_dx_group_to_label(int(v)) for v in raw_y], dtype=int)
                    else:
                        y = raw_y.astype(int)

                    print(f"Wczytano {len(paths)} subjectów z etykietami z subject_index.csv.")
                    return paths, y

                else:
                    print("subject_index.csv ma ścieżki, ale nie ma etykiet.")
                    return load_labels_from_phenotypic(paths, pheno_csv)

    # -----------------------------------------------------
    # Wariant B: bierzemy wszystkie .npy z folderu
    # -----------------------------------------------------
    print("Nie użyto subject_index.csv z etykietami. Szukam plików .npy w folderze.")

    paths = find_npy_files(data_dir)
    print(f"Znaleziono plików .npy: {len(paths)}")

    return load_labels_from_phenotypic(paths, pheno_csv)


# =========================================================
# OPERACJE NA SUBJECTACH
# =========================================================

def load_subject(path):
    """
    Wczytuje jednego subjecta.

    Używamy mmap_mode='r', żeby nie kopiować niepotrzebnie całego pliku.
    Potem astype(np.float32) robi realną tablicę dla obliczeń.
    """
    x = np.load(path, mmap_mode="r")

    if x.shape != EXPECTED_SHAPE:
        raise ValueError(
            f"Nieprawidłowy shape w pliku:\n{path}\n"
            f"Oczekiwano {EXPECTED_SHAPE}, jest {x.shape}"
        )

    x = np.asarray(x, dtype=np.float32)

    if NORMALIZE_SUBJECT:
        mean = x.mean()
        std = x.std() + EPS
        x = (x - mean) / std

    return x


def check_first_subject(paths):
    x = np.load(paths[0], mmap_mode="r")
    print("Pierwszy plik:", paths[0])
    print("Shape pierwszego subjecta:", x.shape)
    print("dtype:", x.dtype)
    print("Rozmiar jednego subjecta w MB:", x.size * np.dtype(x.dtype).itemsize / 1024**2)


# =========================================================
# CP: INICJALIZACJA, NORMALIZACJA, SMOOTHING
# =========================================================

def initialize_cp_factors(shape, rank, seed=42):
    rng = np.random.default_rng(seed)

    p1, p2, p3, p4 = shape
    scale = 0.05

    A = scale * rng.standard_normal((p1, rank))
    B = scale * rng.standard_normal((p2, rank))
    C = scale * rng.standard_normal((p3, rank))
    D = scale * rng.standard_normal((p4, rank))

    return A, B, C, D


def normalize_cp_factors(A, B, C, D):
    rank = A.shape[1]

    for r in range(rank):
        nA = np.linalg.norm(A[:, r]) + EPS
        nB = np.linalg.norm(B[:, r]) + EPS
        nC = np.linalg.norm(C[:, r]) + EPS
        nD = np.linalg.norm(D[:, r]) + EPS

        total = nA * nB * nC * nD
        target = total ** 0.25

        A[:, r] = A[:, r] / nA * target
        B[:, r] = B[:, r] / nB * target
        C[:, r] = C[:, r] / nC * target
        D[:, r] = D[:, r] / nD * target

    return A, B, C, D


def smooth_factor(F, strength=0.10):
    if strength <= 0 or F.shape[0] < 3:
        return F

    F_new = F.copy()

    local_mean = (F[:-2, :] + F[1:-1, :] + F[2:, :]) / 3.0
    F_new[1:-1, :] = (1.0 - strength) * F[1:-1, :] + strength * local_mean

    return F_new


def smooth_all_factors(A, B, C, D, strength=0.10):
    A = smooth_factor(A, strength)
    B = smooth_factor(B, strength)
    C = smooth_factor(C, strength)
    D = smooth_factor(D, strength)
    return A, B, C, D


def count_dense_params(shape):
    return int(np.prod(shape)) + 1


def count_cp_params(shape, rank):
    return rank * sum(shape) + 1


# =========================================================
# CECHY BLOKOWE
# =========================================================

def block_features_for_A(x, B, C, D):
    """
    Aktualizacja A:
    Z_A[x, r] = sum_{y,z,t} X[x,y,z,t] B[y,r] C[z,r] D[t,r]
    """
    Z = np.einsum("xyzt,yr,zr,tr->xr", x, B, C, D, optimize=True)
    return Z.reshape(-1)


def block_features_for_B(x, A, C, D):
    Z = np.einsum("xyzt,xr,zr,tr->yr", x, A, C, D, optimize=True)
    return Z.reshape(-1)


def block_features_for_C(x, A, B, D):
    Z = np.einsum("xyzt,xr,yr,tr->zr", x, A, B, D, optimize=True)
    return Z.reshape(-1)


def block_features_for_D(x, A, B, C):
    Z = np.einsum("xyzt,xr,yr,zr->tr", x, A, B, C, optimize=True)
    return Z.reshape(-1)


def build_design_matrix_for_block(paths, indices, block_name, A, B, C, D):
    """
    Buduje macierz cech dla jednego bloku CP.

    Dla R=2 i shape=(61,73,61,100):
    A: n x 122
    B: n x 146
    C: n x 122
    D: n x 200
    """

    n = len(indices)
    rank = A.shape[1]

    p1 = A.shape[0]
    p2 = B.shape[0]
    p3 = C.shape[0]
    p4 = D.shape[0]

    if block_name == "A":
        n_features = p1 * rank
    elif block_name == "B":
        n_features = p2 * rank
    elif block_name == "C":
        n_features = p3 * rank
    elif block_name == "D":
        n_features = p4 * rank
    else:
        raise ValueError(f"Nieznany blok: {block_name}")

    Z = np.zeros((n, n_features), dtype=np.float64)

    for row_id, idx in enumerate(indices):
        path = paths[idx]
        x = load_subject(path)

        if block_name == "A":
            Z[row_id, :] = block_features_for_A(x, B, C, D)
        elif block_name == "B":
            Z[row_id, :] = block_features_for_B(x, A, C, D)
        elif block_name == "C":
            Z[row_id, :] = block_features_for_C(x, A, B, D)
        elif block_name == "D":
            Z[row_id, :] = block_features_for_D(x, A, B, C)

        del x

        if (row_id + 1) % 25 == 0:
            print(f"    przetworzono {row_id + 1}/{n} subjectów")

    gc.collect()
    return Z


# =========================================================
# REGRESJA LOGISTYCZNA DLA JEDNEGO BLOKU
# =========================================================

def standardize_design_matrix(Z):
    mean = Z.mean(axis=0)
    std = Z.std(axis=0) + EPS
    Zs = (Z - mean) / std
    return Zs, mean, std


def fit_logistic_block(Z, y, C_l2=1.0):
    Zs, mean, std = standardize_design_matrix(Z)

    clf = LogisticRegression(
        penalty="l2",
        C=C_l2,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000
    )

    clf.fit(Zs, y)

    coef_scaled = clf.coef_.reshape(-1)
    intercept_scaled = float(clf.intercept_[0])

    coef_original = coef_scaled / std
    intercept_original = intercept_scaled - np.sum(mean * coef_scaled / std)

    return coef_original, intercept_original


def update_block(paths, train_idx, y_train, block_name, A, B, C, D, C_l2=1.0):
    print(f"  Blok {block_name}: buduję macierz cech...")

    Z = build_design_matrix_for_block(
        paths=paths,
        indices=train_idx,
        block_name=block_name,
        A=A,
        B=B,
        C=C,
        D=D
    )

    print(f"  Z_{block_name}.shape = {Z.shape}")

    coef, bias = fit_logistic_block(Z, y_train, C_l2=C_l2)

    if block_name == "A":
        A = coef.reshape(A.shape)
    elif block_name == "B":
        B = coef.reshape(B.shape)
    elif block_name == "C":
        C = coef.reshape(C.shape)
    elif block_name == "D":
        D = coef.reshape(D.shape)
    else:
        raise ValueError(f"Nieznany blok: {block_name}")

    del Z
    gc.collect()

    return A, B, C, D, bias


# =========================================================
# PREDYKCJA I EWALUACJA
# =========================================================

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def predict_logit_one_subject(x, A, B, C, D, bias):
    comp = np.einsum(
        "xyzt,xr,yr,zr,tr->r",
        x,
        A,
        B,
        C,
        D,
        optimize=True
    )

    return float(comp.sum() + bias)


def predict_proba(paths, indices, A, B, C, D, bias):
    logits = np.zeros(len(indices), dtype=np.float64)

    for row_id, idx in enumerate(indices):
        x = load_subject(paths[idx])
        logits[row_id] = predict_logit_one_subject(x, A, B, C, D, bias)
        del x

    probs = sigmoid(logits)
    return probs, logits


def evaluate(paths, indices, y_true, A, B, C, D, bias):
    probs, logits = predict_proba(paths, indices, A, B, C, D, bias)

    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sens_asd = tp / (tp + fn + EPS)
    spec_control = tn / (tn + fp + EPS)

    return {
        "acc": acc,
        "balanced_acc": bal_acc,
        "auc": auc,
        "sensitivity_ASD": sens_asd,
        "specificity_CONTROL": spec_control,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred_0": int((y_pred == 0).sum()),
        "pred_1": int((y_pred == 1).sum()),
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std())
    }


# =========================================================
# EKSPERYMENT
# =========================================================

def run_experiment():
    print("=" * 80)
    print("CP tensor logistic regression - subjectwise full voxel")
    print("=" * 80)

    paths, y = load_subject_table(DATA_DIR, INDEX_CSV, PHENO_CSV)

    paths = np.asarray(paths)
    y = np.asarray(y, dtype=int)

    check_first_subject(paths)

    print("Liczba subjectów:", len(paths))
    print("Rozkład klas:", np.bincount(y))

    voxel_shape = EXPECTED_SHAPE

    print("voxel shape:", voxel_shape)
    print("dense params:", count_dense_params(voxel_shape))

    n_subjects = len(paths)
    all_indices = np.arange(n_subjects)

    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("train size:", len(train_idx))
    print("test size :", len(test_idx))
    print("train class counts:", np.bincount(y_train))
    print("test class counts :", np.bincount(y_test))

    results = []

    for rank in RANKS:
        print("\n" + "=" * 80)
        print(f"RANK = {rank}")
        print("=" * 80)

        cp_params = count_cp_params(voxel_shape, rank)
        dense_params = count_dense_params(voxel_shape)

        print("CP params:", cp_params)
        print("dense params:", dense_params)
        print("compression:", dense_params / cp_params)

        A, B, C, D = initialize_cp_factors(
            shape=voxel_shape,
            rank=rank,
            seed=RANDOM_STATE + rank
        )

        bias = 0.0
        start_time = time.time()

        for epoch in range(1, N_EPOCHS + 1):
            print("\n" + "-" * 80)
            print(f"R={rank} | epoka {epoch}/{N_EPOCHS}")
            print("-" * 80)

            for block_name in ["A", "B", "C", "D"]:
                block_start = time.time()

                A, B, C, D, bias = update_block(
                    paths=paths,
                    train_idx=train_idx,
                    y_train=y_train,
                    block_name=block_name,
                    A=A,
                    B=B,
                    C=C,
                    D=D,
                    C_l2=C_L2
                )

                print(
                    f"  Zaktualizowano blok {block_name} | "
                    f"bias={bias:.6f} | "
                    f"czas={time.time() - block_start:.1f}s"
                )

            A, B, C, D = normalize_cp_factors(A, B, C, D)

            if USE_SMOOTHING:
                A, B, C, D = smooth_all_factors(
                    A, B, C, D,
                    strength=SMOOTH_STRENGTH
                )
                A, B, C, D = normalize_cp_factors(A, B, C, D)

            print("  Ewaluacja train/test...")

            train_metrics = evaluate(
                paths=paths,
                indices=train_idx,
                y_true=y_train,
                A=A,
                B=B,
                C=C,
                D=D,
                bias=bias
            )

            test_metrics = evaluate(
                paths=paths,
                indices=test_idx,
                y_true=y_test,
                A=A,
                B=B,
                C=C,
                D=D,
                bias=bias
            )

            print(
                f"TRAIN | "
                f"acc={train_metrics['acc']:.4f} | "
                f"bal_acc={train_metrics['balanced_acc']:.4f} | "
                f"auc={train_metrics['auc']:.4f} | "
                f"sens_ASD={train_metrics['sensitivity_ASD']:.4f} | "
                f"spec_CONTROL={train_metrics['specificity_CONTROL']:.4f} | "
                f"pred0={train_metrics['pred_0']} | "
                f"pred1={train_metrics['pred_1']} | "
                f"prob_mean={train_metrics['prob_mean']:.4f} | "
                f"prob_std={train_metrics['prob_std']:.4f}"
            )

            print(
                f"TEST  | "
                f"acc={test_metrics['acc']:.4f} | "
                f"bal_acc={test_metrics['balanced_acc']:.4f} | "
                f"auc={test_metrics['auc']:.4f} | "
                f"sens_ASD={test_metrics['sensitivity_ASD']:.4f} | "
                f"spec_CONTROL={test_metrics['specificity_CONTROL']:.4f} | "
                f"pred0={test_metrics['pred_0']} | "
                f"pred1={test_metrics['pred_1']} | "
                f"prob_mean={test_metrics['prob_mean']:.4f} | "
                f"prob_std={test_metrics['prob_std']:.4f}"
            )

            row = {
                "R": rank,
                "epoch": epoch,
                "CP_params": cp_params,
                "dense_params": dense_params,
                "compression": dense_params / cp_params,
                "C_L2": C_L2,
                "use_smoothing": USE_SMOOTHING,
                "smooth_strength": SMOOTH_STRENGTH,
                "train_acc": train_metrics["acc"],
                "train_balanced_acc": train_metrics["balanced_acc"],
                "train_auc": train_metrics["auc"],
                "train_sensitivity_ASD": train_metrics["sensitivity_ASD"],
                "train_specificity_CONTROL": train_metrics["specificity_CONTROL"],
                "train_pred_0": train_metrics["pred_0"],
                "train_pred_1": train_metrics["pred_1"],
                "train_prob_mean": train_metrics["prob_mean"],
                "train_prob_std": train_metrics["prob_std"],
                "test_acc": test_metrics["acc"],
                "test_balanced_acc": test_metrics["balanced_acc"],
                "test_auc": test_metrics["auc"],
                "test_sensitivity_ASD": test_metrics["sensitivity_ASD"],
                "test_specificity_CONTROL": test_metrics["specificity_CONTROL"],
                "test_pred_0": test_metrics["pred_0"],
                "test_pred_1": test_metrics["pred_1"],
                "test_prob_mean": test_metrics["prob_mean"],
                "test_prob_std": test_metrics["prob_std"],
                "bias": bias,
                "elapsed_sec": time.time() - start_time
            }

            results.append(row)

            df = pd.DataFrame(results)
            df.to_csv(OUT_CSV, index=False)

        factor_path = os.path.join(
            OUT_DIR,
            f"cp_factors_subjectwise_R{rank}.npz"
        )

        np.savez(
            factor_path,
            A=A,
            B=B,
            C=C,
            D=D,
            bias=bias,
            rank=rank,
            voxel_shape=np.asarray(voxel_shape)
        )

        print("Zapisano czynniki:", factor_path)

        del A, B, C, D
        gc.collect()

    print("\n" + "=" * 80)
    print("KONIEC")
    print("=" * 80)
    print("Wyniki zapisano do:", OUT_CSV)

    print(pd.DataFrame(results))


if __name__ == "__main__":
    run_experiment()
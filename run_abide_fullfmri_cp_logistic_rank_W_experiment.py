import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# USTAWIENIA
# =========================================================

DATA_DIR = r"C:\Users\HP\Searches\fmri_datasets\full_voxel\61x73x61_T100_subjectwise"

SUBJECTS_DIR = os.path.join(DATA_DIR, "subjects")
INDEX_CSV = os.path.join(DATA_DIR, "subject_index.csv")

OUT_DIR = "results_cp_logistic_rank_W_fullres_61x73x61_T100"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "abide_fullres_cp_logistic_rank_W_results.csv")
OUT_SUMMARY_CSV = os.path.join(OUT_DIR, "abide_fullres_cp_logistic_rank_W_summary.csv")

RANKS = [1, 2, 3, 5]

N_SPLITS = 3
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-3
WEIGHT_DECAY = 1e-4

RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# DATASET SUBJECTWISE
# =========================================================

class FMriSubjectwiseDataset(Dataset):
    def __init__(self, df, subjects_dir, indices, file_col, label_col):
        self.df = df.reset_index(drop=True)
        self.subjects_dir = subjects_dir
        self.indices = indices
        self.file_col = file_col
        self.label_col = label_col

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row_idx = self.indices[idx]
        row = self.df.iloc[row_idx]

        filename = str(row[self.file_col])

        # Jeśli w CSV jest pełna ścieżka, bierzemy ją bez zmian.
        # Jeśli jest tylko nazwa pliku, dokładamy folder subjects.
        if os.path.isabs(filename):
            x_path = filename
        else:
            if not filename.endswith(".npy"):
                filename = filename + ".npy"
            x_path = os.path.join(self.subjects_dir, filename)

        x = np.load(x_path).astype(np.float32)
        y = np.float32(row[self.label_col])

        return torch.from_numpy(x), torch.tensor(y)


# =========================================================
# MODEL CP-LOGISTIC
# =========================================================

class CPLogisticRegression4D(nn.Module):
    """
    Model:

        P(y=1 | X) = sigmoid(<X, W> + b)

    Tensor wag W ma postać CP:

        W = sum_{r=1}^{R} a_r o b_r o c_r o d_r

    Dla fullres:
        X shape = (61, 73, 61, 100)

    Czynniki:
        A shape = (61, R)
        B shape = (73, R)
        C shape = (61, R)
        D shape = (100, R)
    """

    def __init__(self, tensor_shape, rank):
        super().__init__()

        x_dim, y_dim, z_dim, t_dim = tensor_shape
        self.rank = rank

        self.A = nn.Parameter(0.01 * torch.randn(x_dim, rank))
        self.B = nn.Parameter(0.01 * torch.randn(y_dim, rank))
        self.C = nn.Parameter(0.01 * torch.randn(z_dim, rank))
        self.D = nn.Parameter(0.01 * torch.randn(t_dim, rank))

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x shape: (batch, X, Y, Z, T)

        Liczymy:
            <X_i, W_R> + b

        bez jawnego tworzenia pełnego tensora W.
        """

        scores_per_rank = torch.einsum(
            "bxyzt,xr,yr,zr,tr->br",
            x,
            self.A,
            self.B,
            self.C,
            self.D
        )

        logits = scores_per_rank.sum(dim=1) + self.bias

        return logits


# =========================================================
# FUNKCJE POMOCNICZE
# =========================================================

def count_cp_parameters(tensor_shape, rank):
    x_dim, y_dim, z_dim, t_dim = tensor_shape
    return rank * (x_dim + y_dim + z_dim + t_dim) + 1


def detect_file_column(df):
    possible_file_cols = [
        "filename",
        "file",
        "npy_file",
        "path",
        "filepath",
        "file_path",
        "subject_file"
    ]

    for col in possible_file_cols:
        if col in df.columns:
            return col

    # Szukamy kolumny, która zawiera nazwy .npy
    for col in df.columns:
        values = df[col].astype(str).head(30).tolist()
        if any(".npy" in v for v in values):
            return col

    # Awaryjnie: jeśli liczba plików .npy zgadza się z liczbą wierszy w CSV,
    # tworzymy kolumnę filename na podstawie listy plików z folderu subjects.
    npy_files = sorted([f for f in os.listdir(SUBJECTS_DIR) if f.endswith(".npy")])

    if len(npy_files) == len(df):
        df["filename"] = npy_files
        return "filename"

    raise ValueError(
        "Nie znaleziono kolumny z nazwą pliku .npy i nie da się jej automatycznie odtworzyć."
    )


def detect_label_column(df):
    possible_label_cols = [
        "y",
        "label",
        "target",
        "DX_GROUP",
        "dx_group",
        "diagnosis",
        "class"
    ]

    for col in possible_label_cols:
        if col in df.columns:
            return col

    # Szukamy kolumny z wartościami binarnymi 0/1 albo ABIDE 1/2
    for col in df.columns:
        try:
            vals = df[col].dropna().astype(int).unique()
            vals_set = set(vals.tolist())

            if vals_set == {0, 1} or vals_set == {1, 2}:
                return col
        except Exception:
            pass

    raise ValueError("Nie znaleziono kolumny z etykietą klasy.")


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    total_n = 0

    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        logits = model(xb)
        loss = criterion(logits, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_n += xb.size(0)

    return total_loss / total_n


def evaluate(model, loader):
    model.eval()

    y_true = []
    y_prob = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)

            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()

            y_prob.extend(probs.tolist())
            y_true.extend(yb.numpy().astype(int).tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    sensitivity_asd = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity_control = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    return acc, auc, sensitivity_asd, specificity_control, cm


# =========================================================
# EKSPERYMENT
# =========================================================

def main():
    print("DEVICE:", DEVICE)

    print("\nŁadowanie danych subjectwise...")
    print("DATA_DIR:", DATA_DIR)
    print("SUBJECTS_DIR:", SUBJECTS_DIR)
    print("INDEX_CSV:", INDEX_CSV)

    if not os.path.isdir(SUBJECTS_DIR):
        raise FileNotFoundError(f"Nie znaleziono folderu subjects: {SUBJECTS_DIR}")

    if not os.path.exists(INDEX_CSV):
        raise FileNotFoundError(f"Nie znaleziono subject_index.csv: {INDEX_CSV}")

    df_index = pd.read_csv(INDEX_CSV)

    print("\nKolumny subject_index.csv:")
    print(df_index.columns.tolist())

    print("\nPierwsze wiersze subject_index.csv:")
    print(df_index.head())

    file_col = detect_file_column(df_index)
    label_col = detect_label_column(df_index)

    print("\nUżywam kolumny z plikiem:", file_col)
    print("Używam kolumny z etykietą:", label_col)

    # Mapowanie etykiet
    y_original = df_index[label_col].values.astype(int)

    # Jeśli ABIDE: 1=ASD, 2=CONTROL, mapujemy na 1=ASD, 0=CONTROL
    if set(np.unique(y_original)) == {1, 2}:
        y = np.where(y_original == 1, 1, 0)
    else:
        y = y_original

    df_index["label_binary"] = y
    label_col = "label_binary"

    print("\nRozkład klas po mapowaniu:")
    print(np.unique(y, return_counts=True))
    print("Przyjmujemy: 0 = CONTROL, 1 = ASD")

    n_subjects = len(df_index)

    # Sprawdzenie kształtu pierwszego subjecta
    first_filename = str(df_index.iloc[0][file_col])

    if os.path.isabs(first_filename):
        first_path = first_filename
    else:
        if not first_filename.endswith(".npy"):
            first_filename = first_filename + ".npy"
        first_path = os.path.join(SUBJECTS_DIR, first_filename)

    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Nie znaleziono przykładowego pliku subjecta: {first_path}")

    sample = np.load(first_path, mmap_mode="r")
    tensor_shape = sample.shape

    print("\nLiczba subjectów:", n_subjects)
    print("Przykładowy plik:", first_path)
    print("Pojedynczy tensor fMRI:", tensor_shape)

    full_W_params = int(np.prod(tensor_shape))

    print("\nPełny tensor W miałby parametrów:", full_W_params)

    for rank in RANKS:
        cp_params = count_cp_parameters(tensor_shape, rank)
        print(f"CP W dla R={rank} ma parametrów: {cp_params}")

    indices = np.arange(n_subjects)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    all_results = []

    for rank in RANKS:
        print("\n" + "=" * 80)
        print(f"RANGA CP R = {rank}")
        print("=" * 80)

        cp_params = count_cp_parameters(tensor_shape, rank)

        print(f"Liczba parametrów CP W dla R={rank}: {cp_params}")
        print(f"Liczba parametrów pełnego W: {full_W_params}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(indices, y), start=1):
            print("\n" + "-" * 80)
            print(f"R={rank}, fold {fold}/{N_SPLITS}")
            print("-" * 80)

            train_dataset = FMriSubjectwiseDataset(
                df=df_index,
                subjects_dir=SUBJECTS_DIR,
                indices=train_idx,
                file_col=file_col,
                label_col=label_col
            )

            test_dataset = FMriSubjectwiseDataset(
                df=df_index,
                subjects_dir=SUBJECTS_DIR,
                indices=test_idx,
                file_col=file_col,
                label_col=label_col
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0
            )

            model = CPLogisticRegression4D(
                tensor_shape=tensor_shape,
                rank=rank
            ).to(DEVICE)

            n_pos = np.sum(y[train_idx] == 1)
            n_neg = np.sum(y[train_idx] == 0)
            pos_weight_value = n_neg / max(n_pos, 1)

            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(
                    pos_weight_value,
                    dtype=torch.float32
                ).to(DEVICE)
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=LR,
                weight_decay=WEIGHT_DECAY
            )

            start_time = time.time()
            final_loss = None

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion
                )

                acc, auc, sens, spec, cm = evaluate(model, test_loader)

                final_loss = train_loss

                print(
                    f"R={rank} | fold={fold} | epoka={epoch}/{EPOCHS} | "
                    f"loss={train_loss:.6f} | acc={acc:.4f} | auc={auc:.4f} | "
                    f"sens_ASD={sens:.4f} | spec_CONTROL={spec:.4f}"
                )

            elapsed = time.time() - start_time

            acc, auc, sens, spec, cm = evaluate(model, test_loader)

            print("\nConfusion matrix [0=CONTROL, 1=ASD]:")
            print(cm)

            result = {
                "rank": rank,
                "fold": fold,
                "accuracy": acc,
                "roc_auc": auc,
                "sensitivity_ASD": sens,
                "specificity_CONTROL": spec,
                "final_loss": final_loss,
                "cp_parameters": cp_params,
                "full_W_parameters": full_W_params,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "train_time_sec": elapsed,
                "tensor_shape": str(tensor_shape),
                "n_subjects": n_subjects
            }

            all_results.append(result)

            df_partial = pd.DataFrame(all_results)
            df_partial.to_csv(OUT_CSV, index=False)

            model_path = os.path.join(
                OUT_DIR,
                f"cp_logistic_fullres_rank_{rank}_fold_{fold}.pt"
            )

            torch.save(model.state_dict(), model_path)

            print(f"\nZapisano model: {model_path}")
            print(f"Zapisano częściowe wyniki: {OUT_CSV}")

    df = pd.DataFrame(all_results)

    summary = df.groupby("rank").agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        sensitivity_ASD_mean=("sensitivity_ASD", "mean"),
        sensitivity_ASD_std=("sensitivity_ASD", "std"),
        specificity_CONTROL_mean=("specificity_CONTROL", "mean"),
        specificity_CONTROL_std=("specificity_CONTROL", "std"),
        final_loss_mean=("final_loss", "mean"),
        final_loss_std=("final_loss", "std"),
        train_time_sec_mean=("train_time_sec", "mean"),
        train_time_sec_sum=("train_time_sec", "sum"),
        cp_parameters=("cp_parameters", "first"),
        full_W_parameters=("full_W_parameters", "first")
    ).reset_index()

    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print("PODSUMOWANIE")
    print("=" * 80)
    print(summary)

    print("\nZapisano:")
    print(OUT_CSV)
    print(OUT_SUMMARY_CSV)


if __name__ == "__main__":
    main()
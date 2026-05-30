from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

# =========================
# ŚCIEŻKI
# =========================
DATASET_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\61x73x61_T100_subjectwise")
INDEX_PATH = DATASET_DIR / "subject_index.csv"
INFO_PATH = DATASET_DIR / "info.txt"

# ile subjectów sprawdzić na start
N_EXAMPLES = 5


def main():
    print("=== SPRAWDZENIE DATASETU FULL-RES SUBJECTWISE ===")
    print("Dataset dir:", DATASET_DIR)
    print("Index path :", INDEX_PATH)
    print("Info path  :", INFO_PATH)

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {INDEX_PATH}")

    df = pd.read_csv(INDEX_PATH)

    print("\n=== PODSTAWOWE INFO ===")
    print("Liczba wierszy w subject_index.csv:", len(df))
    print("Kolumny:", list(df.columns))

    if "label_binary" in df.columns:
        print("Rozkład klas:", Counter(df["label_binary"].tolist()))

    print("\n=== PIERWSZE WPISY ===")
    print(df.head())

    print(f"\n=== TEST ODCZYTU {N_EXAMPLES} SUBJECTÓW ===")
    for i in range(min(N_EXAMPLES, len(df))):
        row = df.iloc[i]
        npy_path = Path(row["saved_npy"])

        if not npy_path.exists():
            print(f"[{i}] BRAK PLIKU:", npy_path)
            continue

        x = np.load(npy_path)

        print(f"\n--- SUBJECT {i} ---")
        print("FILE_ID       :", row["FILE_ID"])
        print("label_binary  :", row["label_binary"])
        print("DX original   :", row["DX_GROUP_original"])
        print("saved_npy     :", npy_path)
        print("shape         :", x.shape)
        print("dtype         :", x.dtype)
        print("min           :", float(x.min()))
        print("max           :", float(x.max()))
        print("mean          :", float(x.mean()))
        print("std           :", float(x.std()))

        size_mb = npy_path.stat().st_size / (1024 ** 2)
        print("rozmiar pliku :", round(size_mb, 2), "MB")

    print("\n=== SPRAWDZENIE LOSOWEGO SUBJECTA ===")
    rand_idx = np.random.randint(0, len(df))
    row = df.iloc[rand_idx]
    npy_path = Path(row["saved_npy"])
    x = np.load(npy_path)

    print("Losowy idx    :", rand_idx)
    print("FILE_ID       :", row["FILE_ID"])
    print("label_binary  :", row["label_binary"])
    print("shape         :", x.shape)
    print("mean/std      :", float(x.mean()), float(x.std()))

    print("\n=== GOTOWE ===")
    print("Dataset full-res subjectwise wygląda na poprawnie zapisany.")


if __name__ == "__main__":
    main()
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm

# =========================
# ŚCIEŻKI
# =========================
RAW_DATA_DIR = Path(r"C:\Users\HP\Searches\abide_data")
FMRI_DIR = RAW_DATA_DIR / "ABIDE_pcp" / "cpac" / "filt_global"
PHENO_PATH = RAW_DATA_DIR / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"

OUT_DIR = Path(r"C:\Users\HP\Searches\fmri_datasets\full_voxel\61x73x61_T100_subjectwise")
SUBJECTS_DIR = OUT_DIR / "subjects"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# PARAMETRY DATASETU
# =========================
TARGET_SHAPE = (61, 73, 61)   # pełna rozdzielczość
TARGET_T = 100

# =========================
# FUNKCJE POMOCNICZE
# =========================
def standardize_subject(data_4d: np.ndarray) -> np.ndarray:
    mean = data_4d.mean()
    std = data_4d.std()
    if std < 1e-8:
        return (data_4d - mean).astype(np.float32)
    return ((data_4d - mean) / std).astype(np.float32)

def file_id_from_name(filename: str) -> str:
    return filename.split("_func")[0]

# =========================
# WCZYTANIE FENO
# =========================
if not PHENO_PATH.exists():
    raise FileNotFoundError(f"Nie znaleziono pliku fenotypowego: {PHENO_PATH}")

df = pd.read_csv(PHENO_PATH)

required_cols = {"FILE_ID", "DX_GROUP"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Brakuje kolumn w CSV: {missing}")

df["FILE_ID"] = df["FILE_ID"].astype(str)
dx_map = dict(zip(df["FILE_ID"], df["DX_GROUP"]))

# =========================
# LISTA PLIKÓW
# =========================
all_files = sorted(FMRI_DIR.glob("*.nii.gz"))
print("Liczba wszystkich plików .nii.gz:", len(all_files))

eligible_files = []
eligible_labels = []

for f in all_files:
    file_id = file_id_from_name(f.name)

    if file_id not in dx_map:
        continue

    img = nib.load(f)
    shape = img.header.get_data_shape()

    if len(shape) != 4:
        continue

    # oczekujemy pełnego shape przestrzennego
    if shape[:3] != TARGET_SHAPE:
        continue

    t = shape[3]
    if t < TARGET_T:
        continue

    dx = int(dx_map[file_id])
    if dx not in (1, 2):
        continue

    # ASD=1 -> 1, Control=2 -> 0
    y = 1 if dx == 1 else 0

    eligible_files.append(f)
    eligible_labels.append(y)

print(f"Liczba subjectów z T >= {TARGET_T}:", len(eligible_files))
print("Rozkład klas:", Counter(eligible_labels))

# =========================
# ZAPIS SUBJECT PO SUBJECTCIE
# =========================
subject_rows = []
selected_paths = []

for idx, f in enumerate(tqdm(eligible_files, desc="Zapisywanie subjectów")):
    file_id = file_id_from_name(f.name)
    y = eligible_labels[idx]

    img = nib.load(f)
    data = img.get_fdata().astype(np.float32)   # shape (61,73,61,T)

    # obcięcie czasu
    data = data[:, :, :, :TARGET_T]

    # standaryzacja per subject
    data = standardize_subject(data)

    out_name = f"{idx:04d}_{file_id}.npy"
    out_path = SUBJECTS_DIR / out_name
    np.save(out_path, data)

    selected_paths.append(str(f))
    subject_rows.append({
        "idx": idx,
        "FILE_ID": file_id,
        "source_path": str(f),
        "saved_npy": str(out_path),
        "label_binary": y,
        "DX_GROUP_original": 1 if y == 1 else 2
    })

subject_index_df = pd.DataFrame(subject_rows)
subject_index_df.to_csv(OUT_DIR / "subject_index.csv", index=False)

with open(OUT_DIR / "selected_paths.txt", "w", encoding="utf-8") as f:
    for p in selected_paths:
        f.write(p + "\n")

info_lines = [
    f"TARGET_SHAPE={TARGET_SHAPE}",
    f"TARGET_T={TARGET_T}",
    f"N_SUBJECTS={len(subject_rows)}",
    f"N_CONTROL={sum(row['label_binary'] == 0 for row in subject_rows)}",
    f"N_ASD={sum(row['label_binary'] == 1 for row in subject_rows)}",
    "SAVE_MODE=subjectwise_npy"
]

with open(OUT_DIR / "info.txt", "w", encoding="utf-8") as f:
    for line in info_lines:
        f.write(line + "\n")

print("\nZapisano:")
print(OUT_DIR / "subject_index.csv")
print(OUT_DIR / "selected_paths.txt")
print(OUT_DIR / "info.txt")
print("Folder subjectów:", SUBJECTS_DIR)
print("Liczba zapisanych plików .npy:", len(subject_rows))
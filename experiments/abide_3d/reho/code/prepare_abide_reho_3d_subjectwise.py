from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# KONFIGURACJA
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\HP\Searches\fmri-tensor-project-git")

REHO_DIR = (
    PROJECT_ROOT
    / "ABIDE_CPAC_3D_REHO_FALFF"
    / "Outputs"
    / "cpac"
    / "filt_noglobal"
    / "reho"
)

PHENOTYPIC_CSV = Path(
    r"C:\Users\HP\Searches\abide_data\ABIDE_pcp\Phenotypic_V1_0b_preprocessed1.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "61x73x61_reho_filt_noglobal_subjectwise"

EXPECTED_SHAPE = (61, 73, 61)


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def file_id_from_reho_path(path: Path) -> str:
    """
    Zamienia np.
    Caltech_0051456_reho.nii.gz
    na:
    Caltech_0051456
    """
    return path.name.replace("_reho.nii.gz", "")


def encode_dx_group(dx_group: int) -> int:
    """
    ABIDE:
    DX_GROUP = 1 -> ASD
    DX_GROUP = 2 -> CONTROL

    Do klasyfikacji binarnej:
    ASD     -> 1
    CONTROL -> 0
    """
    if dx_group == 1:
        return 1
    if dx_group == 2:
        return 0

    raise ValueError(f"Nieznana wartość DX_GROUP: {dx_group}")


# ============================================================
# GŁÓWNY SKRYPT
# ============================================================

def main() -> None:
    print("==============================================")
    print("Przygotowanie ABIDE ReHo 3D subjectwise")
    print("==============================================")

    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"REHO_DIR:        {REHO_DIR}")
    print(f"PHENOTYPIC_CSV:  {PHENOTYPIC_CSV}")
    print(f"OUTPUT_DIR:      {OUTPUT_DIR}")
    print()

    if not REHO_DIR.exists():
        raise FileNotFoundError(f"Nie istnieje folder ReHo: {REHO_DIR}")

    if not PHENOTYPIC_CSV.exists():
        raise FileNotFoundError(f"Nie istnieje plik fenotypowy: {PHENOTYPIC_CSV}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Folder na zapisane tensory .npy
    X_DIR = OUTPUT_DIR / "X"
    X_DIR.mkdir(parents=True, exist_ok=True)

    # Wczytanie fenotypów
    pheno = pd.read_csv(PHENOTYPIC_CSV)

    required_columns = ["FILE_ID", "DX_GROUP", "SITE_ID", "AGE_AT_SCAN", "SEX"]
    missing_columns = [c for c in required_columns if c not in pheno.columns]

    if missing_columns:
        raise ValueError(f"Brakuje kolumn w pliku fenotypowym: {missing_columns}")

    # Usuwamy rekordy bez nazwy pliku
    pheno = pheno[pheno["FILE_ID"].astype(str) != "no_filename"].copy()

    # Indeks po FILE_ID
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str)
    pheno_by_file_id = pheno.set_index("FILE_ID", drop=False)

    reho_files = sorted(REHO_DIR.glob("*_reho.nii.gz"))
    print(f"Liczba znalezionych plików ReHo: {len(reho_files)}")

    if len(reho_files) == 0:
        raise RuntimeError("Nie znaleziono plików *_reho.nii.gz")

    rows = []
    skipped = []

    for idx, reho_path in enumerate(reho_files, start=1):
        file_id = file_id_from_reho_path(reho_path)

        if file_id not in pheno_by_file_id.index:
            skipped.append(
                {
                    "file_id": file_id,
                    "reason": "brak FILE_ID w phenotypic CSV",
                    "path": str(reho_path),
                }
            )
            continue

        meta = pheno_by_file_id.loc[file_id]

        dx_group = int(meta["DX_GROUP"])
        y = encode_dx_group(dx_group)

        img = nib.load(str(reho_path))
        X = img.get_fdata(dtype=np.float32)

        if X.shape != EXPECTED_SHAPE:
            skipped.append(
                {
                    "file_id": file_id,
                    "reason": f"niepoprawny shape: {X.shape}",
                    "path": str(reho_path),
                }
            )
            continue

        # Sanityzacja numeryczna:
        # zamienia NaN/inf na 0, żeby model nie wysypał się podczas uczenia.
        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        out_name = f"{file_id}_reho.npy"
        out_path = X_DIR / out_name
        np.save(out_path, X)

        rows.append(
            {
                "file_id": file_id,
                "npy_path": str(out_path),
                "nii_path": str(reho_path),
                "y": y,
                "dx_group": dx_group,
                "diagnosis": "ASD" if y == 1 else "CONTROL",
                "site_id": meta["SITE_ID"],
                "age_at_scan": meta["AGE_AT_SCAN"],
                "sex": meta["SEX"],
                "shape": "61x73x61",
                "derivative": "reho",
                "pipeline": "cpac",
                "strategy": "filt_noglobal",
            }
        )

        if idx % 50 == 0 or idx == len(reho_files):
            print(f"Przetworzono {idx}/{len(reho_files)} plików...")

    index_df = pd.DataFrame(rows)

    index_path = OUTPUT_DIR / "subject_index_reho_filt_noglobal.csv"
    skipped_path = OUTPUT_DIR / "skipped_reho_filt_noglobal.csv"

    index_df.to_csv(index_path, index=False)
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)

    print()
    print("==============================================")
    print("ZAKOŃCZONO")
    print("==============================================")
    print(f"Liczba zapisanych subjectów: {len(index_df)}")
    print(f"Liczba pominiętych plików:   {len(skipped)}")
    print(f"Index CSV:                  {index_path}")
    print(f"Skipped CSV:                {skipped_path}")

    if len(index_df) > 0:
        print()
        print("Rozkład klas:")
        print(index_df["diagnosis"].value_counts())
        print()
        print("Pierwsze wiersze indeksu:")
        print(index_df.head())


if __name__ == "__main__":
    main()
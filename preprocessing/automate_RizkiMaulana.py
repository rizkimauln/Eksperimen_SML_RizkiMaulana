import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_dir):
    """
    Otomatisasi preprocessing dataset diabetes.

    Parameters:
    input_path : str / Path
        Path menuju file diabetes.csv mentah.

    output_dir : str / Path
        Folder output untuk menyimpan hasil preprocessing.
    """

    input_path = Path(input_path)
    output_dir = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"File dataset tidak ditemukan: {input_path}")

    # 1. Load dataset
    df = pd.read_csv(input_path)

    # 2. Kolom yang tidak wajar jika bernilai 0
    zero_cols = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    # 3. Ganti nilai 0 menjadi NaN
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    # 4. Imputasi nilai kosong menggunakan median
    imputer = SimpleImputer(strategy="median")
    df[zero_cols] = imputer.fit_transform(df[zero_cols])

    # 5. Hapus data duplikat
    df = df.drop_duplicates()

    # 6. Cap outlier menggunakan metode IQR
    for col in df.columns[:-1]:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df[col] = np.clip(df[col], lower_bound, upper_bound)

    # 7. Pisahkan fitur dan target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # 8. Split data train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 9. Standarisasi fitur
    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns
    )

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns
    )

    # 10. Buat folder output jika belum ada
    os.makedirs(output_dir, exist_ok=True)

    # 11. Simpan hasil preprocessing
    X_train_scaled.to_csv(output_dir / "X_train.csv", index=False)
    X_test_scaled.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Preprocessing selesai.")
    print(f"Dataset input  : {input_path}")
    print(f"Output disimpan: {output_dir}")

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_path = BASE_DIR / "diabetes_raw" / "diabetes.csv"
    output_dir = BASE_DIR / "preprocessing" / "diabetes_preprocessing"

    preprocess_data(
        input_path=input_path,
        output_dir=output_dir
    )
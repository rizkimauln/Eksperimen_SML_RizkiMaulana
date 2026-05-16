import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(input_path, output_dir):
    """
    Otomatisasi preprocessing dataset diabetes.
    input_path: path ke diabetes.csv mentah
    output_dir: folder output untuk simpan hasil
    """
    # 1. Load
    df = pd.read_csv(input_path)
    
    # 2. Ganti 0 tidak wajar -> NaN
    zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    
    # 3. Imputasi median
    imputer = SimpleImputer(strategy='median')
    df[zero_cols] = imputer.fit_transform(df[zero_cols])
    
    # 4. Hapus duplikat
    df = df.drop_duplicates()
    
    # 5. Cap outlier IQR
    for col in df.columns[:-1]:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1-1.5*IQR, Q3+1.5*IQR)
    
    # 6. Split fitur-target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # 7. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 8. Standarisasi
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # 9. Simpan
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test_scaled.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print(f"✓ Preprocessing selesai. Output: {output_dir}")
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    file_input = os.path.join(script_dir, '../diabetes_raw/diabetes.csv')
    folder_output = os.path.join(script_dir, 'diabetes_preprocessing')

    preprocess_data(
        input_path=file_input,
        output_dir=folder_output
    )
# src/preprocess.py
"""Reusable preprocessing utilities for the AI4I 2020 predictive maintenance dataset.

- Load CSV from the project root (or provide a full path).
- Perform the exact steps described in Unified_Predictive_Maintenance_Assessment.docx:
  1. Drop rows where the target column `failure` is missing.
  2. Median‑impute numeric columns.
  3. One‑hot encode categorical columns (drop_first=True).
  4. Return cleaned DataFrame and the name of the target column.

These functions can be imported from any notebook or script:
    from src.preprocess import load_and_clean
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean(csv_path: str | Path = "ai4i2020.csv") -> tuple[pd.DataFrame, str]:
    """Load the AI4I 2020 CSV and apply the standard preprocessing.

    Parameters
    ----------
    csv_path: str or Path, optional
        Path to the CSV file. If a relative path is given it is interpreted
        relative to the project root (the folder that contains this script).

    Returns
    -------
    df_clean: pd.DataFrame
        Pre‑processed DataFrame ready for modelling.
    target_col: str
        Name of the target column (defaults to ``failure`` if present).
    """
    # Resolve the path – allow both relative and absolute inputs
    csv_path = Path(csv_path).expanduser().resolve()
    df = pd.read_csv(csv_path)

    # 1️⃣ Target handling – drop rows with missing target
    target_col = "failure" if "failure" in df.columns else df.columns[-1]
    df = df.dropna(subset=[target_col])

    # 2️⃣ Numeric imputation – median for each numeric column
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median)

    # 3️⃣ Categorical encoding – one‑hot, drop_first to avoid dummy trap
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df, target_col

if __name__ == "__main__":
    # Simple sanity‑check when the module is run directly
    cleaned, target = load_and_clean()
    print("Loaded & cleaned dataset:")
    print("  shape:", cleaned.shape)
    print("  target column:", target)
    print(cleaned.head())

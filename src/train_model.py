# src/train_model.py
"""Reusable model‑training utilities for the AI4I 2020 predictive‑maintenance project.

The script assumes that the data has already been pre‑processed using
`src.preprocess.load_and_clean`. It provides a single helper function
`train_random_forest` that:

1. Splits the cleaned DataFrame into train / test (80/20 stratified).
2. Trains a `RandomForestClassifier` with configurable hyper‑parameters.
3. Returns the trained model, the test set, and a dictionary of evaluation
   metrics (accuracy, precision, recall, f1‑score).

The module can be imported from notebooks or other scripts:

    from src.train_model import train_random_forest
    model, X_test, y_test, metrics = train_random_forest(df, target_col)
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

def train_random_forest(
    df: pd.DataFrame,
    target_col: str,
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
    class_weight: str | dict | None = None,
) -> tuple[RandomForestClassifier, pd.DataFrame, pd.Series, dict]:
    """Train a RandomForest model on the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Pre‑processed DataFrame (features + target).
    target_col : str
        Name of the column containing the target labels.
    n_estimators : int, default 100
        Number of trees in the forest.
    max_depth : int or None, default None
        Maximum depth of each tree (None → expand until leaves are pure).
    random_state : int, default 42
        Seed for reproducibility.
    test_size : float, default 0.2
        Fraction of data to set aside for the test split.
    class_weight : str or dict or None, default None
        Weighting strategy for handling class imbalance (e.g. "balanced").

    Returns
    -------
    model : RandomForestClassifier
        The fitted model.
    X_test : pd.DataFrame
        Features of the held‑out test set.
    y_test : pd.Series
        True labels of the test set.
    metrics : dict
        Dictionary with keys ``accuracy``, ``precision``, ``recall``, ``f1`` and a
        full ``classification_report`` string.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Stratified split keeps the minority‑class proportion in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
    )
    clf.fit(X_train, y_train)

    # Predictions & basic metrics
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
        "classification_report": classification_report(y_test, y_pred),
    }
    return clf, X_test, y_test, metrics

if __name__ == "__main__":
    # Quick demo when run directly – assumes the CSV lives next to the repo root
    from src.preprocess import load_and_clean
    df_clean, target = load_and_clean()
    model, X_test, y_test, mets = train_random_forest(df_clean, target, class_weight='balanced')
    print("Model trained. Metrics:")
    for k, v in mets.items():
        if k == "classification_report":
            print(v)
        else:
            print(f"{k}: {v:.4f}")

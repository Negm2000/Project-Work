from __future__ import annotations

"""
Feature engineering and simple model-based feature importance search.

Reads `features_labeled.csv`, adds engineered features, trains simple models
and writes results under `results/`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"


NON_FEATURE_COLUMNS = {
    "sample_id",
    "board_id",
    "connector_name",
    "filename",
    "label",
}


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    feature_names: List[str]


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_labeled_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"sample_id", "board_id", "connector_name", "filename", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"features_labeled.csv is missing columns: {missing}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic derived features if base columns exist."""
    df_out = df.copy()
    has_gray_mean = "gray_mean" in df_out.columns
    has_gray_std = "gray_std" in df_out.columns
    has_band_mean = "band_mean" in df_out.columns
    has_band_std = "band_std" in df_out.columns

    if has_band_mean and has_gray_mean:
        df_out["band_minus_gray"] = df_out["band_mean"] - df_out["gray_mean"]
        df_out["band_div_gray"] = df_out["band_mean"] / (df_out["gray_mean"] + 1e-6)

    if has_band_std and has_gray_std:
        df_out["std_ratio"] = df_out["band_std"] / (df_out["gray_std"] + 1e-6)

    if has_gray_std:
        df_out["inv_std"] = 1.0 / (df_out["gray_std"] + 1e-6)

    if has_band_std and has_band_mean:
        df_out["contrast_band"] = df_out["band_std"] / (df_out["band_mean"] + 1e-6)

    if has_gray_std and has_gray_mean:
        df_out["local_contrast"] = df_out["gray_std"] / (df_out["gray_mean"] + 1e-6)

    return df_out


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return numeric feature matrix X_df and label vector y.
    
    Note: Only OK and KO labels are used for training. OCCLUSION labels are excluded.
    """
    df_labeled = df[df["label"].isin(["OK", "KO"])].copy()
    if df_labeled.empty:
        raise ValueError("No labeled rows with label in {OK, KO} found.")

    y = np.where(df_labeled["label"] == "KO", 1, 0)

    numeric_cols = df_labeled.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in NON_FEATURE_COLUMNS]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    X_df = df_labeled[feature_cols].copy()
    return X_df, y


def split_by_board(
    df_features: pd.DataFrame, y: np.ndarray
) -> DatasetSplit:
    """Split train/validation by board_id (80/20)."""
    board_ids = df_features["board_id"].values
    unique_boards = np.unique(board_ids)
    train_boards, val_boards = train_test_split(
        unique_boards, test_size=0.2, random_state=42
    )

    train_mask = np.isin(board_ids, train_boards)
    val_mask = np.isin(board_ids, val_boards)

    X = df_features.drop(columns=["board_id"]).values
    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    feature_names = [c for c in df_features.columns if c != "board_id"]
    return DatasetSplit(X_train, X_val, y_train, y_val, feature_names)


def evaluate_model(
    name: str, y_true_train: np.ndarray, y_pred_train: np.ndarray, y_proba_train: np.ndarray,
    y_true_val: np.ndarray, y_pred_val: np.ndarray, y_proba_val: np.ndarray,
) -> None:
    def metrics(y_true, y_pred, y_proba) -> Tuple[float, float, float, float, float]:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
        return acc, f1, precision, recall, auc

    acc_tr, f1_tr, prec_tr, rec_tr, auc_tr = metrics(y_true_train, y_pred_train, y_proba_train)
    acc_va, f1_va, prec_va, rec_va, auc_va = metrics(y_true_val, y_pred_val, y_proba_val)

    print(f"\n=== {name} ===")
    print(f"Train  - Acc: {acc_tr:.3f}, F1: {f1_tr:.3f}, Prec: {prec_tr:.3f}, Rec: {rec_tr:.3f}, ROC-AUC: {auc_tr:.3f}")
    print(f"Valid. - Acc: {acc_va:.3f}, F1: {f1_va:.3f}, Prec: {prec_va:.3f}, Rec: {rec_va:.3f}, ROC-AUC: {auc_va:.3f}")
    
    # Confusion matrix per validation
    cm_val = confusion_matrix(y_true_val, y_pred_val)
    print(f"Validation Confusion Matrix:\n{cm_val}")
    print(f"  (True Neg: {cm_val[0,0]}, False Pos: {cm_val[0,1]}, False Neg: {cm_val[1,0]}, True Pos: {cm_val[1,1]})")


def fit_logistic(split: DatasetSplit) -> None:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=5000, class_weight="balanced")),
        ]
    )
    pipe.fit(split.X_train, split.y_train)

    proba_train = pipe.predict_proba(split.X_train)[:, 1]
    proba_val = pipe.predict_proba(split.X_val)[:, 1]
    pred_train = (proba_train >= 0.5).astype(int)
    pred_val = (proba_val >= 0.5).astype(int)

    evaluate_model(
        "LogisticRegression (L1)",
        split.y_train,
        pred_train,
        proba_train,
        split.y_val,
        pred_val,
        proba_val,
    )

    clf: LogisticRegression = pipe.named_steps["clf"]
    coefs = clf.coef_.ravel()
    coef_df = pd.DataFrame(
        {"feature_name": split.feature_names, "coefficient": coefs}
    ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False)
    coef_path = RESULTS_DIR / "feature_coefficients_logreg.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"Saved logistic coefficients to {coef_path}")


def fit_random_forest(split: DatasetSplit) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(split.X_train, split.y_train)

    proba_train = rf.predict_proba(split.X_train)[:, 1]
    proba_val = rf.predict_proba(split.X_val)[:, 1]
    pred_train = (proba_train >= 0.5).astype(int)
    pred_val = (proba_val >= 0.5).astype(int)

    evaluate_model(
        "RandomForest",
        split.y_train,
        pred_train,
        proba_train,
        split.y_val,
        pred_val,
        proba_val,
    )

    importances = rf.feature_importances_
    imp_df = pd.DataFrame(
        {"feature_name": split.feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    imp_path = RESULTS_DIR / "feature_importance_rf.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"Saved RF feature importances to {imp_path}")

    # Bar plot of top 15 features
    top_n = 15
    top_df = imp_df.head(top_n)
    plt.figure(figsize=(10, 6))
    plt.barh(top_df["feature_name"][::-1], top_df["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} features - RandomForest")
    plt.tight_layout()
    plot_path = PLOTS_DIR / "feature_importance_rf.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved RF importance plot to {plot_path}")

    return rf


def compute_permutation_importance(
    rf: RandomForestClassifier, split: DatasetSplit
) -> None:
    result = permutation_importance(
        rf,
        split.X_val,
        split.y_val,
        scoring="f1",
        n_repeats=20,
        random_state=42,
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature_name": split.feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    path = RESULTS_DIR / "feature_importance_perm.csv"
    perm_df.to_csv(path, index=False)
    print(f"Saved permutation importance to {path}")

    print("\nTop 10 features by permutation importance (F1 on validation):")
    for _, row in perm_df.head(10).iterrows():
        print(f"- {row['feature_name']}: {row['importance_mean']:.4f}")


def main() -> None:
    ensure_output_dirs()
    input_path = Path("features_labeled.csv")
    df = load_labeled_features(input_path)

    df_engineered = engineer_features(df)
    fe_path = RESULTS_DIR / "feature_engineered.csv"
    df_engineered.to_csv(fe_path, index=False)
    print(f"Saved engineered features to {fe_path}")

    X_df, y = prepare_xy(df_engineered)
    
    # Warn about class imbalance
    n_ok = np.sum(y == 0)
    n_ko = np.sum(y == 1)
    print(f"\n=== Dataset Info ===")
    print(f"Total samples: {len(y)}")
    print(f"  OK: {n_ok} ({100*n_ok/len(y):.1f}%)")
    print(f"  KO: {n_ko} ({100*n_ko/len(y):.1f}%)")
    if n_ko < 20:
        print(f"\n⚠️  WARNING: Very few KO examples ({n_ko}). Model may struggle to learn.")
        print(f"   Consider labeling more KO examples for better performance.\n")
    
    split = split_by_board(X_df.join(df_engineered["board_id"]), y)
    
    print(f"Train set: {len(split.y_train)} samples ({np.sum(split.y_train==0)} OK, {np.sum(split.y_train==1)} KO)")
    print(f"Validation set: {len(split.y_val)} samples ({np.sum(split.y_val==0)} OK, {np.sum(split.y_val==1)} KO)\n")

    fit_logistic(split)
    rf = fit_random_forest(split)
    compute_permutation_importance(rf, split)


if __name__ == "__main__":
    main()

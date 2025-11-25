
"""
Feature selection and interpretability pipeline.

Loads engineered features from `results/feature_engineered.csv`, runs RFECV-based
feature selection and SHAP analysis, and writes outputs under `results/`.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV


RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"

NON_FEATURE_COLUMNS = {
    "sample_id",
    "board_id",
    "connector_name",
    "filename",
    "label",
}


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_engineered_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Engineered feature CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"board_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"feature_engineered.csv is missing columns: {missing}")
    return df


def prepare_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return numeric feature matrix X_df and label vector y.
    
    Note: Only OK and KO labels are used for training. OCCLUSION labels are excluded.
    """
    df_labeled = df[df["label"].isin(["OK", "KO"])].copy()
    if df_labeled.empty:
        raise ValueError("No labeled rows found with label in {OK, KO}.")

    y = np.where(df_labeled["label"] == "KO", 1, 0)
    numeric_cols = df_labeled.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in NON_FEATURE_COLUMNS]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after exclusions.")

    X_df = df_labeled[feature_cols].copy()
    return X_df, y


def split_by_board(
    df_features: pd.DataFrame, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
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
    return X_train, X_val, y_train, y_val, feature_names


def run_rfecv(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str]
) -> Tuple[RFECV, List[str]]:
    estimator = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    rfecv.fit(X_train, y_train)

    selected_mask = rfecv.support_
    selected_features = [f for f, keep in zip(feature_names, selected_mask) if keep]

    print(f"RFECV selected {len(selected_features)} features out of {len(feature_names)}.")
    print("Selected features:")
    for name in selected_features:
        print(f"- {name}")

    # Save selected feature list
    sel_path = RESULTS_DIR / "selected_features.txt"
    with sel_path.open("w", encoding="utf-8") as f:
        for name in selected_features:
            f.write(f"{name}\n")
    print(f"Saved selected features to {sel_path}")

    # RFECV curve
    try:
        # Newer sklearn versions
        scores = rfecv.cv_results_["mean_test_score"]
    except AttributeError:
        # Older versions
        scores = rfecv.grid_scores_

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(scores) + 1), scores, marker="o")
    plt.xlabel("Number of selected features")
    plt.ylabel("CV score (F1)")
    plt.title("RFECV - Feature selection curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    curve_path = PLOTS_DIR / "rfecv_curve.png"
    plt.savefig(curve_path)
    plt.close()
    print(f"Saved RFECV curve to {curve_path}")

    return rfecv, selected_features


def evaluate_on_validation(
    model: RandomForestClassifier,
    X_train_sel: np.ndarray,
    y_train: np.ndarray,
    X_val_sel: np.ndarray,
    y_val: np.ndarray,
) -> None:
    model.fit(X_train_sel, y_train)
    proba_train = model.predict_proba(X_train_sel)[:, 1]
    proba_val = model.predict_proba(X_val_sel)[:, 1]

    pred_train = (proba_train >= 0.5).astype(int)
    pred_val = (proba_val >= 0.5).astype(int)

    f1_tr = f1_score(y_train, pred_train, zero_division=0)
    f1_va = f1_score(y_val, pred_val, zero_division=0)
    prec_tr = precision_score(y_train, pred_train, zero_division=0)
    prec_va = precision_score(y_val, pred_val, zero_division=0)
    rec_tr = recall_score(y_train, pred_train, zero_division=0)
    rec_va = recall_score(y_val, pred_val, zero_division=0)
    auc_tr = roc_auc_score(y_train, proba_train) if len(np.unique(y_train)) > 1 else 0.0
    auc_va = roc_auc_score(y_val, proba_val) if len(np.unique(y_val)) > 1 else 0.0

    print("\nFinal RandomForest on selected features:")
    print(f"Train  - F1: {f1_tr:.3f}, Prec: {prec_tr:.3f}, Rec: {rec_tr:.3f}, ROC-AUC: {auc_tr:.3f}")
    print(f"Valid. - F1: {f1_va:.3f}, Prec: {prec_va:.3f}, Rec: {rec_va:.3f}, ROC-AUC: {auc_va:.3f}")
    
    cm_val = confusion_matrix(y_val, pred_val)
    print(f"Validation Confusion Matrix:\n{cm_val}")
    print(f"  (True Neg: {cm_val[0,0]}, False Pos: {cm_val[0,1]}, False Neg: {cm_val[1,0]}, True Pos: {cm_val[1,1]})")


def run_shap_analysis(
    model: RandomForestClassifier,
    X_val_sel: np.ndarray,
    selected_features: List[str],
) -> None:
    """Compute SHAP values on validation set and save summary/bar plots and CSV."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val_sel)

    # Binary classification: pick class 1 (KO)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    mean_abs = np.mean(np.abs(sv), axis=0)
    shap_df = pd.DataFrame(
        {"feature_name": selected_features, "mean_absolute_shap_value": mean_abs}
    ).sort_values("mean_absolute_shap_value", ascending=False)

    shap_path = RESULTS_DIR / "feature_rank_shap.csv"
    shap_df.to_csv(shap_path, index=False)
    print(f"Saved SHAP feature ranking to {shap_path}")

    # Summary plot
    shap.summary_plot(
        sv,
        features=X_val_sel,
        feature_names=selected_features,
        show=False,
    )
    summary_path = PLOTS_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(summary_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary plot to {summary_path}")

    # Bar plot
    shap.summary_plot(
        sv,
        features=X_val_sel,
        feature_names=selected_features,
        plot_type="bar",
        show=False,
    )
    bar_path = PLOTS_DIR / "shap_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP bar plot to {bar_path}")


def main() -> None:
    ensure_output_dirs()
    input_path = RESULTS_DIR / "feature_engineered.csv"
    df = load_engineered_features(input_path)

    X_df, y = prepare_xy(df)
    # Attach board_id for splitting
    X_df_with_board = X_df.join(df.loc[X_df.index, "board_id"])
    X_train, X_val, y_train, y_val, feature_names = split_by_board(
        X_df_with_board, y
    )

    rfecv, selected_features = run_rfecv(X_train, y_train, feature_names)

    # Reduce features according to RFECV mask
    mask = rfecv.support_
    X_train_sel = X_train[:, mask]
    X_val_sel = X_val[:, mask]

    final_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    evaluate_on_validation(final_rf, X_train_sel, y_train, X_val_sel, y_val)

    # Re-fit final model on selected features for SHAP
    final_rf.fit(X_train_sel, y_train)
    run_shap_analysis(final_rf, X_val_sel, selected_features)


if __name__ == "__main__":
    main()

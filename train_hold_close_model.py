"""
Trains the hold/close model on the dataset built by build_hold_close_dataset.py.

Design choices (matching what already worked best for the entry classifier at this sample
size): a small, heavily-regularized logistic regression on a handful of features, rather than
a complex model — with only ~3,000 positions (and far fewer independent ones once you account
for date-overlap), a complex model is more likely to memorize noise than find real signal.

Critical correctness points, all implemented below:
  - Split by POSITION and by DATE (walk-forward), not by row — rows from the same position are
    highly correlated (they're the same trade on consecutive days), so a random row-level split
    would leak: the model would see day 8 of a position in training and day 9 in test.
  - Any training position whose window (entry to exit) overlaps the test period is dropped
    entirely, not just its rows in the test window — otherwise the model still learns from
    "the future" relative to the test set.
  - Rows are weighted by 1/(days in that position) so a position held 20 days doesn't get 4x
    the influence of one held 5 days.
  - Evaluated on POLICY P&L, not accuracy/AUC — "follow the model" vs "always hold to planned
    exit" vs a simple "close if drawdown from peak exceeds threshold" rule, on held-out
    positions. A model that isn't clearly better than the simple rule isn't worth deploying.

Usage:
    from train_hold_close_model import train_hold_close_model
    train_hold_close_model(data_path=..., output_dir=...)
"""

import os
import sys
import json
import glob
import joblib
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from hold_close_features import MODEL_FEATURES


def find_latest_dataset(search_root):
    matches = sorted(glob.glob(os.path.join(search_root, "**", "hold_close_training_data_*.csv"), recursive=True))
    if not matches:
        raise FileNotFoundError(f"No hold_close_training_data_*.csv found under {search_root}")
    return matches[-1]


def add_position_id(df):
    df = df.copy()
    df["position_id"] = df["ticker"].astype(str) + "|" + df["entry_date"].astype(str) + "|" + df["strike"].astype(str)
    return df


def time_based_split(df, test_fraction=0.2):
    """Split by entry_date, then drop any TRAIN position whose window overlaps the test
    period at all — not just rows, whole positions — to prevent any future leakage."""
    df = df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    positions = df[["position_id", "entry_date", "exit_date"]].drop_duplicates().sort_values("entry_date")
    split_idx = int(len(positions) * (1 - test_fraction))
    test_start_date = positions.iloc[split_idx]["entry_date"]

    test_position_ids = set(positions[positions["entry_date"] >= test_start_date]["position_id"])
    train_candidate_ids = set(positions[positions["entry_date"] < test_start_date]["position_id"])

    # Drop training positions whose exit overlaps the test window (i.e. extends into or past it)
    overlapping = positions[(positions["position_id"].isin(train_candidate_ids)) & (positions["exit_date"] >= test_start_date)]
    train_position_ids = train_candidate_ids - set(overlapping["position_id"])

    dropped = len(overlapping)
    if dropped:
        print(f"Dropped {dropped} training position(s) whose holding window overlapped the test period.")

    train_df = df[df["position_id"].isin(train_position_ids)].copy()
    test_df = df[df["position_id"].isin(test_position_ids)].copy()
    return train_df, test_df


def add_sample_weights(df):
    df = df.copy()
    counts = df.groupby("position_id")["position_id"].transform("count")
    df["sample_weight"] = 1.0 / counts
    return df


def simulate_policy(df, predict_fn=None, drawdown_threshold=None):
    """
    For each position (in day order), simulate closing on the first day a rule says to close,
    else holding to the position's natural final mtm_pnl (the last row we have for it).

    Exactly one of predict_fn (a function: DataFrame of a position's rows -> array of 0/1
    close decisions) or drawdown_threshold (a simple "close if drawdown_from_peak worse than
    this" rule) should be given. Returns a DataFrame with one row per position: realized P&L
    under that policy, and under "always hold".
    """
    results = []
    for position_id, group in df.groupby("position_id"):
        group = group.sort_values("snapshot_date")
        always_hold_pnl = group["mtm_pnl"].iloc[-1]

        if predict_fn is not None:
            decisions = predict_fn(group)
        else:
            decisions = (group["drawdown_from_peak"] <= drawdown_threshold).astype(int).values

        close_idx = np.argmax(decisions) if decisions.any() else None
        policy_pnl = group["mtm_pnl"].iloc[close_idx] if close_idx is not None else always_hold_pnl

        results.append({"position_id": position_id, "policy_pnl": policy_pnl, "always_hold_pnl": always_hold_pnl})
    return pd.DataFrame(results)


def train_hold_close_model(
        data_path: str = None,
        search_root: str = None,
        output_dir: str = None,
        test_fraction: float = 0.2,
        C: float = 0.5,
        drawdown_rule_threshold: float = -50.0,
):
    """
    Train the hold/close model. Returns a dict with the model, metrics, and policy comparison.
    """
    search_root = search_root if search_root else os.path.dirname(os.path.abspath(__file__))
    output_dir = output_dir if output_dir else os.path.join(search_root, "hold_close_model")
    os.makedirs(output_dir, exist_ok=True)

    data_path = data_path if data_path else find_latest_dataset(search_root)
    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    df = add_position_id(df)

    available_features = [f for f in MODEL_FEATURES if f in df.columns]
    df = df.dropna(subset=available_features + ["close_today"]).copy()
    print(f"{len(df)} usable rows across {df['position_id'].nunique()} positions "
          f"(using {len(available_features)}/{len(MODEL_FEATURES)} features — "
          f"any missing ones weren't present in this dataset).")

    train_df, test_df = time_based_split(df, test_fraction=test_fraction)
    train_df = add_sample_weights(train_df)
    print(f"Train: {len(train_df)} rows / {train_df['position_id'].nunique()} positions. "
          f"Test: {len(test_df)} rows / {test_df['position_id'].nunique()} positions.")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(train_df[available_features], train_df["close_today"], clf__sample_weight=train_df["sample_weight"])

    test_probs = pipeline.predict_proba(test_df[available_features])[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(test_df["close_today"], test_preds),
        "auc": roc_auc_score(test_df["close_today"], test_probs) if test_df["close_today"].nunique() > 1 else None,
        "n_test_rows": len(test_df),
        "n_test_positions": int(test_df["position_id"].nunique()),
    }
    print(f"Test accuracy: {metrics['accuracy']:.3f}" + (f", AUC: {metrics['auc']:.3f}" if metrics["auc"] else ""))

    # --- Policy P&L comparison on held-out positions ---
    def model_predict_fn(group):
        probs = pipeline.predict_proba(group[available_features])[:, 1]
        return (probs > 0.5).astype(int)

    model_policy = simulate_policy(test_df, predict_fn=model_predict_fn)
    rule_policy = simulate_policy(test_df, drawdown_threshold=drawdown_rule_threshold)

    policy_comparison = {
        "always_hold_total_pnl": round(model_policy["always_hold_pnl"].sum(), 2),
        "model_policy_total_pnl": round(model_policy["policy_pnl"].sum(), 2),
        "simple_drawdown_rule_total_pnl": round(rule_policy["policy_pnl"].sum(), 2),
        "n_positions_compared": len(model_policy),
    }
    print(f"\nPolicy comparison over {policy_comparison['n_positions_compared']} held-out positions:")
    print(f"  Always hold:            ${policy_comparison['always_hold_total_pnl']:,.2f}")
    print(f"  Model policy:           ${policy_comparison['model_policy_total_pnl']:,.2f}")
    print(f"  Simple drawdown rule:   ${policy_comparison['simple_drawdown_rule_total_pnl']:,.2f}")

    # --- Save ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d__%H%M")
    model_path = os.path.join(output_dir, f"hold_close_model__{timestamp}.joblib")
    joblib.dump(pipeline, model_path)

    metadata = {
        "features": available_features,
        "timestamp": timestamp,
        "metrics": metrics,
        "policy_comparison": policy_comparison,
        "C": C,
    }
    metadata_path = os.path.join(output_dir, f"hold_close_metadata__{timestamp}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")

    return {
        "pipeline": pipeline, "metrics": metrics, "policy_comparison": policy_comparison,
        "model_path": model_path, "metadata_path": metadata_path, "features": available_features,
    }


if __name__ == "__main__":
    train_hold_close_model(search_root=os.getcwd(), output_dir=os.path.join(os.getcwd(), "hold_close_model"))

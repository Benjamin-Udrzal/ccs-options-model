import re
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os.path
import optuna
import joblib
from pathlib import Path
import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import shap
import matplotlib.pyplot as plt
import inspect
from sklearn.base import RegressorMixin

###############################################################################
# -------------------------- TRAINING CONFIGURATION ---------------------------
###############################################################################
#
# train_model() below picks up the LATEST committed "<date> <run_reason> Model Run" folder by
# default (the same naming convention func_import.py uses) — pass train_import_folder explicitly
# to pin a specific dated run instead. See the function's docstring for all parameters.

model_output_folder = 'Options Trading Model'
import_outputs_folder = 'train Import Outputs'
training_dataset_prefix = '2.CCS_Training_Data'
testing_dataset_prefix = '1.CCS_Testing_Data'

###############################################################################
# -------------------------- MODEL CONFIGURATION ------------------------------
###############################################################################

DROP_COLS = [
    "Ticker", "Share_Price_Date", "Exit_Date", "Call_strike_price",
    "Call_1_expiration_date", "Call_2_expiration_date",
    "Day_1_Cashflow", "Day_2_Cashflow", "Call_1_P&L", "Call_2_P&L"
]

TARGET = "Overall Profit"
DATE_COL = "Share_Price_Date"  # used only for time-ordering the CV splits, then dropped from X

N_CV_SPLITS = 4  # TimeSeriesSplit folds for hyperparameter search

# Small, hand-picked, theoretically-motivated feature set — NOW THE PRIMARY CLASSIFIER'S feature
# set, not just a sanity check, based on walk-forward evidence it generalizes more reliably than
# full XGBoost at this sample size. Only features actually present in the data are used.
SIMPLE_SANITY_CHECK_FEATURES = [
    'IV_Term_Slope', 'IV_Term_Ratio', 'Theta_Ratio',
    'DTE_Gap', 'RV_14', 'Vol_Regime_Ratio', 'IV_Rank_26', 'IV_vs_RV_Spread'
]
# NOTE: 'Moneyness' and 'Vega_Ratio' were dropped — the latest run's coefficient diagnostic
# showed both consistently near zero (|coef| < 0.05), i.e. dead weight not contributing signal.
# The remaining 8 features carry a coherent, economically sensible story for this strategy:
# Theta_Ratio, RV_14, IV_vs_RV_Spread, and DTE_Gap were the strongest, which lines up with what
# you'd expect to matter for a vol-selling calendar spread (decay differential, realized vol,
# the vol-selling edge itself, and structural leg spacing).

###############################################################################
# ------------------------------ HELPERS --------------------------------------
###############################################################################

def find_dataset(prefix, target_directory):
    target_directory_v2 = Path(target_directory)
    files = list(target_directory_v2.glob(f"{prefix}*.csv"))
    if len(files) == 0:
        return None
    if len(files) > 1:
        raise ValueError(f"Multiple CSV files found starting with '{prefix}' in {target_directory}")
    return files[0]


def compute_sample_weights(y):
    return 1.0 + (y < 0) * (1 + (-y / max(1, y.std())).clip(0, 10))


def _fit_with_compat(model, X_tr, y_tr, w_tr, X_val, y_val, w_val, early_rounds=50):
    fit_sig = inspect.signature(model.fit)
    fit_params = set(fit_sig.parameters.keys())

    common_kwargs = {"eval_set": [(X_val, y_val)], "verbose": 0}
    if w_tr is not None and "sample_weight" in fit_params:
        common_kwargs["sample_weight"] = w_tr
    if w_val is not None and "sample_weight_eval_set" in fit_params:
        common_kwargs["sample_weight_eval_set"] = [w_val]

    if "callbacks" in fit_params:
        try:
            common_kwargs["callbacks"] = [xgb.callback.EarlyStopping(rounds=early_rounds, save_best=True)]
            model.fit(X_tr, y_tr, **common_kwargs)
            return
        except TypeError:
            pass

    if "early_stopping_rounds" in fit_params:
        try:
            common_kwargs["early_stopping_rounds"] = early_rounds
            model.fit(X_tr, y_tr, **common_kwargs)
            return
        except TypeError:
            pass

    safe_kwargs = {k: v for k, v in common_kwargs.items() if k in fit_params}
    model.fit(X_tr, y_tr, **safe_kwargs)


def find_latest_train_run_folder(search_root, run_reason='train'):
    """
    Recursively search `search_root` for a folder matching the naming convention func_import
    uses, e.g. "2026-08-13__13.37 train Model Run", and return the path to the most recent one
    (by the timestamp embedded in the folder name). Returns None if none are found.
    """
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}__\d{2}\.\d{2}) ' + re.escape(run_reason) + r' Model Run$')
    candidates = []
    for p in Path(search_root).rglob('*'):
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                candidates.append((m.group(1), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])  # "YYYY-MM-DD__HH.MM" sorts correctly as plain strings
    return candidates[-1][1]


def resolve_train_run_folder(train_import_folder, search_root, run_reason='train'):
    """
    Resolve the actual folder to train from:
      - if train_import_folder is a real path on disk, use it directly
      - if it's just a folder NAME, search for it under search_root
      - if it's None, auto-detect the latest "<date> <run_reason> Model Run" folder under search_root
    """
    if train_import_folder:
        direct_path = Path(train_import_folder)
        if direct_path.is_dir():
            return direct_path
        matches = [p for p in Path(search_root).rglob(train_import_folder) if p.is_dir()]
        if not matches:
            raise FileNotFoundError(
                f"Could not find a folder named '{train_import_folder}' anywhere under {search_root}.")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple folders named '{train_import_folder}' found under {search_root}: {matches}")
        return matches[0]

    latest = find_latest_train_run_folder(search_root, run_reason=run_reason)
    if latest is None:
        raise FileNotFoundError(
            f"No '<date> {run_reason} Model Run' folder found anywhere under {search_root}. "
            f"Either commit one first, or pass train_import_folder explicitly.")
    print(f"No train_import_folder specified — auto-detected latest: {latest}")
    return latest


###############################################################################
# ----------------------- HYPERPARAMETER OPTIMISATION -------------------------
###############################################################################
# NOTE: X, y, sample_weights passed in here must already be sorted by date
# (which they are, since df_train was sorted above and indices carried through).

def optimize_regression(X, y, sample_weights, n_trials=50):
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_split_loss": trial.suggest_float("min_split_loss", 0, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 20),
            "alpha": trial.suggest_float("alpha", 0, 5),
            "lambda": trial.suggest_float("lambda", 0, 5),
        }

        tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_tr, w_val = sample_weights.iloc[train_idx], sample_weights.iloc[val_idx]

            model = xgb.XGBRegressor(**params, enable_categorical=True)
            _fit_with_compat(model, X_tr, y_tr, w_tr, X_val, y_val, w_val, early_rounds=50)
            preds = model.predict(X_val)
            scores.append(mean_absolute_error(y_val, preds))

        return sum(scores) / len(scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    print("Best regression params:", study.best_params)
    return study.best_params


###############################################################################
# ------------------------- FEATURE IMPORTANCE PRUNING ------------------------
###############################################################################

def prune_features(X, y):
    print("Running feature importance pruning...")
    model = xgb.XGBRegressor(objective="reg:squarederror", enable_categorical=True)
    model.fit(X, y)
    importances = pd.Series(model.feature_importances_, index=X.columns)
    keep_features = importances[importances > 0.001].index.tolist()
    print(f"Pruned from {len(X.columns)} → {len(keep_features)} features")
    return X[keep_features], keep_features


def func_train_model(
        train_import_folder: str = None,
        # Name (or full path) of a specific "<date> train Model Run" folder to train from.
        # Leave as None to auto-detect the LATEST one committed under search_root.
        search_root: str = None,
        # Where to search for train-run folders (and, by default, where the found folder lives).
        # Leave as None to search next to this script — matches where func_import.py's daily
        # workflow checks out the repo, so this "just works" when run from the repo root.
        output_dir: str = None,
        # Override where model artifacts are saved. Leave as None to save them inside the same
        # dated folder the training data came from (recommended — keeps a model traceable to
        # exactly the data version it was trained on).
        run_reason: str = 'train',
        # Matches the run_reason used when the training data was imported (func_import's folders
        # are named "<date> {run_reason} Model Run"). Only change this if you're training from a
        # differently-labelled import run.
        run_test: bool = True,
        run_walk_forward: bool = True,
        n_walk_forward_splits: int = 5,
        min_walk_forward_train_fraction: float = 0.5,
        n_null_shuffles: int = 20,
        ):
    """
    Train the Call Calendar Spread models from a committed training dataset.

    By default, finds the most recently committed "<date> train Model Run" folder anywhere under
    search_root (or next to this script if not given) and trains on it. Pass train_import_folder
    to pin a specific one instead (either its bare folder name, or a full path).

    Returns a dict with the trained models, metrics, and the paths everything was saved to.
    """
    # --- Resolve which "train Model Run" folder to train from, and where to save outputs ---
    search_root = search_root if search_root else os.path.dirname(os.path.abspath(__file__))
    found_folder_path = resolve_train_run_folder(train_import_folder, search_root, run_reason=run_reason)

    target_directory = os.path.join(found_folder_path, import_outputs_folder)
    # By default, save model artifacts INSIDE the same dated folder the training data came from, so
    # a model is always traceable back to exactly the data it was trained on. Pass output_dir to
    # override this (e.g. to save somewhere else entirely).
    output_directory = output_dir if output_dir else os.path.join(found_folder_path, model_output_folder)

    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d__%H%M")
    OUTPUT_MODEL_REG_FILE = f"xgb_model_reg__{TIMESTAMP}.json"
    OUTPUT_MODEL_CLS_PIPELINE_FILE = f"cls_pipeline__{TIMESTAMP}.joblib"
    OUTPUT_MODEL_WIN_FILE = f"xgb_model_win__{TIMESTAMP}.json"
    OUTPUT_MODEL_LOSS_FILE = f"xgb_model_loss__{TIMESTAMP}.json"
    OUTPUT_METADATA_FILE = f"model_metadata__{TIMESTAMP}.json"
    OUTPUT_TEST_PREDICTIONS_FILE = f"model_test_predictions__{TIMESTAMP}.csv"

    # ---------------------------- LOAD DATA --------------------------------------
    ###############################################################################

    TRAIN_PATH = find_dataset(training_dataset_prefix, target_directory)
    TEST_PATH = find_dataset(testing_dataset_prefix, target_directory)

    print("\nSearching for training dataset...")
    if TRAIN_PATH is None:
        raise FileNotFoundError(f"No training dataset found starting with '{training_dataset_prefix}' in {target_directory}.")
    print(f"Training dataset: {TRAIN_PATH}")

    print("\nSearching for testing dataset...")
    if TEST_PATH is None:
        print("No testing dataset found — proceeding without test evaluation.\n")
    else:
        print(f"Testing dataset found: {TEST_PATH}\n")

    print("Loading training file...")
    df_train = pd.read_csv(TRAIN_PATH)

    if TARGET not in df_train.columns:
        raise ValueError(f"Missing target column {TARGET}.")

    # Sort by entry date BEFORE splitting, so TimeSeriesSplit respects chronology
    # (random shuffling here would let overlapping-holding-period trades leak across folds).
    df_train[DATE_COL] = pd.to_datetime(df_train[DATE_COL])
    df_train = df_train.sort_values(by=DATE_COL).reset_index(drop=True)

    df_train_model = df_train.drop(columns=[c for c in DROP_COLS if c in df_train.columns])

    X_train_full = df_train_model.drop(columns=[TARGET])
    cat_cols = X_train_full.select_dtypes(include=["object"]).columns
    category_levels = {}  # remember exact category values used at train time, so the test/inference
                           # scripts can apply the SAME encoding rather than re-deriving their own —
                           # independently re-deriving categories on a test set is what causes
                           # XGBoost's "Invalid new DataFrame input" categorical mismatch error, since
                           # the same string values can get assigned different internal codes.
    dropped_empty_cat_cols = []
    for col in cat_cols:
        # An object column that is entirely NaN in this dataset (e.g. Days_To_Next_Earnings when the
        # earnings endpoint isn't available on your plan) gets ZERO categories once cast to 'category'
        # dtype, which crashes XGBoost's categorical handling ("cannot call vectorize on size 0
        # inputs"). Drop any such column here — it carries no information anyway — rather than
        # crashing training.
        if X_train_full[col].notna().sum() == 0:
            dropped_empty_cat_cols.append(col)
            X_train_full = X_train_full.drop(columns=[col])
            continue
        X_train_full[col] = X_train_full[col].astype("category")
        category_levels[col] = X_train_full[col].cat.categories.tolist()

    if dropped_empty_cat_cols:
        print(f"Dropped columns with no non-null values (would crash XGBoost's categorical handling): "
              f"{dropped_empty_cat_cols}")

    y_train_reg = df_train_model[TARGET]
    weights = compute_sample_weights(y_train_reg)
    y_train_cls = (y_train_reg > 0).astype(int)

    ###############################################################################
    # --------------------------- FULL OPTIMISATION PROCESS -----------------------
    ###############################################################################

    X_train_pruned, kept_features = prune_features(X_train_full, y_train_reg)

    print("Optimising regression model (all outcomes)...")
    XGB_REG_PARAMS = optimize_regression(X_train_pruned, y_train_reg, weights, n_trials=50)

    # --- Primary classifier: calibrated logistic regression on the small hand-picked feature set ---
    # (replaces the XGBoost classifier — see header note for why).
    cls_features = [c for c in SIMPLE_SANITY_CHECK_FEATURES if c in X_train_full.columns]
    missing_cls_features = [c for c in SIMPLE_SANITY_CHECK_FEATURES if c not in X_train_full.columns]
    if missing_cls_features:
        print(f"NOTE: these hand-picked classifier features aren't present in this dataset and will "
              f"be skipped: {missing_cls_features}")
    print(f"Training primary classifier (calibrated logistic regression) on features: {cls_features}")

    X_train_cls = X_train_full[cls_features].astype(float)

    # --- EV decomposition sub-models: separately model win magnitude and loss magnitude ---
    win_mask = y_train_reg > 0
    loss_mask = ~win_mask

    print(f"Optimising win-magnitude model on {win_mask.sum()} winning trades...")
    if win_mask.sum() >= 30:
        XGB_WIN_PARAMS = optimize_regression(X_train_pruned[win_mask], y_train_reg[win_mask], weights[win_mask], n_trials=30)
    else:
        print("WARNING: fewer than 30 winning trades — using default params instead of tuning (too little data to tune reliably).")
        XGB_WIN_PARAMS = {"objective": "reg:squarederror", "max_depth": 3, "learning_rate": 0.05, "n_estimators": 200}

    print(f"Optimising loss-magnitude model on {loss_mask.sum()} losing trades...")
    if loss_mask.sum() >= 30:
        XGB_LOSS_PARAMS = optimize_regression(X_train_pruned[loss_mask], y_train_reg[loss_mask], weights[loss_mask], n_trials=30)
    else:
        print("WARNING: fewer than 30 losing trades — using default params instead of tuning (too little data to tune reliably).")
        XGB_LOSS_PARAMS = {"objective": "reg:squarederror", "max_depth": 3, "learning_rate": 0.05, "n_estimators": 200}

    ###############################################################################
    # ------------------------------- TRAIN FINAL MODELS --------------------------
    ###############################################################################

    if RegressorMixin not in xgb.XGBRegressor.__bases__:
        xgb.XGBRegressor.__bases__ += (RegressorMixin,)

    print("Training final regression model (all outcomes)...")
    model_reg = xgb.XGBRegressor(**XGB_REG_PARAMS, enable_categorical=True)
    model_reg.fit(X_train_pruned, y_train_reg, sample_weight=weights)

    print("Training final classification pipeline (scaler + calibrated logistic regression)...")
    # CalibratedClassifierCV internally cross-validates to produce well-calibrated probabilities —
    # directly addressing the miscalibration seen with the XGBoost classifier (predicted 55-65%
    # buckets actually winning only 40-47% of the time). 'sigmoid' (Platt scaling) is used since it
    # behaves better than isotonic on smaller datasets.
    cls_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", CalibratedClassifierCV(LogisticRegression(max_iter=2000, C=1.0), method="sigmoid", cv=5))
    ])
    cls_pipeline.fit(X_train_cls, y_train_cls)

    # Also fit a plain (uncalibrated) logistic regression on the same scaled data purely for
    # coefficient inspection — CalibratedClassifierCV wraps multiple internal folds, so it doesn't
    # expose a single clean coefficient vector the way a plain LogisticRegression does.
    coef_scaler = StandardScaler()
    X_train_cls_scaled = coef_scaler.fit_transform(X_train_cls)
    coef_model = LogisticRegression(max_iter=2000, C=1.0)
    coef_model.fit(X_train_cls_scaled, y_train_cls)

    print("Training final win-magnitude model...")
    model_win = xgb.XGBRegressor(**{k: v for k, v in XGB_WIN_PARAMS.items() if k != "objective"},
                                  objective="reg:squarederror", enable_categorical=True)
    model_win.fit(X_train_pruned[win_mask], y_train_reg[win_mask], sample_weight=weights[win_mask])

    print("Training final loss-magnitude model...")
    model_loss = xgb.XGBRegressor(**{k: v for k, v in XGB_LOSS_PARAMS.items() if k != "objective"},
                                   objective="reg:squarederror", enable_categorical=True)
    model_loss.fit(X_train_pruned[loss_mask], y_train_reg[loss_mask], sample_weight=weights[loss_mask])

    ###############################################################################
    # ------------------------------- SAVE ARTIFACTS ------------------------------
    ###############################################################################

    print("Saving models & metadata...")
    Path(output_directory).mkdir(parents=True, exist_ok=True)

    for m, fname in [(model_reg, OUTPUT_MODEL_REG_FILE),
                      (model_win, OUTPUT_MODEL_WIN_FILE), (model_loss, OUTPUT_MODEL_LOSS_FILE)]:
        booster = m.get_booster()
        booster.feature_names = X_train_pruned.columns.tolist()
        booster.save_model(os.path.join(output_directory, fname))

    joblib.dump(cls_pipeline, os.path.join(output_directory, OUTPUT_MODEL_CLS_PIPELINE_FILE))

    metadata = {
        "timestamp": TIMESTAMP,
        "kept_features": kept_features,
        "cls_features": cls_features,
        "drop_columns": DROP_COLS,
        "training_columns": list(X_train_pruned.columns),
        "category_levels": category_levels,
        "target": TARGET,
        "xgb_reg_params": XGB_REG_PARAMS,
        "xgb_win_params": XGB_WIN_PARAMS,
        "xgb_loss_params": XGB_LOSS_PARAMS,
        "model_files": {
            "regression": OUTPUT_MODEL_REG_FILE,
            "classification_pipeline": OUTPUT_MODEL_CLS_PIPELINE_FILE,
            "win_magnitude": OUTPUT_MODEL_WIN_FILE,
            "loss_magnitude": OUTPUT_MODEL_LOSS_FILE,
        }
    }

    with open(os.path.join(output_directory, OUTPUT_METADATA_FILE), "w") as f:
        json.dump(metadata, f, indent=4)

    print("Models saved successfully.")

    ###############################################################################
    # ------------------------------- TEST EVALUATION -----------------------------
    ###############################################################################

    if run_test and TEST_PATH:
        print("Running test evaluation...")
        df_test_full = pd.read_csv(TEST_PATH)

        for col in df_test_full.select_dtypes(include=["object"]).columns:
            if col in category_levels:
                df_test_full[col] = pd.Categorical(df_test_full[col], categories=category_levels[col])
            else:
                # Not a real categorical feature during training — likely a numeric column that
                # degraded to object dtype in this file (e.g. an all-empty column). Coerce back to
                # numeric rather than risk a zero-category crash from blindly categorizing it.
                coerced = pd.to_numeric(df_test_full[col], errors='coerce')
                if coerced.notna().sum() > 0 or df_test_full[col].notna().sum() == 0:
                    df_test_full[col] = coerced
                else:
                    df_test_full[col] = df_test_full[col].astype("category")

        non_model_columns = df_test_full[[c for c in DROP_COLS if c in df_test_full.columns] + ["Overall Profit"]]
        df_test = df_test_full[kept_features]
        df_test_cls = df_test_full[cls_features].astype(float)

        preds_reg = model_reg.predict(df_test)
        preds_prob = cls_pipeline.predict_proba(df_test_cls)[:, 1]
        preds_cls = (preds_prob >= 0.5).astype(int)
        preds_win = model_win.predict(df_test)
        preds_loss = model_loss.predict(df_test)

        # Expected value decomposition: P(profit)*E[win magnitude] + (1-P(profit))*E[loss magnitude]
        preds_ev = preds_prob * preds_win + (1 - preds_prob) * preds_loss

        df_out = df_test.copy()
        df_out.insert(0, "Expected_Value", preds_ev)
        df_out.insert(0, "Predicted_Loss_If_Loss", preds_loss)
        df_out.insert(0, "Predicted_Win_If_Win", preds_win)
        df_out.insert(0, "Pred_Prob_Profitable", preds_prob)
        df_out.insert(0, "Predicted_Profit_(single_model)", preds_reg)

        df_out = pd.concat([non_model_columns.reset_index(drop=True), df_out.reset_index(drop=True)], axis=1)
        df_out.to_csv(os.path.join(output_directory, OUTPUT_TEST_PREDICTIONS_FILE), index=False)
        print("Test predictions saved.")

        ###############################################################################
        # --------------------- PERFORMANCE METRICS + CALIBRATION ---------------------
        ###############################################################################

        print("\n===== GENERATING PERFORMANCE METRICS =====\n")

        y_test_reg = df_test_full["Overall Profit"]
        y_test_cls = (y_test_reg > 0).astype(int)

        try:
            rmse = mean_squared_error(y_test_reg, preds_reg, squared=False)
        except TypeError:
            rmse = mean_squared_error(y_test_reg, preds_reg) ** 0.5

        reg_metrics = {
            "MAE_single_model": float(mean_absolute_error(y_test_reg, preds_reg)),
            "RMSE_single_model": float(rmse),
            "R2_single_model": float(r2_score(y_test_reg, preds_reg)),
            "MAE_EV_decomposition": float(mean_absolute_error(y_test_reg, preds_ev)),
        }

        cm = confusion_matrix(y_test_cls, preds_cls)
        cls_metrics = {
            "Accuracy": float(accuracy_score(y_test_cls, preds_cls)),
            "Precision": float(precision_score(y_test_cls, preds_cls, zero_division=0)),
            "Recall": float(recall_score(y_test_cls, preds_cls)),
            "F1": float(f1_score(y_test_cls, preds_cls)),
            "ROC_AUC": float(roc_auc_score(y_test_cls, preds_prob)) if len(set(y_test_cls)) > 1 else None,
            "Confusion_Matrix": cm.tolist(),
        }

        # Calibration: bucket predicted probabilities and compare to actual win rate in each bucket.
        # If the model is well-calibrated, "predicted ~0.7" buckets should have an actual win rate ~0.7.
        calib_df = pd.DataFrame({"pred_prob": preds_prob, "actual": y_test_cls})
        calib_df["bucket"] = pd.cut(calib_df["pred_prob"], bins=[0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0], include_lowest=True)
        calibration_table = calib_df.groupby("bucket", observed=False).agg(
            n=("actual", "size"), predicted_mean=("pred_prob", "mean"), actual_win_rate=("actual", "mean")
        ).reset_index()
        calibration_table["bucket"] = calibration_table["bucket"].astype(str)

        performance_output_file = os.path.join(output_directory, f"model_performance__{TIMESTAMP}.json")
        with open(performance_output_file, "w") as f:
            json.dump({
                "timestamp": TIMESTAMP,
                "regression_metrics": reg_metrics,
                "classification_metrics": cls_metrics,
                "calibration_table": calibration_table.to_dict(orient="records"),
            }, f, indent=4)

        print(f"\nSaved performance metrics (incl. calibration table):\n{performance_output_file}\n")
        print("Calibration table (predicted vs. actual win rate by bucket):")
        print(calibration_table)

        # Calibration plot
        plt.figure()
        plt.plot(calibration_table["predicted_mean"], calibration_table["actual_win_rate"], marker="o", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        plt.xlabel("Predicted P(profit)")
        plt.ylabel("Actual win rate")
        plt.title("Calibration: predicted vs. actual")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, "Calibration_plot.png"))
        plt.close()
        print("Saved calibration plot.")

        ###############################################################################
        # ----------------------------- SHAP (REGRESSION) -----------------------------
        ###############################################################################

        print("Computing SHAP values for regression model...")
        explainer_reg = shap.TreeExplainer(model_reg)
        shap_values_reg = explainer_reg(df_test)

        plt.figure()
        shap.summary_plot(shap_values_reg, df_test, max_display=25, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, "SHAP_summary_regression.png"))
        plt.close()

        plt.figure()
        shap.summary_plot(shap_values_reg, df_test, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, "SHAP_feature_importance_regression.png"))
        plt.close()

        print("Saved regression SHAP plots.")

        ###############################################################################
        # ------------------- CLASSIFIER COEFFICIENTS (LOGISTIC REGRESSION) -----------
        ###############################################################################
        # A linear model's coefficients ARE its explanation — no SHAP needed. Positive coefficient =
        # higher values of that (standardized) feature push toward predicting profit; negative =
        # push toward predicting loss. This is arguably more directly interpretable than SHAP values
        # would have been for the old XGBoost classifier.

        print("Plotting classifier coefficients...")
        coef_series = pd.Series(coef_model.coef_[0], index=cls_features).sort_values()

        plt.figure(figsize=(8, max(3, 0.4 * len(coef_series))))
        colors = ["tab:red" if v < 0 else "tab:blue" for v in coef_series.values]
        plt.barh(coef_series.index, coef_series.values, color=colors)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("Standardized logistic regression coefficient")
        plt.title("Classifier feature coefficients (blue = pushes toward profit)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, "Classifier_coefficients.png"))
        plt.close()

        coef_output_file = os.path.join(output_directory, f"classifier_coefficients__{TIMESTAMP}.json")
        with open(coef_output_file, "w") as f:
            json.dump({"intercept": float(coef_model.intercept_[0]),
                        "coefficients": coef_series.to_dict()}, f, indent=4)

        print(f"Saved classifier coefficients: {coef_output_file}")

        # --- Weak-feature diagnostic ---
        # A near-zero standardized coefficient means the model found essentially no usable
        # relationship between that feature and the outcome on this data — this is a direct,
        # concrete way to see whether a feature is "carrying its weight" rather than inferring it
        # indirectly from calibration behavior.
        WEAK_COEF_THRESHOLD = 0.05
        abs_coefs = coef_series.abs().sort_values(ascending=False)
        print("\nClassifier features ranked by |standardized coefficient| (strongest signal first):")
        for feat, val in abs_coefs.items():
            flag = "  <- WEAK (near zero)" if val < WEAK_COEF_THRESHOLD else ""
            print(f"  {feat:25s} {coef_series[feat]:+.4f}{flag}")

        weak_features = abs_coefs[abs_coefs < WEAK_COEF_THRESHOLD].index.tolist()
        if len(weak_features) == len(cls_features):
            print("\nWARNING: every feature has a near-zero coefficient — this model found essentially "
                  "no linear signal in the current feature set on this data. This is consistent with "
                  "(and helps explain) predictions collapsing toward the base rate in the calibration "
                  "table above.")
        elif weak_features:
            print(f"\nWeak (near-zero) features that may be worth dropping or replacing: {weak_features}")
        else:
            print("\nAll classifier features show non-trivial coefficients.")

        print("\n===== METRICS + SHAP GENERATION COMPLETE =====\n")

    ###############################################################################
    # ------------------------- WALK-FORWARD VALIDATION ---------------------------
    ###############################################################################
    # A single train/test split is one data point — with a small test set, you can't tell
    # "no real edge" apart from "a modest edge that happened to land on an unlucky slice of
    # time." This re-combines train+test chronologically and re-splits it into several
    # rolling, EXPANDING-WINDOW folds (train on everything up to date X, test on the period
    # right after, then move X forward and repeat). Each fold also trains a second model on
    # the SAME data with labels randomly shuffled — that "null" AUC shows what performance
    # looks like from pure chance at this sample size, giving you something concrete to
    # compare your real AUC against rather than eyeballing whether 0.55 or 0.58 "seems okay."

    if run_walk_forward and TEST_PATH:
        print("\n===== RUNNING WALK-FORWARD VALIDATION =====\n")

        df_test_raw = pd.read_csv(TEST_PATH)
        df_test_raw[DATE_COL] = pd.to_datetime(df_test_raw[DATE_COL])

        combined = pd.concat([df_train, df_test_raw], ignore_index=True, sort=False)
        combined = combined.sort_values(by=DATE_COL).reset_index(drop=True)
        wf_cls_cols = [c for c in cls_features if c in combined.columns]
        combined = combined.dropna(subset=wf_cls_cols + [TARGET]).reset_index(drop=True)

        n_combined = len(combined)
        min_train_size = int(min_walk_forward_train_fraction * n_combined)
        remaining = n_combined - min_train_size
        fold_size = max(10, remaining // n_walk_forward_splits)

        walk_forward_results = []
        fold_start = min_train_size
        fold_num = 0

        while fold_start < n_combined and fold_num < n_walk_forward_splits:
            fold_end = min(fold_start + fold_size, n_combined)
            if fold_end - fold_start < 10:  # too few test rows in this fold to draw any conclusion
                break
            fold_num += 1

            train_fold = combined.iloc[:fold_start]
            test_fold = combined.iloc[fold_start:fold_end]

            X_tr_fold = train_fold[wf_cls_cols].astype(float)
            y_tr_fold_cls = (train_fold[TARGET] > 0).astype(int)
            X_te_fold = test_fold[wf_cls_cols].astype(float)
            y_te_fold_cls = (test_fold[TARGET] > 0).astype(int)

            fold_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", CalibratedClassifierCV(LogisticRegression(max_iter=2000, C=1.0), method="sigmoid", cv=5))
            ])
            fold_pipeline.fit(X_tr_fold, y_tr_fold_cls)
            fold_preds = fold_pipeline.predict_proba(X_te_fold)[:, 1]

            fold_auc = roc_auc_score(y_te_fold_cls, fold_preds) if len(set(y_te_fold_cls)) > 1 else None
            fold_acc = accuracy_score(y_te_fold_cls, (fold_preds >= 0.5).astype(int))

            # Permutation test: fit N_NULL_SHUFFLES models on the SAME fold with training labels
            # randomly shuffled each time. This gives an empirical DISTRIBUTION of "what AUC looks
            # like from pure chance at this fold's sample size" — a single shuffle is itself a noisy
            # estimate, so we want to know if the real AUC beats MOST of the null distribution, not
            # just one arbitrary draw.
            null_aucs_this_fold = []
            for shuffle_i in range(n_null_shuffles):
                y_tr_fold_shuffled = y_tr_fold_cls.sample(frac=1, random_state=fold_num * 1000 + shuffle_i).reset_index(drop=True)
                null_pipeline = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", CalibratedClassifierCV(LogisticRegression(max_iter=2000, C=1.0), method="sigmoid", cv=5))
                ])
                null_pipeline.fit(X_tr_fold, y_tr_fold_shuffled)
                null_preds = null_pipeline.predict_proba(X_te_fold)[:, 1]
                if len(set(y_te_fold_cls)) > 1:
                    null_aucs_this_fold.append(roc_auc_score(y_te_fold_cls, null_preds))

            if null_aucs_this_fold:
                null_auc_mean = float(np.mean(null_aucs_this_fold))
                null_auc_p90 = float(np.percentile(null_aucs_this_fold, 90))
                # Empirical percentile of the real AUC within the null distribution: e.g. 0.85 means
                # the real model beat 85% of pure-chance draws at this sample size.
                percentile_rank = float(np.mean([fold_auc > n for n in null_aucs_this_fold])) if fold_auc is not None else None
                beats_null_p90 = bool(fold_auc is not None and fold_auc > null_auc_p90)
            else:
                null_auc_mean = null_auc_p90 = percentile_rank = None
                beats_null_p90 = None

            walk_forward_results.append({
                "fold": fold_num,
                "train_start": str(train_fold[DATE_COL].min().date()),
                "train_end": str(train_fold[DATE_COL].max().date()),
                "test_start": str(test_fold[DATE_COL].min().date()),
                "test_end": str(test_fold[DATE_COL].max().date()),
                "n_train": len(train_fold),
                "n_test": len(test_fold),
                "auc": fold_auc,
                "accuracy": float(fold_acc),
                "null_auc_mean": null_auc_mean,
                "null_auc_p90": null_auc_p90,
                "percentile_rank_vs_null": percentile_rank,
                "beats_null_p90": beats_null_p90,
            })

            verdict = "BEATS null p90" if beats_null_p90 else ("below null p90" if beats_null_p90 is not None else "n/a")
            print(f"Fold {fold_num}: train {len(train_fold)} rows ({train_fold[DATE_COL].min().date()} to "
                  f"{train_fold[DATE_COL].max().date()}), test {len(test_fold)} rows "
                  f"({test_fold[DATE_COL].min().date()} to {test_fold[DATE_COL].max().date()}) "
                  f"-> Real AUC={fold_auc}, Accuracy={fold_acc:.3f}, Null mean={null_auc_mean}, "
                  f"Null p90={null_auc_p90}, Percentile vs null={percentile_rank}, Verdict: {verdict}")

            fold_start = fold_end

        if walk_forward_results:
            aucs = [r["auc"] for r in walk_forward_results if r["auc"] is not None]
            null_means = [r["null_auc_mean"] for r in walk_forward_results if r["null_auc_mean"] is not None]
            beats_flags = [r["beats_null_p90"] for r in walk_forward_results if r["beats_null_p90"] is not None]

            print(f"\nWalk-forward Calibrated LogReg AUC across {len(aucs)} folds: mean={sum(aucs)/len(aucs):.3f}, "
                  f"min={min(aucs):.3f}, max={max(aucs):.3f}")
            if null_means:
                print(f"Null distribution mean AUC (across {n_null_shuffles} shuffles/fold): "
                      f"mean={sum(null_means)/len(null_means):.3f}")
            if beats_flags:
                n_beats = sum(beats_flags)
                print(f"Folds where real AUC beat the 90th percentile of its null distribution: "
                      f"{n_beats} / {len(beats_flags)}")
                if n_beats == 0:
                    print("-> No fold showed a real AUC that a permutation test would call statistically "
                          "distinguishable from chance. Treat this as 'no reliable edge detected yet.'")
                elif n_beats == len(beats_flags):
                    print("-> EVERY fold beat its null p90 — this is meaningfully stronger evidence of a "
                          "real, durable signal than a single good run would be.")
                else:
                    print(f"-> Mixed result ({n_beats}/{len(beats_flags)} folds) — edge may be present but "
                          f"inconsistent across time periods; treat with caution.")

            walk_forward_output_file = os.path.join(output_directory, f"walk_forward_validation__{TIMESTAMP}.json")
            with open(walk_forward_output_file, "w") as f:
                json.dump({"timestamp": TIMESTAMP, "folds": walk_forward_results}, f, indent=4)
            print(f"Saved walk-forward validation results: {walk_forward_output_file}\n")
        else:
            print("Not enough data to run walk-forward validation with the configured settings "
                  "(try lowering N_WALK_FORWARD_SPLITS or MIN_WALK_FORWARD_TRAIN_FRACTION).\n")

        print("===== WALK-FORWARD VALIDATION COMPLETE =====\n")

    print("DONE.")


    # ------------------------------------------------------------------
    # Return everything useful as function outputs, so other code (or a
    # notebook, or another step in the same workflow) can use it directly.
    # ------------------------------------------------------------------
    return {
        "found_folder": str(found_folder_path),
        "target_directory": str(target_directory),
        "output_dir": str(output_directory),
        "timestamp": TIMESTAMP,
        "kept_features": kept_features,
        "cls_features": cls_features,
        "metadata": metadata,
        "reg_metrics": reg_metrics if run_test and TEST_PATH else None,
        "cls_metrics": cls_metrics if run_test and TEST_PATH else None,
        "calibration_table": calibration_table.to_dict(orient="records") if run_test and TEST_PATH else None,
        "walk_forward_results": walk_forward_results if run_walk_forward and TEST_PATH else None,
        "model_reg": model_reg,
        "cls_pipeline": cls_pipeline,
        "model_win": model_win,
        "model_loss": model_loss,
    }

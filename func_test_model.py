import os.path
import re
import sys
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from pathlib import Path

###############################################################################
# -------------------------- USER CONFIGURATION -------------------------------
###############################################################################
#

MODEL_FOLDER_INNER = "Options Trading Model"
IMPORT_OUTPUTS_FOLDER_PATTERN = "{run_reason} Import Outputs"
TEST_DATASET_PREFIX = "1.CCS_Testing_Data"

PROF_PROB_BOUNDARY = 0.5   # minimum P(profit) to consider opening a trade
EV_BOUNDARY = 0.0          # minimum expected value ($ per contract, x100 multiplier already applied) to open a trade
MAX_KELLY_FRACTION = 0.25  # hard cap on suggested position size — treat this as a ceiling, not a target


###############################################################################
# ------------------------------ HELPERS --------------------------------------
###############################################################################

def find_latest_model_run_folder(search_root, run_reason='train'):
    """
    Recursively search `search_root` for a folder matching func_import's naming convention,
    e.g. "2026-08-13__13.37 train Model Run", and return the most recent one (by the timestamp
    embedded in the folder name). Returns None if none are found.
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


def resolve_model_run_folder(folder, search_root, run_reason='train'):
    """
    Resolve the actual "<date> <run_reason> Model Run" folder to use:
      - if `folder` is a real path on disk, use it directly
      - if it's just a folder NAME, search for it under search_root
      - if it's None, auto-detect the latest matching folder under search_root
    """
    if folder:
        direct_path = Path(folder)
        if direct_path.is_dir():
            return direct_path
        matches = [p for p in Path(search_root).rglob(folder) if p.is_dir()]
        if not matches:
            raise FileNotFoundError(f"Could not find a folder named '{folder}' anywhere under {search_root}.")
        if len(matches) > 1:
            raise ValueError(f"Multiple folders named '{folder}' found under {search_root}: {matches}")
        return matches[0]

    latest = find_latest_model_run_folder(search_root, run_reason=run_reason)
    if latest is None:
        raise FileNotFoundError(
            f"No '<date> {run_reason} Model Run' folder found anywhere under {search_root}. "
            f"Either commit one first, or pass the folder explicitly.")
    return latest


def find_latest_file(directory, pattern):
    """Glob for `pattern` inside `directory` and return the most recent match by filename
    (works because filenames embed a zero-padded timestamp, e.g. '..._20260813__1410.json')."""
    matches = sorted(Path(directory).glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' found in {directory}")
    return matches[-1]


def find_test_csv(test_data, search_root, run_reason='dtest'):
    """
    Resolve the test CSV to run predictions on:
      - if `test_data` is a path to an actual CSV file, use it directly
      - if it's a folder (name or path), look for TEST_DATASET_PREFIX*.csv inside it
      - if it's None, auto-detect the latest "<date> <run_reason> Model Run" folder and use its
        testing CSV (defaults to the latest daily "dtest" import, not a training run's own
        held-out set)
    """
    if test_data:
        p = Path(test_data)
        if p.is_file():
            return p
        folder = resolve_model_run_folder(test_data, search_root, run_reason=run_reason)
    else:
        folder = resolve_model_run_folder(None, search_root, run_reason=run_reason)
        print(f"No test_data specified — auto-detected latest '{run_reason}' run: {folder}")

    import_outputs_folder = IMPORT_OUTPUTS_FOLDER_PATTERN.format(run_reason=run_reason)
    candidate_dir = folder / import_outputs_folder
    search_dir = candidate_dir if candidate_dir.is_dir() else folder
    matches = list(search_dir.glob(f"{TEST_DATASET_PREFIX}*.csv"))
    if not matches:
        raise FileNotFoundError(f"No '{TEST_DATASET_PREFIX}*.csv' file found in {search_dir}")
    if len(matches) > 1:
        raise ValueError(f"Multiple '{TEST_DATASET_PREFIX}*.csv' files found in {search_dir}: {matches}")
    return matches[0]


###############################################################################
# ------------------------------ MAIN FUNCTION ---------------------------------
###############################################################################

def test_model(
        model_folder: str = None,
        # Name (or full path) of a specific "<date> train Model Run" folder holding the trained
        # model to test. Leave None to auto-detect the LATEST one committed under search_root.
        model_run_reason: str = 'train',
        test_data: str = None,
        # Path to a specific test CSV, OR the bare name of a "<date> <run> Model Run" folder to
        # pull "1.CCS_Testing_Data*.csv" from. Leave None to auto-detect the latest matching run
        # (see test_data_run_reason).
        test_data_run_reason: str = 'dtest',
        # Which run_reason's folder to search for when auto-detecting test_data — defaults to
        # the latest daily "dtest" import, not a training run's own held-out test set.
        search_root: str = None,
        output_dir: str = None,
        # Where to save the predictions CSV. Leave None to save inside the same folder the test
        # data came from.
        prof_prob_boundary: float = PROF_PROB_BOUNDARY,
        ev_boundary: float = EV_BOUNDARY,
        max_kelly_fraction: float = MAX_KELLY_FRACTION,
        output_filename: str = "test_predictions_output.csv",
        ):
    """
    Run the trained model against a test dataset and save trade-signal predictions.

    Returns a dict with the predictions DataFrame, the paths used, and the model metadata.
    """
    search_root = search_root if search_root else os.path.dirname(os.path.abspath(__file__))

    ###########################################################################
    # --------------------------- LOAD MODELS & METADATA ----------------------
    ###########################################################################

    print("Loading saved models and metadata...")

    found_model_folder = resolve_model_run_folder(model_folder, search_root, run_reason=model_run_reason)
    model_folder_directory = os.path.join(found_model_folder, MODEL_FOLDER_INNER)

    def load_regressor(path):
        m = xgb.XGBRegressor(enable_categorical=True)
        try:
            m.load_model(str(path))
        except xgb.core.XGBoostError:
            sys.exit(f"ERROR: Unable to load model file: {path}")
        return m

    model_reg = load_regressor(find_latest_file(model_folder_directory, "xgb_model_reg__*.json"))
    model_win = load_regressor(find_latest_file(model_folder_directory, "xgb_model_win__*.json"))
    model_loss = load_regressor(find_latest_file(model_folder_directory, "xgb_model_loss__*.json"))
    cls_pipeline_path = find_latest_file(model_folder_directory, "cls_pipeline__*.joblib")
    cls_pipeline = joblib.load(cls_pipeline_path)

    metadata_path = find_latest_file(model_folder_directory, "model_metadata__*.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    print("Models found in the following folder:")
    print("============================")
    print(model_folder_directory)
    print("============================")

    DROP_COLS = metadata["drop_columns"]
    TRAINING_COLUMNS = metadata["training_columns"]
    CLS_FEATURES = metadata["cls_features"]
    CATEGORY_LEVELS = metadata.get("category_levels", {})

    ###########################################################################
    # ---------------------------- FIND TEST DATA ------------------------------
    ###########################################################################

    print("\nSearching for test dataset...")
    TEST_PATH = find_test_csv(test_data, search_root, run_reason=test_data_run_reason)
    print(f"Test dataset found: {TEST_PATH}")

    ###########################################################################
    # --------------------------- LOAD TEST DATA -------------------------------
    ###########################################################################

    df_test = pd.read_csv(TEST_PATH)

    for col in df_test.select_dtypes(include=["object"]).columns:
        if col in CATEGORY_LEVELS:
            # Use the EXACT categories seen at training time, not a fresh re-derivation from this
            # file — re-deriving independently is what causes XGBoost's categorical mismatch error,
            # since the same string values can otherwise get assigned different internal codes.
            df_test[col] = pd.Categorical(df_test[col], categories=CATEGORY_LEVELS[col])
        else:
            # This column wasn't a real categorical feature during training, so it's almost
            # certainly a numeric column that degraded to object dtype in this specific CSV (e.g.
            # entirely empty in a small single-day dtest file). Coerce back to numeric rather than
            # blindly treating it as a brand-new category — casting an all-NaN column straight to
            # 'category' gives it ZERO categories, which crashes XGBoost with "cannot call
            # vectorize on size 0 inputs".
            coerced = pd.to_numeric(df_test[col], errors='coerce')
            if coerced.notna().sum() > 0 or df_test[col].notna().sum() == 0:
                df_test[col] = coerced
            else:
                df_test[col] = df_test[col].astype("category")

    non_model_columns = df_test.drop(columns=TRAINING_COLUMNS, errors="ignore")
    test_dataset = df_test[TRAINING_COLUMNS]

    missing_cols = set(TRAINING_COLUMNS) - set(test_dataset.columns)
    if missing_cols:
        raise ValueError(f"Test dataset is missing required columns: {missing_cols}")
    else:
        print("Testing dataset has all required data fields for model")

    # XGBoost (model_reg/win/loss) handles NaN natively, but the classifier's LogisticRegression
    # does not — a row missing any of the classifier features (e.g. IV_Rank_26 not having enough
    # rolling history yet) would crash cls_pipeline.predict_proba() otherwise. Drop those rows
    # here instead, with a clear warning, rather than losing the whole run over one ticker.
    cls_nan_mask = df_test[CLS_FEATURES].isna().any(axis=1)
    if cls_nan_mask.any():
        dropped_tickers = df_test.loc[cls_nan_mask, 'Ticker'].tolist() if 'Ticker' in df_test.columns else cls_nan_mask.sum()
        print(f"WARNING: dropping {cls_nan_mask.sum()} row(s) with missing classifier features "
              f"(cannot predict for): {dropped_tickers}")
        df_test = df_test[~cls_nan_mask].reset_index(drop=True)
        non_model_columns = non_model_columns[~cls_nan_mask.values].reset_index(drop=True)
        test_dataset = test_dataset[~cls_nan_mask.values].reset_index(drop=True)

    if test_dataset.empty:
        sys.exit("No rows remain after dropping incomplete classifier features — nothing to predict.")

    X_test = test_dataset[TRAINING_COLUMNS]

    ###########################################################################
    # -------------------------- RUN PREDICTIONS -------------------------------
    ###########################################################################

    print("Running predictions on test dataset...")

    X_test_cls = df_test[CLS_FEATURES].astype(float)

    preds_reg = model_reg.predict(X_test)
    preds_prob = cls_pipeline.predict_proba(X_test_cls)[:, 1]
    preds_win = model_win.predict(X_test)
    preds_loss = model_loss.predict(X_test)

    preds_ev = preds_prob * preds_win + (1 - preds_prob) * preds_loss

    preds_label = ((preds_prob > prof_prob_boundary) & (preds_ev > ev_boundary)).astype(int)

    # --- Reference-only Kelly-fraction sizing suggestion ---
    # f* = p - (1-p)/b, where b = avg_win / |avg_loss|. Clipped to [0, max_kelly_fraction].
    # This is a rough guide, not a sizing instruction — always sanity-check manually and
    # consider using a fraction of full Kelly (e.g. half-Kelly) given model uncertainty.
    b_ratio = preds_win / preds_loss.clip(min=-1e-6) * -1  # avg_win / |avg_loss|, guarding div-by-zero
    b_ratio = np.where(b_ratio <= 0, np.nan, b_ratio)
    kelly_fraction = preds_prob - (1 - preds_prob) / b_ratio
    kelly_fraction = np.clip(np.nan_to_num(kelly_fraction, nan=0.0), 0, max_kelly_fraction)

    ###########################################################################
    # ---------------------------- SAVE PREDICTIONS ----------------------------
    ###########################################################################

    df_out = test_dataset.copy()

    df_out.insert(0, "Suggested_Max_Kelly_Fraction", kelly_fraction)
    df_out.insert(0, "Open_Trade", preds_label)
    df_out.insert(0, "Expected_Value", preds_ev)
    df_out.insert(0, "Predicted_Loss_If_Loss", preds_loss)
    df_out.insert(0, "Predicted_Win_If_Win", preds_win)
    df_out.insert(0, "Pred_Prob_Profitable", preds_prob)
    df_out.insert(0, "Predicted_Profit_(single_model)", preds_reg)

    df_out = pd.concat([non_model_columns.reset_index(drop=True), df_out.reset_index(drop=True)], axis=1)

    # By default, save predictions inside the same folder the test data came from, so the
    # output stays traceable to exactly which day's data and which model produced it.
    save_directory = output_dir if output_dir else os.path.dirname(TEST_PATH)
    os.makedirs(save_directory, exist_ok=True)
    output_path = os.path.join(save_directory, output_filename)

    df_out.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print("Done.")

    return {
        "predictions": df_out,
        "output_path": str(output_path),
        "model_folder": str(found_model_folder),
        "test_data_path": str(TEST_PATH),
        "metadata": metadata,
    }
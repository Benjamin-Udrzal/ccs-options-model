"""
Builds the training dataset for the hold/close model: one row per (position, day held),
covering every day from entry to exit for every historical position, with the features
needed to decide "should I close today?" and a forward-looking label.

SOURCE DATA: reads func_import.py's own "3.CCS_Summary_Data_*.csv" output (found via the same
"latest train Model Run" auto-detection CCS_Model_Train_v7.py already uses) — this file
already has every historical position's ticker, entry/exit dates, strike, both leg expiries,
entry cashflow, and final Overall Profit, computed by func_import.py itself. This script only
READS that file; it never imports or calls func_import.py, so it cannot affect it.

WHAT THIS ADDS ON TOP: for each position, it walks every day between entry and exit and asks
Polygon for that day's stock price and both legs' option prices, to compute a genuine
mark-to-market P&L path — not just the entry and exit snapshots func_import.py already has.
This is the expensive part: roughly (avg holding days) x 3 API calls x (number of positions).
For ~3,000 positions held ~15 days, that's on the order of 130,000 API calls — expect this to
take a long time and to need a Polygon plan with a high enough rate limit. Consider running it
once, saving the result, and only re-running for new positions afterward (see --resume below).

LABEL: for each day (except the final exit day, which is dropped — seeing "today" is the last
day of the position is not something the model would have at inference time), label
close_today = 1 if today's mark-to-market P&L is within `cost_buffer` of the best
mark-to-market P&L achieved on any REMAINING day of the position (today through exit),
else 0. This captures both "take profit at the peak" and "cut losses before they worsen"
under one target, per the reasoning discussed with the user.

Usage:
    from build_hold_close_dataset import build_dataset
    build_dataset(polygon_api_key=..., search_root=..., output_dir=..., cost_buffer=15.0)
"""

import os
import re
import sys
import time
import json
import datetime
import pandas as pd
from pathlib import Path
from polygon import RESTClient

from hold_close_features import compute_snapshot_features, find_option_contract, get_daily_close, get_stock_close


def find_latest_summary_file(search_root, run_reason="train"):
    """Find the latest committed '<date> <run_reason> Model Run' folder (same convention as
    CCS_Model_Train_v7.py) and return the path to its 3.CCS_Summary_Data*.csv file."""
    pattern = re.compile(r'^(\d{4}-\d{2}-\d{2}__\d{2}\.\d{2}) ' + re.escape(run_reason) + r' Model Run$')
    candidates = []
    for p in Path(search_root).rglob('*'):
        if p.is_dir() and pattern.match(p.name):
            candidates.append((pattern.match(p.name).group(1), p))
    if not candidates:
        raise FileNotFoundError(f"No '<date> {run_reason} Model Run' folder found under {search_root}.")
    candidates.sort(key=lambda x: x[0])
    folder = candidates[-1][1]

    import_outputs = folder / f"{run_reason} Import Outputs"
    search_dir = import_outputs if import_outputs.is_dir() else folder
    matches = list(search_dir.glob("3.CCS_Summary_Data*.csv"))
    if not matches:
        raise FileNotFoundError(f"No '3.CCS_Summary_Data*.csv' found in {search_dir}")
    print(f"Using summary file: {matches[0]}")
    return matches[0]


def business_days_between(start_date, end_date):
    """Every calendar day from start to end inclusive, skipping weekends. Doesn't account for
    market holidays — Polygon simply returns no bar for those days, which is handled as a
    missing snapshot (forward-filled from the prior available day) further down."""
    days = pd.bdate_range(start_date, end_date)
    return [d.date() for d in days]


def build_position_snapshots(client, position, cost_buffer, rate_limit_sleep=0.0):
    """Walk one historical position day-by-day and return a list of feature-row dicts
    (one per day, un-labeled — labels are added afterward once the whole path is known)."""
    ticker = position["Ticker"]
    entry_date = position["Share_Price_Date"]
    exit_date = position["Exit_Date"]
    strike = position["Call_strike_price"]
    front_expiry = position["Call_1_expiration_date"]
    back_expiry = position["Call_2_expiration_date"]
    entry_cashflow = position["Day_1_Cashflow"]

    front_ticker = find_option_contract(client, ticker, strike, front_expiry, "call")
    back_ticker = find_option_contract(client, ticker, strike, back_expiry, "call")
    if front_ticker is None or back_ticker is None:
        print(f"  WARNING: could not resolve option contract(s) for {ticker} {strike} "
              f"({front_expiry}/{back_expiry}) — skipping this position.")
        return []

    days = business_days_between(entry_date, exit_date)

    entry_stock_price = get_stock_close(client, ticker, entry_date)
    entry_front_price = get_daily_close(client, front_ticker, entry_date)
    entry_back_price = get_daily_close(client, back_ticker, entry_date)
    if entry_stock_price is None or entry_front_price is None or entry_back_price is None:
        print(f"  WARNING: missing entry-day price data for {ticker} {strike} — skipping.")
        return []

    from hold_close_features import calculate_iv_and_greeks
    entry_front_T = max((front_expiry - entry_date).days, 0) / 365
    entry_back_T = max((back_expiry - entry_date).days, 0) / 365
    entry_front_iv = calculate_iv_and_greeks(entry_stock_price, strike, entry_front_T, entry_front_price, "c")["implied_volatility"]
    entry_back_iv = calculate_iv_and_greeks(entry_stock_price, strike, entry_back_T, entry_back_price, "c")["implied_volatility"]

    rows = []
    last_stock_price, last_front_price, last_back_price = entry_stock_price, entry_front_price, entry_back_price
    peak = entry_cashflow
    mtm_history = []  # (date, mtm_pnl) for computing 1d/3d changes

    for day in days:
        stock_price = get_stock_close(client, ticker, day) or last_stock_price
        front_price = get_daily_close(client, front_ticker, day) or last_front_price
        back_price = get_daily_close(client, back_ticker, day) or last_back_price
        last_stock_price, last_front_price, last_back_price = stock_price, front_price, back_price
        if rate_limit_sleep:
            time.sleep(rate_limit_sleep)

        mtm_1d_ago = mtm_history[-1][1] if len(mtm_history) >= 1 else None
        mtm_3d_ago = mtm_history[-3][1] if len(mtm_history) >= 3 else None

        feats = compute_snapshot_features(
            snapshot_date=day, entry_date=entry_date, front_expiry=front_expiry, back_expiry=back_expiry,
            strike=strike, entry_cashflow=entry_cashflow, entry_underlying_price=entry_stock_price,
            entry_front_iv=entry_front_iv, entry_back_iv=entry_back_iv,
            stock_price_today=stock_price, front_price_today=front_price, back_price_today=back_price,
            peak_mtm_pnl_so_far=peak, mtm_pnl_1d_ago=mtm_1d_ago, mtm_pnl_3d_ago=mtm_3d_ago,
        )
        if feats["mtm_pnl"] is not None:
            peak = feats["peak_mtm_pnl_so_far"]
            mtm_history.append((day, feats["mtm_pnl"]))

        feats["ticker"] = ticker
        feats["entry_date"] = entry_date
        feats["exit_date"] = exit_date
        feats["strike"] = strike
        rows.append(feats)

    return rows


def add_labels(rows, cost_buffer):
    """Given the full day-by-day rows for ONE position (in date order), add the forward-looking
    close_today label and drop the final (exit) day."""
    valid = [r for r in rows if r["mtm_pnl"] is not None]
    if len(valid) < 2:
        return []  # nothing usable (need at least one non-final day to label)

    labeled = []
    for i in range(len(valid) - 1):  # exclude the final day
        remaining_max = max(r["mtm_pnl"] for r in valid[i:])
        r = dict(valid[i])
        r["close_today"] = int(r["mtm_pnl"] >= remaining_max - cost_buffer)
        labeled.append(r)
    return labeled


def build_dataset(
        polygon_api_key: str,
        search_root: str = None,
        run_reason: str = "train",
        output_dir: str = None,
        cost_buffer: float = 15.0,
        max_positions: int = None,
        rate_limit_sleep: float = 0.0,
):
    """
    Build the full hold/close training dataset from the latest committed training import.

    cost_buffer: your estimated round-trip cost of closing early (bid-ask + slippage on both
        legs, in dollars per contract). A day only counts as "should have closed" if it beats
        every later day by more than this — tune it once you have a real cost estimate.
    max_positions: cap for a quick test run (e.g. 20) before committing to the full dataset.
    rate_limit_sleep: seconds to sleep between API calls, if your Polygon plan needs throttling.

    Returns a dict with the resulting DataFrame and the path it was saved to.
    """
    search_root = search_root if search_root else os.path.dirname(os.path.abspath(__file__))
    output_dir = output_dir if output_dir else os.path.join(search_root, "hold_close_data")
    os.makedirs(output_dir, exist_ok=True)

    summary_path = find_latest_summary_file(search_root, run_reason=run_reason)
    df = pd.read_csv(summary_path)
    df = df.dropna(subset=["Overall Profit", "Call_1_expiration_date", "Call_2_expiration_date"]).copy()
    for col in ("Share_Price_Date", "Exit_Date", "Call_1_expiration_date", "Call_2_expiration_date"):
        df[col] = pd.to_datetime(df[col]).dt.date

    if max_positions:
        df = df.head(max_positions)

    client = RESTClient(polygon_api_key, retries=5)

    all_rows = []
    n = len(df)
    for i, (_, position) in enumerate(df.iterrows()):
        print(f"[{i+1}/{n}] {position['Ticker']} entered {position['Share_Price_Date']}...")
        raw_rows = build_position_snapshots(client, position, cost_buffer, rate_limit_sleep)
        labeled_rows = add_labels(raw_rows, cost_buffer)
        all_rows.extend(labeled_rows)

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(output_dir, f"hold_close_training_data_{datetime.datetime.now().strftime('%Y%m%d__%H%M')}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nBuilt {len(result_df)} labeled snapshot rows from {n} positions.")
    print(f"Saved to: {out_path}")

    return {"data": result_df, "output_path": out_path}


if __name__ == "__main__":
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable is not set.")
        sys.exit(1)
    build_dataset(
        polygon_api_key=api_key,
        search_root=os.getcwd(),
        output_dir=os.path.join(os.getcwd(), "hold_close_data"),
    )

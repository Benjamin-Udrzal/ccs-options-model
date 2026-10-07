"""
Shared logic for the hold/close model: fetching option contracts and prices from Polygon,
computing IV/Greeks, and building the feature set for a position on a given day.

Used by BOTH build_hold_close_dataset.py (historical, day-by-day over completed positions)
and run_daily_hold_close_check.py (live, one snapshot per open position per day) — keeping
this in one shared module guarantees the features the model was trained on and the features
it sees at inference time are computed identically.

Deliberately does NOT import or call func_import.py — it only reads func_import's OUTPUT CSVs
(via build_hold_close_dataset.py) and replicates func_import's exact pricing/IV formulas
here as pure, self-contained functions. This means nothing here can ever break func_import's
own behaviour, at the cost of keeping these formulas in sync by hand if func_import.py's
conventions ever change.

Pricing convention (matches func_import.py exactly):
  - Option prices are the day's CLOSING aggregate bar (not real bid/ask).
  - Call_1 = front/short leg (sold to open), Call_2 = back/long leg (bought to open).
  - Entry cashflow  = 100 * (front_price_at_entry - back_price_at_entry)
  - Exit  cashflow  = 100 * (back_price_on_day  - front_price_on_day)   <- if closed on that day
  - mark-to-market P&L if closed on a given day = entry_cashflow + exit_cashflow_on_that_day
"""

import numpy as np
import pandas as pd
from py_vollib.black_scholes import implied_volatility
from py_vollib.black_scholes.greeks import analytical

RISK_FREE_RATE = 0.0432  # matches func_import.py's calculate_iv_and_greeks default


def calculate_iv_and_greeks(S, K, T, option_price, option_type="c", r=RISK_FREE_RATE):
    """Single-contract version of func_import.py's calculate_iv_and_greeks. Returns a dict;
    every value is None (with an 'error' key) if py_vollib can't solve for IV (e.g. a price
    outside the no-arbitrage bounds, which happens for illiquid/stale option quotes)."""
    try:
        iv = implied_volatility.implied_volatility(option_price, S, K, T, r, option_type)
        return {
            "implied_volatility": iv,
            "delta": analytical.delta(option_type, S, K, T, r, iv),
            "gamma": analytical.gamma(option_type, S, K, T, r, iv),
            "theta": analytical.theta(option_type, S, K, T, r, iv),
            "vega": analytical.vega(option_type, S, K, T, r, iv),
        }
    except Exception as e:
        return {"implied_volatility": None, "delta": None, "gamma": None, "theta": None,
                "vega": None, "error": str(e)}


def find_option_contract(client, ticker, strike, expiration_date, contract_type="call"):
    """Look up the Polygon options-ticker string for one specific contract — same method
    func_import.py uses (client.list_options_contracts), so the returned ticker is guaranteed
    to match what a live run would have traded. Returns None if not found."""
    results = list(client.list_options_contracts(
        underlying_ticker=ticker,
        contract_type=contract_type,
        strike_price=strike,
        expiration_date=expiration_date.isoformat() if hasattr(expiration_date, "isoformat") else expiration_date,
        limit=5,
    ))
    return results[0].ticker if results else None


def get_daily_close(client, options_ticker, on_date):
    """Fetch the daily closing aggregate bar for one option contract on one date.
    Returns None if there's no bar that day (contract not traded, market holiday, etc.)."""
    date_str = on_date.isoformat() if hasattr(on_date, "isoformat") else str(on_date)
    bars = client.get_aggs(ticker=options_ticker, multiplier=1, timespan="day",
                            from_=date_str, to=date_str)
    return bars[0].close if bars else None


def get_stock_close(client, ticker, on_date):
    """Fetch the underlying's daily closing price on one date. Returns None if no bar."""
    date_str = on_date.isoformat() if hasattr(on_date, "isoformat") else str(on_date)
    bars = client.get_aggs(ticker=ticker, multiplier=1, timespan="day", from_=date_str, to=date_str)
    return bars[0].close if bars else None


def compute_snapshot_features(
        *,
        snapshot_date,
        entry_date,
        front_expiry,
        back_expiry,
        strike,
        entry_cashflow,
        entry_underlying_price,
        entry_front_iv,
        entry_back_iv,
        stock_price_today,
        front_price_today,
        back_price_today,
        peak_mtm_pnl_so_far,
        mtm_pnl_1d_ago=None,
        mtm_pnl_3d_ago=None,
):
    """
    Build one row of features for a single position on a single day, given prices already
    fetched for that day. Kept as a pure function (no API calls inside) so it's trivially
    testable and identical whether called from the historical builder or the daily checker.

    Returns a dict with the features plus the computed mtm_pnl for this day (the caller is
    responsible for tracking peak_mtm_pnl_so_far and the 1d/3d-ago values across days).
    """
    days_held = (snapshot_date - entry_date).days
    days_to_front_expiry = (front_expiry - snapshot_date).days
    days_to_back_expiry = (back_expiry - snapshot_date).days

    front_T = max(days_to_front_expiry, 0) / 365
    back_T = max(days_to_back_expiry, 0) / 365

    front_greeks = (calculate_iv_and_greeks(stock_price_today, strike, front_T, front_price_today, "c")
                    if front_T > 0 and front_price_today is not None else
                    {"implied_volatility": None, "delta": None, "gamma": None, "theta": None, "vega": None})
    back_greeks = (calculate_iv_and_greeks(stock_price_today, strike, back_T, back_price_today, "c")
                   if back_T > 0 and back_price_today is not None else
                   {"implied_volatility": None, "delta": None, "gamma": None, "theta": None, "vega": None})

    # mark-to-market: what you'd net TODAY if you closed the position right now
    # (matches func_import's Day_2_Cashflow formula, generalized to any day, not just exit)
    if front_price_today is not None and back_price_today is not None:
        exit_cashflow_today = 100 * (back_price_today - front_price_today)
        mtm_pnl_today = round(entry_cashflow + exit_cashflow_today, 2)
    else:
        mtm_pnl_today = None

    underlying_move_pct = (
        100 * (stock_price_today / entry_underlying_price - 1)
        if stock_price_today is not None and entry_underlying_price else None
    )

    front_iv_change = (
        front_greeks["implied_volatility"] - entry_front_iv
        if front_greeks["implied_volatility"] is not None and entry_front_iv is not None else None
    )
    back_iv_change = (
        back_greeks["implied_volatility"] - entry_back_iv
        if back_greeks["implied_volatility"] is not None and entry_back_iv is not None else None
    )

    new_peak = max(peak_mtm_pnl_so_far, mtm_pnl_today) if mtm_pnl_today is not None else peak_mtm_pnl_so_far
    drawdown_from_peak = (mtm_pnl_today - new_peak) if mtm_pnl_today is not None else None

    # Default to 0 ("no known change") rather than None when there isn't enough history yet
    # (a position's first 1-3 days). Using None here would otherwise mean the very start of
    # every position's life has an incomplete feature row — in training that means those rows
    # get silently dropped from EVERY position, biasing the model away from ever learning what
    # early days look like; at inference it would mean a brand-new position can't get a
    # recommendation at all until it's 3 days old.
    mtm_pnl_chg_1d = (mtm_pnl_today - mtm_pnl_1d_ago) if (mtm_pnl_today is not None and mtm_pnl_1d_ago is not None) else 0.0
    mtm_pnl_chg_3d = (mtm_pnl_today - mtm_pnl_3d_ago) if (mtm_pnl_today is not None and mtm_pnl_3d_ago is not None) else 0.0

    return {
        "snapshot_date": snapshot_date,
        "days_held": days_held,
        "days_to_front_expiry": days_to_front_expiry,
        "days_to_back_expiry": days_to_back_expiry,
        "mtm_pnl": mtm_pnl_today,
        "mtm_pnl_pct_of_debit": (100 * mtm_pnl_today / abs(entry_cashflow)) if (mtm_pnl_today is not None and entry_cashflow) else None,
        "peak_mtm_pnl_so_far": new_peak,
        "drawdown_from_peak": drawdown_from_peak,
        "mtm_pnl_chg_1d": mtm_pnl_chg_1d,
        "mtm_pnl_chg_3d": mtm_pnl_chg_3d,
        "underlying_move_pct": underlying_move_pct,
        "front_iv": front_greeks["implied_volatility"],
        "back_iv": back_greeks["implied_volatility"],
        "front_iv_change": front_iv_change,
        "back_iv_change": back_iv_change,
        "front_theta": front_greeks["theta"],
        "back_theta": back_greeks["theta"],
    }


# The feature columns actually fed to the model (everything else in compute_snapshot_features'
# output is identity/bookkeeping, not a model input). Kept as one list so the dataset builder,
# trainer, and daily checker can never disagree about what the model expects.
MODEL_FEATURES = [
    "days_held", "days_to_front_expiry", "days_to_back_expiry",
    "mtm_pnl_pct_of_debit", "drawdown_from_peak", "mtm_pnl_chg_1d", "mtm_pnl_chg_3d",
    "underlying_move_pct", "front_iv", "back_iv", "front_iv_change", "back_iv_change",
    "front_theta", "back_theta",
]

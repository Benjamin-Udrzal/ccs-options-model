"""
Entry point for the "Daily Hold/Close Check" GitHub Actions workflow.

For every OPEN position in your Google Sheet tracker, fetches today's prices, computes the
same features the model was trained on, and sends a Telegram message recommending "keep
holding" or "consider closing" for each — purely advisory, this never touches the sheet
itself. You'd act on a recommendation the same way as any other close, via the existing
"closed <strike> <ticker> <profit>" Telegram command.

WHY THIS NEEDS ITS OWN STATE FILE: the model needs each position's peak P&L so far and its
P&L 1/3 days ago, but the tracker sheet doesn't store daily history (only entry and exit).
Rather than changing the sheet schema or the existing alert/import scripts, this keeps its
own small history file (hold_close_data/position_pnl_history.json), appended to each day this
runs and committed back to the repo like the other state files in this repo. It also caches
each position's ENTRY-day prices/IV the first time it's seen, so they're only fetched once
per position rather than re-fetched every single day.

This script does not modify func_import.py, CCS_Model_Train_v7.py, CCS_Model_Test_v7.py,
run_daily_test_and_alert.py, or check_telegram_replies.py, and does not change what they do.

Requires these GitHub Actions secrets (all already used elsewhere in this repo):
  POLYGON_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
  GOOGLE_SERVICE_ACCOUNT_JSON, GOOGLE_SHEET_ID
"""

import os
import sys
import glob
import json
import datetime
import requests
import gspread
import joblib
import pandas as pd
from polygon import RESTClient

from hold_close_features import compute_snapshot_features, find_option_contract, get_daily_close, get_stock_close, calculate_iv_and_greeks, MODEL_FEATURES

STATE_PATH = os.path.join("hold_close_data", "position_pnl_history.json")
SHEET_TAB_NAME = "Open Positions"


def find_latest_model(search_root):
    model_matches = sorted(glob.glob(os.path.join(search_root, "**", "hold_close_model__*.joblib"), recursive=True))
    if not model_matches:
        raise FileNotFoundError(f"No hold_close_model__*.joblib found under {search_root}. "
                                 f"Run build_hold_close_dataset.py and train_hold_close_model.py first.")
    model_path = model_matches[-1]
    metadata_path = model_path.replace("hold_close_model__", "hold_close_metadata__").replace(".joblib", ".json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    return joblib.load(model_path), metadata


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def get_open_positions(sheet):
    try:
        ws = sheet.worksheet(SHEET_TAB_NAME)
    except gspread.exceptions.WorksheetNotFound:
        return []
    records = ws.get_all_records()
    return [r for r in records if str(r.get("Status", "")).strip().lower() == "open"]


def get_or_init_entry_data(client, state, position_id, record):
    """Fetch and cache entry-day stock price + IV for a position the first time it's seen.
    On later days, reuses the cached values instead of re-fetching (they never change)."""
    if position_id in state and "entry_underlying_price" in state[position_id]:
        e = state[position_id]
        return e["entry_underlying_price"], e["entry_front_iv"], e["entry_back_iv"], e["front_ticker"], e["back_ticker"]

    ticker = record["Ticker"]
    strike = float(record["Strike"])
    entry_date = datetime.date.fromisoformat(str(record["Date Opened"]))
    front_expiry = datetime.date.fromisoformat(str(record["Short Call Expiry"]))
    back_expiry = datetime.date.fromisoformat(str(record["Long Call Expiry"]))

    front_ticker = find_option_contract(client, ticker, strike, front_expiry, "call")
    back_ticker = find_option_contract(client, ticker, strike, back_expiry, "call")
    entry_stock = get_stock_close(client, ticker, entry_date)
    entry_front_price = get_daily_close(client, front_ticker, entry_date) if front_ticker else None
    entry_back_price = get_daily_close(client, back_ticker, entry_date) if back_ticker else None

    entry_front_iv = entry_back_iv = None
    if entry_stock and entry_front_price:
        front_T = max((front_expiry - entry_date).days, 0) / 365
        entry_front_iv = calculate_iv_and_greeks(entry_stock, strike, front_T, entry_front_price, "c")["implied_volatility"]
    if entry_stock and entry_back_price:
        back_T = max((back_expiry - entry_date).days, 0) / 365
        entry_back_iv = calculate_iv_and_greeks(entry_stock, strike, back_T, entry_back_price, "c")["implied_volatility"]

    state.setdefault(position_id, {})
    state[position_id].update({
        "entry_underlying_price": entry_stock, "entry_front_iv": entry_front_iv, "entry_back_iv": entry_back_iv,
        "front_ticker": front_ticker, "back_ticker": back_ticker,
    })
    return entry_stock, entry_front_iv, entry_back_iv, front_ticker, back_ticker


def fmt_money(value):
    """Signed dollar amount, e.g. +$123.45 / -$12.30 (plain f"${-12.3:.2f}" gives "$-12.30")."""
    sign = "-" if value < 0 else "+"
    return f"{sign}${abs(value):,.2f}"


def format_recommendations_message(recommendations):
    """The daily digest: every open position, one line each, grouped into close / keep.
    Positions flagged for closing ALSO get their own detailed alert (format_close_alert_message)."""
    if not recommendations:
        return "📋 <b>Hold/Close Check</b>\n\nNo open positions to check today."

    close_recs = [r for r in recommendations if r["recommend_close"]]
    hold_recs = [r for r in recommendations if not r["recommend_close"]]

    lines = [f"📋 <b>Hold/Close Check — {len(recommendations)} open position(s)</b>\n"]

    if close_recs:
        lines.append(f"⚠️ <b>Consider closing ({len(close_recs)}) — details in the alert below:</b>")
        for r in close_recs:
            lines.append(
                f"• {r['ticker']} strike ${r['strike']:.2f} (expiry {r['short_call_expiry']}) — "
                f"current P&L {fmt_money(r['mtm_pnl'])}, close confidence {r['close_probability']:.0%}"
            )
        lines.append("")

    if hold_recs:
        lines.append(f"✅ <b>Keep holding ({len(hold_recs)}):</b>")
        for r in hold_recs:
            lines.append(
                f"• {r['ticker']} strike ${r['strike']:.2f} (expiry {r['short_call_expiry']}) — "
                f"current P&L {fmt_money(r['mtm_pnl'])}, close confidence {r['close_probability']:.0%}"
            )

    return "\n".join(lines)


def format_close_alert_message(close_recs, today):
    """A dedicated alert, sent ONLY on days at least one position is flagged to close, with
    what you need to act on it: the confidence, what closing right now would pay out, the
    resulting profit, and a ready-to-send "closed ..." reply for the tracker.

    Figures use today's closing-bar prices (same basis the model was trained on and the same
    convention as func_import.py), not live bid/ask, so real fills will differ a little."""
    if not close_recs:
        return None

    # Most confident first
    close_recs = sorted(close_recs, key=lambda r: r["close_probability"], reverse=True)

    blocks = [f"🚨 <b>CLOSE ALERT — {today.isoformat()}</b>\n"
              f"{len(close_recs)} position(s) the model says to close today:"]

    for i, r in enumerate(close_recs, start=1):
        reason = "Lock in profit" if r["mtm_pnl"] > 0 else "Limit the loss"
        profit_cmd = f"{r['mtm_pnl']:.2f}"
        # Only mention the peak if the position has actually been in profit; "peak: -$45" is noise.
        # (It's the best P&L seen on days this check ran, so it can understate the true intraday peak.)
        peak_text = f"  |  Peak P&L seen: {fmt_money(r['peak_mtm_pnl'])}" if r["peak_mtm_pnl"] > 0 else ""
        blocks.append(
            f"<b>{i}. {r['ticker']}</b> — strike ${r['strike']:.2f}  ({reason})\n"
            f"   Close confidence: <b>{r['close_probability']:.0%}</b>\n"
            f"   Cashflow if closed now: <b>{fmt_money(r['exit_cashflow_today'])}</b>"
            f" (sell long call, buy back short call)\n"
            f"   Expected profit: <b>{fmt_money(r['mtm_pnl'])}</b>"
            f" (entry cashflow {fmt_money(r['entry_cashflow'])})\n"
            f"   Short call expiry: {r['short_call_expiry']}  |  Long call expiry: {r['long_call_expiry']}\n"
            f"   Opened {r['date_opened']} ({r['days_held']} days ago){peak_text}\n"
            f"   Once closed, reply: <code>closed {r['strike']:g} {r['ticker']} {profit_cmd}</code>"
            f" (swap in your actual profit)"
        )

    blocks.append("<i>Based on today's closing prices; actual fills will differ.</i>")
    return "\n\n".join(blocks)


def send_telegram_message(bot_token, chat_id, text):
    """Send `text`, splitting on blank lines into chunks under Telegram's 4096-char cap
    (splitting on message-block boundaries rather than mid-text, so HTML tags never get cut)."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    max_len = 3500
    chunks, current = [], ""
    for block in text.split("\n\n"):
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > max_len and current:
            chunks.append(current)
            current = block
        else:
            current = candidate
    if current:
        chunks.append(current)

    for chunk in chunks:
        resp = requests.post(url, data={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"})
        if resp.status_code != 200:
            print(f"WARNING: Telegram send failed ({resp.status_code}): {resp.text}")


def check_open_positions(polygon_api_key, sheet, pipeline, metadata, state, today=None):
    """Core logic, kept separate from main() so it's testable without real API calls. Returns
    (recommendations list, updated state)."""
    today = today or datetime.date.today()
    client = RESTClient(polygon_api_key, retries=5)
    features_needed = metadata["features"]

    open_positions = get_open_positions(sheet)
    recommendations = []

    for record in open_positions:
        ticker = record["Ticker"]
        strike = float(record["Strike"])
        entry_date = datetime.date.fromisoformat(str(record["Date Opened"]))
        front_expiry = datetime.date.fromisoformat(str(record["Short Call Expiry"]))
        back_expiry = datetime.date.fromisoformat(str(record["Long Call Expiry"]))
        entry_cashflow = float(record["Entry Cashflow"])
        position_id = f"{ticker}|{entry_date.isoformat()}|{strike}"

        entry_stock, entry_front_iv, entry_back_iv, front_ticker, back_ticker = get_or_init_entry_data(
            client, state, position_id, record)
        if front_ticker is None or back_ticker is None or entry_stock is None:
            print(f"  WARNING: could not resolve contracts/entry data for {ticker} {strike} — skipping.")
            continue

        stock_today = get_stock_close(client, ticker, today)
        front_today = get_daily_close(client, front_ticker, today)
        back_today = get_daily_close(client, back_ticker, today)
        if stock_today is None or front_today is None or back_today is None:
            print(f"  WARNING: missing today's price data for {ticker} {strike} — skipping.")
            continue

        history = state[position_id].get("history", [])
        peak_so_far = max([entry_cashflow] + [h["mtm_pnl"] for h in history]) if history else entry_cashflow
        mtm_1d_ago = history[-1]["mtm_pnl"] if len(history) >= 1 else None
        mtm_3d_ago = history[-3]["mtm_pnl"] if len(history) >= 3 else None

        feats = compute_snapshot_features(
            snapshot_date=today, entry_date=entry_date, front_expiry=front_expiry, back_expiry=back_expiry,
            strike=strike, entry_cashflow=entry_cashflow, entry_underlying_price=entry_stock,
            entry_front_iv=entry_front_iv, entry_back_iv=entry_back_iv,
            stock_price_today=stock_today, front_price_today=front_today, back_price_today=back_today,
            peak_mtm_pnl_so_far=peak_so_far, mtm_pnl_1d_ago=mtm_1d_ago, mtm_pnl_3d_ago=mtm_3d_ago,
        )

        if feats["mtm_pnl"] is None:
            continue

        state[position_id].setdefault("history", []).append({"date": today.isoformat(), "mtm_pnl": feats["mtm_pnl"]})

        row_df = pd.DataFrame([{f: feats.get(f) for f in features_needed}])
        if row_df.isna().any(axis=None):
            print(f"  WARNING: missing feature value(s) for {ticker} {strike} — skipping model prediction.")
            continue

        close_probability = pipeline.predict_proba(row_df[features_needed])[0, 1]
        recommendations.append({
            "ticker": ticker, "strike": strike, "short_call_expiry": record["Short Call Expiry"],
            "mtm_pnl": feats["mtm_pnl"], "close_probability": float(close_probability),
            "recommend_close": bool(close_probability > 0.5),
            # Extra detail for the close alert (none of this feeds the model):
            "long_call_expiry": record["Long Call Expiry"],
            "date_opened": entry_date.isoformat(),
            "days_held": feats["days_held"],
            "entry_cashflow": entry_cashflow,
            # What closing right now would pay out — same formula as func_import's Day_2_Cashflow
            # and compute_snapshot_features' mtm calculation: sell the long (back) call, buy back
            # the short (front) call. Expected profit (mtm_pnl) = entry_cashflow + this.
            "exit_cashflow_today": round(100 * (back_today - front_today), 2),
            "peak_mtm_pnl": feats["peak_mtm_pnl_so_far"],
        })

    return recommendations, state


def main():
    polygon_api_key = os.environ.get("POLYGON_API_KEY")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")

    missing = [name for name, val in [
        ("POLYGON_API_KEY", polygon_api_key), ("TELEGRAM_BOT_TOKEN", bot_token),
        ("TELEGRAM_CHAT_ID", chat_id), ("GOOGLE_SERVICE_ACCOUNT_JSON", service_account_json),
        ("GOOGLE_SHEET_ID", sheet_id),
    ] if not val]
    if missing:
        print(f"ERROR: missing environment variable(s): {', '.join(missing)}")
        sys.exit(1)

    search_root = os.getcwd()
    pipeline, metadata = find_latest_model(search_root)
    print(f"Using model: {metadata['timestamp']} (features: {metadata['features']})")

    gc = gspread.service_account_from_dict(json.loads(service_account_json))
    sheet = gc.open_by_key(sheet_id)

    state = load_state()
    recommendations, state = check_open_positions(polygon_api_key, sheet, pipeline, metadata, state)
    save_state(state)

    close_recs = [r for r in recommendations if r["recommend_close"]]

    # Wrapped so a Telegram/formatting problem can't make the script exit non-zero — the
    # position history was already saved above, and a failed run could skip the workflow's
    # commit step and lose a day of history.
    try:
        # 1. The daily digest (every open position, one line each)
        send_telegram_message(bot_token, chat_id, format_recommendations_message(recommendations))

        # 2. A separate, detailed alert — only on days something should be closed. Sent last so
        #    it's the message you see at the bottom of the chat.
        close_alert = format_close_alert_message(close_recs, datetime.date.today())
        if close_alert:
            send_telegram_message(bot_token, chat_id, close_alert)
    except Exception as e:
        print(f"WARNING: Telegram message failed, but position history was saved. Error: {e}")

    print(f"Checked {len(recommendations)} open position(s); "
          f"{len(close_recs)} recommended for closing.")


if __name__ == "__main__":
    main()

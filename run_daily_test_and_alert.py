"""
Entry point for the "Daily Test + Alert" GitHub Actions workflow.

Runs on weekdays at 17:15 Europe/London:
  1. Runs the daily "dtest" import (func_import) — same as run_daily_import.py.
  2. Tests that fresh dataset against the LATEST committed trained model (test_model()).
  3. Sends a Telegram message listing every suggested trade (Open_Trade == 1), with the
     details needed to open it: ticker, strike, short/long call expiry, expected Day 1 cashflow.
  4. Saves a numbered reference of today's positions to alerts/latest_signal.json, so that
     check_telegram_replies.py can resolve "opened 2" back to the actual position details.

Requires these GitHub Actions secrets:
  POLYGON_API_KEY      - as used by the daily import
  TELEGRAM_BOT_TOKEN   - your bot's token from @BotFather
  TELEGRAM_CHAT_ID     - the chat ID to send alerts to
"""

import os
import sys
import json
import datetime
from zoneinfo import ZoneInfo
import requests

from func_import import func_import
from CCS_Model_Test_v7 import test_model

SIGNAL_REFERENCE_PATH = os.path.join("alerts", "latest_signal.json")
ALERT_LOG_PATH = os.path.join("alerts", "alert_log.json")


def today_uk_date():
    """The current calendar date in Europe/London, used as the per-day key for both the
    signal reference and the alert dedup log.

    NOTE: an earlier version of this script derived "today's date" from
    os.path.dirname(test_result["test_data_path"]) — but that path's immediate parent folder
    is func_import's constant "<run_reason> Import Outputs" directory name, not a date at all.
    That meant every day's Telegram header read "CCS Daily Signal — dtest Import Outputs"
    instead of an actual date, and — more seriously — once per-day alert dedup was added, it
    would have kept matching that same constant string forever, silently blocking EVERY future
    alert after the first one ever sent. Using the real Europe/London calendar date here avoids
    both problems and doesn't depend on func_import's folder-naming scheme at all."""
    return datetime.datetime.now(ZoneInfo("Europe/London")).date().isoformat()


def build_positions_list(df_open):
    """Turn the Open_Trade==1 rows into a plain list of dicts, numbered the same way the
    Telegram message numbers them (position 1, 2, 3...) — this numbering is the contract
    between the alert message and check_telegram_replies.py."""
    missing_cols = [c for c in ("Call_1_expiration_date", "Call_2_expiration_date", "Call_strike_price", "Day_1_Cashflow")
                     if c not in df_open.columns]
    if missing_cols:
        print(f"WARNING: predictions are missing expected column(s) {missing_cols} — "
              f"showing 'N/A' for those fields. This usually means the test data was generated "
              f"by an older version of func_import.py that doesn't save these columns yet.")

    positions = []
    for _, row in df_open.iterrows():
        strike = row.get("Call_strike_price", None)
        cashflow = row.get("Day_1_Cashflow", None)
        positions.append({
            "ticker": str(row.get("Ticker", "Unknown")),
            "strike": float(strike) if isinstance(strike, (int, float)) else None,
            "short_call_expiry": str(row.get("Call_1_expiration_date", "N/A")),
            "long_call_expiry": str(row.get("Call_2_expiration_date", "N/A")),
            "day_1_cashflow": float(cashflow) if isinstance(cashflow, (int, float)) else None,
        })
    return positions


def format_trade_message(positions, signal_date):
    """Build an HTML-formatted Telegram message from a positions list (see build_positions_list)."""
    header = f"📊 <b>CCS Daily Signal — {signal_date}</b>"

    if not positions:
        return f"{header}\n\nNo trades suggested today."

    lines = [header, f"\n<b>{len(positions)} position(s) suggested to open:</b>\n"]
    for i, pos in enumerate(positions, start=1):
        strike_str = f"${pos['strike']:.2f}" if pos["strike"] is not None else "N/A"
        cashflow_str = f"${pos['day_1_cashflow']:.2f}" if pos["day_1_cashflow"] is not None else "N/A"
        lines.append(
            f"<b>{i}. {pos['ticker']}</b>\n"
            f"   Strike: {strike_str}\n"
            f"   Short Call Expiry: {pos['short_call_expiry']}\n"
            f"   Long Call Expiry: {pos['long_call_expiry']}\n"
            f"   Expected Day 1 Cashflow: {cashflow_str}\n"
        )
    lines.append(
        "\n💡 Reply <b>opened &lt;number&gt;</b> (e.g. \"opened 2\") to log a position to your tracker."
    )
    return "\n".join(lines)


def save_signal_reference(positions, signal_date):
    """Save today's numbered positions so check_telegram_replies.py can look up what
    'opened N' refers to. Overwrites the previous day's file — replies should be sent the
    same day the alert arrives, before the next day's run overwrites this. Includes a Unix
    timestamp so the reply-checker can ignore any message sent BEFORE this signal existed
    (e.g. a stale "opened 2" left over from before today's alert went out).

    If a signal for THIS SAME signal_date already exists (e.g. a manual run happens on a day
    a scheduled alert already went out), it is left untouched rather than overwritten.
    Rewriting it would bump "sent_at" forward, which could make the reply-checker wrongly
    discard an "opened N" reply that the user already sent in response to the earlier alert,
    just because it now looks "older than the signal"."""
    if os.path.exists(SIGNAL_REFERENCE_PATH):
        try:
            with open(SIGNAL_REFERENCE_PATH) as f:
                existing = json.load(f)
            if existing.get("date") == signal_date:
                print(f"Signal reference for {signal_date} already exists — leaving it as-is "
                      f"(not resetting sent_at, so any reply already sent today still resolves correctly).")
                return
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/unreadable — fall through and just (re)write a fresh one

    os.makedirs(os.path.dirname(SIGNAL_REFERENCE_PATH), exist_ok=True)
    with open(SIGNAL_REFERENCE_PATH, "w") as f:
        json.dump({
            "date": signal_date,
            "sent_at": datetime.datetime.now(datetime.timezone.utc).timestamp(),
            "positions": positions,
        }, f, indent=2)
    print(f"Signal reference saved to: {SIGNAL_REFERENCE_PATH}")


def load_alert_log():
    if os.path.exists(ALERT_LOG_PATH):
        try:
            with open(ALERT_LOG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"sent_dates": []}


def save_alert_log(log):
    os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
    # Keep the log small — only recent dates are ever relevant for same-day-repeat detection.
    log["sent_dates"] = log.get("sent_dates", [])[-30:]
    with open(ALERT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def send_telegram_message(text, bot_token, chat_id):
    """Send `text` via the Telegram Bot API, splitting into chunks if over Telegram's 4096-char cap."""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    max_len = 3500
    chunks = [text[i:i + max_len] for i in range(0, len(text), max_len)] or [text]
    for chunk in chunks:
        resp = requests.post(url, data={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML"})
        if resp.status_code != 200:
            print(f"WARNING: Telegram send failed ({resp.status_code}): {resp.text}")


def main():
    api_key = os.environ.get("POLYGON_API_KEY")
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")

    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable is not set.")
        sys.exit(1)
    if not bot_token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID environment variable is not set.")
        sys.exit(1)

    # --- 0. Decide up front whether this run should do anything at all ---
    #
    # GitHub Actions' `schedule` trigger is best-effort and can be delayed by hours — enough
    # that this workflow's two DST-safe cron slots (see the workflow YAML) can both end up
    # firing on the same day, in either order, at unpredictable times. Rather than trying to
    # guess which slot is "correct" by checking the clock (the old approach: skip unless the
    # current London hour is 17 — which wrongly skips a legitimately first, just-late run), the
    # rule here is simpler and matches what's actually wanted: whichever scheduled run fires
    # FIRST each day does the real work and sends the alert; any other scheduled run that fires
    # later the same day is a repeat, and skips immediately, before spending any time or
    # Polygon API quota on the import/test.
    #
    # A manual run (workflow_dispatch — clicking "Run workflow", or re-running a job) is always
    # a deliberate request, so it always runs fully and always sends, regardless of whether a
    # scheduled alert already went out today. It also never writes to alert_log.json, so it can
    # neither block nor be blocked by the schedule's own once-a-day tracking — the two are
    # tracked completely independently.
    is_scheduled_run = os.environ.get("GITHUB_EVENT_NAME") == "schedule"
    signal_date = today_uk_date()

    if is_scheduled_run:
        alert_log = load_alert_log()
        if signal_date in alert_log.get("sent_dates", []):
            print(f"Alert for {signal_date} was already sent by an earlier scheduled run today — "
                  f"this run is a repeat/delayed trigger (see DST cron notes in the workflow). "
                  f"Skipping the import/test/alert pipeline entirely — nothing left to do.")
            return

    # --- 1. Run the daily import ---
    output_dir = os.path.join(os.getcwd(), "dtest_outputs")
    import_result = func_import(
        polygon_api_key=api_key,
        run_reason="dtest",
        default_settings=True,
        output_dir=output_dir,
    )
    print(f"Import saved to: {import_result['output_dir']}")

    # --- 2. Test the fresh data against the latest trained model ---
    # test_data is left as None deliberately: since this dtest folder was JUST created above,
    # it's guaranteed to be the latest one — no need to point at it explicitly.
    test_result = test_model(search_root=os.getcwd())
    print(f"Model used: {test_result['model_folder']}")
    print(f"Test data used: {test_result['test_data_path']}")
    print(f"Predictions saved to: {test_result['output_path']}")

    # --- 3. Send the Telegram alert, and save today's numbered signal for later replies ---
    # Wrapped in try/except deliberately: the import and predictions above are the valuable,
    # expensive part of this run. A bug in message formatting or a Telegram API hiccup should
    # never cause the whole script to exit non-zero — that would skip the "commit and push"
    # step in the workflow and throw away a perfectly good day's data. Worst case here, you
    # miss one alert but still get the commit; check the Action log if that happens.
    #
    # Note: whether to send at all was already decided in step 0, before the expensive import
    # and test ran — by this point we always want to send (it's either a manual run, or the
    # first scheduled run of the day).
    try:
        df = test_result["predictions"]
        df_open = df[df["Open_Trade"] == 1].copy()

        positions = build_positions_list(df_open)
        save_signal_reference(positions, signal_date)

        message = format_trade_message(positions, signal_date)
        send_telegram_message(message, bot_token, chat_id)

        if is_scheduled_run:
            alert_log = load_alert_log()
            alert_log.setdefault("sent_dates", []).append(signal_date)
            save_alert_log(alert_log)

        print(f"Telegram alert sent ({len(positions)} trade(s) to open) "
              f"[trigger: {os.environ.get('GITHUB_EVENT_NAME', 'unknown')}].")
    except Exception as e:
        print(f"WARNING: Telegram alert failed, but import/test data was saved successfully. "
              f"Error: {e}")


if __name__ == "__main__":
    main()

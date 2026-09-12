"""
Entry point for the "Daily Test + Alert" GitHub Actions workflow.

Runs on weekdays at 17:15 Europe/London:
  1. Runs the daily "dtest" import (func_import) — same as run_daily_import.py.
  2. Tests that fresh dataset against the LATEST committed trained model (test_model()).
  3. Sends a Telegram message listing every suggested trade (Open_Trade == 1), with the
     details needed to open it: ticker, strike, short/long call expiry, expected Day 1 cashflow.

Requires these GitHub Actions secrets:
  POLYGON_API_KEY      - as used by the daily import
  TELEGRAM_BOT_TOKEN   - your bot's token from @BotFather
  TELEGRAM_CHAT_ID     - the chat ID to send alerts to
"""

import os
import sys
import requests

from func_import import func_import
from CCS_Model_Test_v7 import test_model


def format_trade_message(df_open, test_date):
    """Build an HTML-formatted Telegram message from the rows marked Open_Trade == 1."""
    header = f"📊 <b>CCS Daily Signal — {test_date}</b>"

    if df_open.empty:
        return f"{header}\n\nNo trades suggested today."

    lines = [header, f"\n<b>{len(df_open)} position(s) suggested to open:</b>\n"]
    missing_cols = [c for c in ("Call_1_expiration_date", "Call_2_expiration_date", "Call_strike_price", "Day_1_Cashflow")
                     if c not in df_open.columns]
    if missing_cols:
        print(f"WARNING: predictions are missing expected column(s) {missing_cols} — "
              f"showing 'N/A' for those fields. This usually means the test data was generated "
              f"by an older version of func_import.py that doesn't save these columns yet.")

    for i, (_, row) in enumerate(df_open.iterrows(), start=1):
        strike = row.get("Call_strike_price", "N/A")
        short_expiry = row.get("Call_1_expiration_date", "N/A")
        long_expiry = row.get("Call_2_expiration_date", "N/A")
        cashflow = row.get("Day_1_Cashflow", "N/A")
        strike_str = f"${strike:.2f}" if isinstance(strike, (int, float)) else str(strike)
        cashflow_str = f"${cashflow:.2f}" if isinstance(cashflow, (int, float)) else str(cashflow)
        lines.append(
            f"<b>{i}. {row.get('Ticker', 'Unknown')}</b>\n"
            f"   Strike: {strike_str}\n"
            f"   Short Call Expiry: {short_expiry}\n"
            f"   Long Call Expiry: {long_expiry}\n"
            f"   Expected Day 1 Cashflow: {cashflow_str}\n"
        )
    return "\n".join(lines)


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

    # --- 3. Send the Telegram alert ---
    # Wrapped in try/except deliberately: the import and predictions above are the valuable,
    # expensive part of this run. A bug in message formatting or a Telegram API hiccup should
    # never cause the whole script to exit non-zero — that would skip the "commit and push"
    # step in the workflow and throw away a perfectly good day's data. Worst case here, you
    # miss one alert but still get the commit; check the Action log if that happens.
    try:
        df = test_result["predictions"]
        df_open = df[df["Open_Trade"] == 1].copy()
        test_date = os.path.basename(os.path.dirname(test_result["test_data_path"]))

        message = format_trade_message(df_open, test_date)
        send_telegram_message(message, bot_token, chat_id)
        print(f"Telegram alert sent ({len(df_open)} trade(s) to open).")
    except Exception as e:
        print(f"WARNING: Telegram alert failed, but import/test data was saved successfully. "
              f"Error: {e}")


if __name__ == "__main__":
    main()

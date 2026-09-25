"""
Entry point for the "Check Telegram Replies" GitHub Actions workflow.

Polls Telegram for new messages (GitHub Actions can't run a persistent webhook server, so
short-interval polling is the practical alternative). Looks for replies of the form:

    opened <number>

e.g. "opened 2", matching the position numbers from the day's Telegram alert (see
run_daily_test_and_alert.py). When matched, looks up that position in
alerts/latest_signal.json and appends it as a new row to the "Open Positions" tab of your
Google Sheet, then replies in Telegram to confirm.

State (which Telegram updates have been processed, and which position numbers have already
been logged today) is kept in alerts/telegram_state.json and committed back to the repo by
the workflow, the same way the other scripts persist state.

Requires these GitHub Actions secrets:
  TELEGRAM_BOT_TOKEN            - your bot's token from @BotFather
  TELEGRAM_CHAT_ID              - your chat ID (only messages from this chat are processed)
  GOOGLE_SERVICE_ACCOUNT_JSON   - full contents of a Google Cloud service account JSON key
  GOOGLE_SHEET_ID               - the target spreadsheet's ID (from its URL)
"""

import os
import re
import sys
import json
import datetime
import requests
import gspread

SIGNAL_REFERENCE_PATH = os.path.join("alerts", "latest_signal.json")
STATE_PATH = os.path.join("alerts", "telegram_state.json")
SHEET_TAB_NAME = "Open Positions"
SHEET_HEADERS = [
    "Date Opened", "Ticker", "Strike", "Short Call Expiry", "Long Call Expiry",
    "Entry Cashflow", "Status", "Exit Date", "Exit Cashflow",
]

OPENED_PATTERN = re.compile(r"^\s*opened\s+(\d+)\s*$", re.IGNORECASE)


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_update_id": 0, "logged_positions": {}}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_signal_reference():
    if not os.path.exists(SIGNAL_REFERENCE_PATH):
        return None
    with open(SIGNAL_REFERENCE_PATH) as f:
        return json.load(f)


def get_telegram_updates(bot_token, offset):
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    resp = requests.get(url, params={"offset": offset, "timeout": 10})
    resp.raise_for_status()
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram getUpdates failed: {data}")
    return data["result"]


def send_telegram_message(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    resp = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
    if resp.status_code != 200:
        print(f"WARNING: Telegram send failed ({resp.status_code}): {resp.text}")


def get_or_create_worksheet(sheet):
    try:
        ws = sheet.worksheet(SHEET_TAB_NAME)
    except gspread.exceptions.WorksheetNotFound:
        print(f"'{SHEET_TAB_NAME}' tab not found — creating it with headers.")
        ws = sheet.add_worksheet(title=SHEET_TAB_NAME, rows=1000, cols=len(SHEET_HEADERS))
        ws.append_row(SHEET_HEADERS)
    return ws


def append_position_to_sheet(worksheet, position):
    row = [
        datetime.date.today().isoformat(),
        position["ticker"],
        position["strike"] if position["strike"] is not None else "",
        position["short_call_expiry"],
        position["long_call_expiry"],
        position["day_1_cashflow"] if position["day_1_cashflow"] is not None else "",
        "Open",
        "",  # Exit Date - filled in later when the position is closed
        "",  # Exit Cashflow - filled in later
    ]
    worksheet.append_row(row)


def main():
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    service_account_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_id = os.environ.get("GOOGLE_SHEET_ID")

    if not bot_token or not chat_id:
        print("ERROR: TELEGRAM_BOT_TOKEN and/or TELEGRAM_CHAT_ID environment variable is not set.")
        sys.exit(1)
    if not service_account_json or not sheet_id:
        print("ERROR: GOOGLE_SERVICE_ACCOUNT_JSON and/or GOOGLE_SHEET_ID environment variable is not set.")
        sys.exit(1)

    state = load_state()
    updates = get_telegram_updates(bot_token, offset=state["last_update_id"] + 1)

    if not updates:
        print("No new Telegram messages.")
        return

    signal = load_signal_reference()

    # Only act on "opened N" messages; every OTHER update still advances last_update_id below,
    # so we never re-process old chatter, questions, or unrelated messages on future runs.
    # A message is only "relevant" if it also arrived AFTER the current signal was sent —
    # otherwise it's a stale reply to a PREVIOUS day's (now-overwritten) signal, and matching
    # it against today's position numbers would silently log the wrong trade.
    relevant = []
    stale_count = 0
    for update in updates:
        state["last_update_id"] = max(state["last_update_id"], update["update_id"])
        message = update.get("message")
        if not message:
            continue
        if str(message.get("chat", {}).get("id")) != str(chat_id):
            print(f"Ignoring message from unauthorized chat {message.get('chat', {}).get('id')}.")
            continue
        text = message.get("text", "")
        match = OPENED_PATTERN.match(text)
        if not match:
            continue

        message_sent_at = message.get("date")  # Telegram gives this as a Unix timestamp
        if signal is not None and message_sent_at is not None and message_sent_at < signal["sent_at"]:
            stale_count += 1
            print(f"Ignoring \"{text}\" — sent before today's signal, refers to a previous day's positions.")
            continue

        relevant.append(int(match.group(1)))

    if not relevant:
        print(f"No actionable 'opened <number>' messages among {len(updates)} update(s) "
              f"({stale_count} stale message(s) ignored).")
        save_state(state)
        return

    if signal is None:
        for n in relevant:
            send_telegram_message(bot_token, chat_id,
                                   f"⚠️ Got \"opened {n}\" but no signal has been sent yet today — nothing to log.")
        save_state(state)
        return

    gc = gspread.service_account_from_dict(json.loads(service_account_json))
    sheet = gc.open_by_key(sheet_id)
    worksheet = get_or_create_worksheet(sheet)

    signal_date = signal["date"]
    logged_today = set(state["logged_positions"].get(signal_date, []))

    for n in relevant:
        idx = n - 1  # positions are 1-indexed in the message, 0-indexed in the list
        if idx < 0 or idx >= len(signal["positions"]):
            send_telegram_message(bot_token, chat_id,
                                   f"⚠️ \"opened {n}\" doesn't match any position in today's signal "
                                   f"(there were {len(signal['positions'])}). Nothing logged.")
            continue
        if n in logged_today:
            send_telegram_message(bot_token, chat_id, f"ℹ️ Position {n} was already logged today — skipping.")
            continue

        position = signal["positions"][idx]
        append_position_to_sheet(worksheet, position)
        logged_today.add(n)
        send_telegram_message(
            bot_token, chat_id,
            f"✅ Logged to tracker: <b>{position['ticker']}</b> "
            f"(strike ${position['strike']:.2f}, short {position['short_call_expiry']}, "
            f"long {position['long_call_expiry']})."
        )
        print(f"Logged position {n} ({position['ticker']}) to the sheet.")

    state["logged_positions"][signal_date] = sorted(logged_today)
    save_state(state)


if __name__ == "__main__":
    main()

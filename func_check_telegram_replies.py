"""
Entry point for the "Check Telegram Replies" GitHub Actions workflow.

Polls Telegram for new messages (GitHub Actions can't run a persistent webhook server, so
short-interval polling is the practical alternative). Handles two kinds of replies:

    opened <number>[,<number>...]
        e.g. "opened 2" or "opened 1,2,3" (commas and/or spaces both work) — matches the
        position numbers from the day's Telegram alert (see run_daily_test_and_alert.py).
        Looks each one up in alerts/latest_signal.json and appends it as a new row to the
        "Open Positions" tab of your Google Sheet.

    closed <strike> <ticker> <profit>
        e.g. "closed 56 SLV 23.6" — finds the matching OPEN row(s) in the sheet by ticker and
        strike, and marks it closed: Status -> "Closed", Exit Date -> today, Profit -> the
        given profit, Exit Cashflow -> profit - entry cashflow.
        If more than one open position matches (same ticker and strike, different expiry),
        a Telegram message is sent asking which expiration date is meant, listing the
        available options. Your next reply naming that expiry is picked up on the NEXT run
        of this workflow (state persists across runs in alerts/telegram_state.json) and
        completes the close on the correct row.
        If no open position matches at all, you get a message explaining that.

State (which Telegram updates have been processed, which position numbers have already been
logged today, and any pending close-disambiguation) is kept in alerts/telegram_state.json and
committed back to the repo by the workflow, the same way the other scripts persist state.

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
    "Entry Cashflow", "Status", "Exit Date", "Exit Cashflow", "Profit",
]

# Accepts one or more position numbers after "opened", separated by commas and/or whitespace:
# "opened 2", "opened 1,2,3", "opened 1, 2, 3", and "opened 1 2 3" all match.
OPENED_PATTERN = re.compile(r"^\s*opened\s+(\d+(?:\s*[,\s]\s*\d+)*)\s*$", re.IGNORECASE)
CLOSED_PATTERN = re.compile(r"^\s*closed\s+(-?[\d.]+)\s+([A-Za-z.]+)\s+(-?[\d.]+)\s*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# State / signal persistence
# ---------------------------------------------------------------------------

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
    else:
        state = {}
    state.setdefault("last_update_id", 0)
    state.setdefault("logged_positions", {})
    state.setdefault("pending_close", None)
    return state


def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def load_signal_reference():
    if not os.path.exists(SIGNAL_REFERENCE_PATH):
        return None
    with open(SIGNAL_REFERENCE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Google Sheet helpers
# ---------------------------------------------------------------------------

def get_or_create_worksheet(sheet):
    try:
        ws = sheet.worksheet(SHEET_TAB_NAME)
    except gspread.exceptions.WorksheetNotFound:
        print(f"'{SHEET_TAB_NAME}' tab not found — creating it.")
        ws = sheet.add_worksheet(title=SHEET_TAB_NAME, rows=1000, cols=len(SHEET_HEADERS))
    return ws


def ensure_worksheet_headers(worksheet):
    """Make sure every column in SHEET_HEADERS exists in row 1, appending any that are
    missing (e.g. 'Profit' on a sheet created before that column existed). Returns the
    final header row, in whatever order it actually is in the sheet."""
    header_row = worksheet.row_values(1)
    changed = False
    for col_name in SHEET_HEADERS:
        if col_name not in header_row:
            header_row.append(col_name)
            changed = True
    if changed:
        worksheet.update("A1", [header_row])
        print(f"Updated sheet headers: {header_row}")
    return header_row


def append_position_to_sheet(worksheet, header_row, position):
    row = [""] * len(header_row)
    values = {
        "Date Opened": datetime.date.today().isoformat(),
        "Ticker": position["ticker"],
        "Strike": position["strike"] if position["strike"] is not None else "",
        "Short Call Expiry": position["short_call_expiry"],
        "Long Call Expiry": position["long_call_expiry"],
        "Entry Cashflow": position["day_1_cashflow"] if position["day_1_cashflow"] is not None else "",
        "Status": "Open",
    }
    for col_name, value in values.items():
        row[header_row.index(col_name)] = value
    worksheet.append_row(row)


def find_open_positions(worksheet, header_row, ticker, strike):
    """Return a list of (row_number, record_dict) for OPEN rows matching ticker + strike.
    row_number is the actual 1-indexed sheet row, ready to pass to update_cell/update."""
    records = worksheet.get_all_records(expected_headers=header_row)
    matches = []
    for i, rec in enumerate(records):
        row_number = i + 2  # +1 for header row, +1 because sheet rows are 1-indexed
        if str(rec.get("Status", "")).strip().lower() != "open":
            continue
        if str(rec.get("Ticker", "")).strip().upper() != ticker.strip().upper():
            continue
        try:
            rec_strike = float(rec.get("Strike", ""))
        except (TypeError, ValueError):
            continue
        if abs(rec_strike - strike) > 1e-6:
            continue
        matches.append((row_number, rec))
    return matches


def close_position_row(worksheet, header_row, row_number, record, profit):
    entry_cashflow = record.get("Entry Cashflow", "")
    try:
        entry_cashflow_val = float(entry_cashflow)
    except (TypeError, ValueError):
        entry_cashflow_val = 0.0
    exit_cashflow = round(profit - entry_cashflow_val, 2)

    updates = {
        "Status": "Closed",
        "Exit Date": datetime.date.today().isoformat(),
        "Exit Cashflow": exit_cashflow,
        "Profit": profit,
    }
    for col_name, value in updates.items():
        col_index = header_row.index(col_name) + 1  # gspread columns are 1-indexed
        worksheet.update_cell(row_number, col_index, value)
    return exit_cashflow


def format_disambiguation_message(ticker, strike, candidates):
    lines = [
        f"❓ Found {len(candidates)} open positions matching <b>{ticker}</b> strike ${strike:.2f}.",
        "Reply with the exact Short Call Expiry to specify which one:\n",
    ]
    for row_number, rec in candidates:
        lines.append(
            f"• {rec.get('Short Call Expiry', 'N/A')} "
            f"(opened {rec.get('Date Opened', 'N/A')}, entry cashflow {rec.get('Entry Cashflow', 'N/A')})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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

    # Classify each new message into one of: an "opened N" request, a "closed T S P" request,
    # or (if a disambiguation is currently pending) a plain reply naming an expiry date.
    # Everything else is ignored. Every update still advances last_update_id, so nothing gets
    # reprocessed on future runs regardless of whether we acted on it.
    opened_requests = []
    closed_requests = []
    disambiguation_replies = []
    unrecognized = []  # (chat_id-safe) raw text of messages that matched nothing, for a reply

    for update in updates:
        state["last_update_id"] = max(state["last_update_id"], update["update_id"])
        message = update.get("message")
        if not message:
            continue
        if str(message.get("chat", {}).get("id")) != str(chat_id):
            print(f"Ignoring message from unauthorized chat {message.get('chat', {}).get('id')}.")
            continue

        text = message.get("text", "")
        message_sent_at = message.get("date")

        opened_match = OPENED_PATTERN.match(text)
        closed_match = CLOSED_PATTERN.match(text)

        if opened_match:
            if signal is not None and message_sent_at is not None and message_sent_at < signal["sent_at"]:
                print(f"Ignoring \"{text}\" — sent before today's signal, refers to a previous day's positions.")
                continue
            # group(1) is everything after "opened ", e.g. "1,2,3" or "1, 2 3" — split on any
            # mix of commas/whitespace and de-duplicate while preserving order.
            numbers = [int(n) for n in re.split(r"[,\s]+", opened_match.group(1).strip()) if n]
            for n in numbers:
                if n not in opened_requests:
                    opened_requests.append(n)
        elif closed_match:
            closed_requests.append({
                "strike": float(closed_match.group(1)),
                "ticker": closed_match.group(2),
                "profit": float(closed_match.group(3)),
            })
        elif state.get("pending_close"):
            disambiguation_replies.append(text.strip())
        else:
            unrecognized.append(text)

    for text in unrecognized:
        send_telegram_message(
            bot_token, chat_id,
            f"⚠️ Didn't understand: \"{text}\". Expected formats are \"opened &lt;number&gt;\" "
            f"(or \"opened 1,2,3\" for several), or \"closed &lt;strike&gt; &lt;ticker&gt; &lt;profit&gt;\"."
        )
        print(f"Unrecognized message, replied with format help: \"{text}\"")

    if not opened_requests and not closed_requests and not disambiguation_replies:
        print(f"No actionable messages among {len(updates)} update(s).")
        save_state(state)
        return

    # --- Handle "opened N" ---
    if opened_requests and signal is None:
        for n in opened_requests:
            send_telegram_message(bot_token, chat_id,
                                   f"⚠️ Got \"opened {n}\" but no signal has been sent yet today — nothing to log.")
        opened_requests = []

    # Only connect to Google Sheets if there's actually something to do there.
    if opened_requests or closed_requests or disambiguation_replies:
        gc = gspread.service_account_from_dict(json.loads(service_account_json))
        sheet = gc.open_by_key(sheet_id)
        worksheet = get_or_create_worksheet(sheet)
        header_row = ensure_worksheet_headers(worksheet)
    else:
        worksheet = header_row = None

    if opened_requests:
        signal_date = signal["date"]
        logged_today = set(state["logged_positions"].get(signal_date, []))

        for n in opened_requests:
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
            append_position_to_sheet(worksheet, header_row, position)
            logged_today.add(n)
            send_telegram_message(
                bot_token, chat_id,
                f"✅ Logged to tracker: <b>{position['ticker']}</b> "
                f"(strike ${position['strike']:.2f}, short {position['short_call_expiry']}, "
                f"long {position['long_call_expiry']})."
            )
            print(f"Logged position {n} ({position['ticker']}) to the sheet.")

        state["logged_positions"][signal_date] = sorted(logged_today)

    # --- Handle "closed <strike> <ticker> <profit>" ---
    for req in closed_requests:
        matches = find_open_positions(worksheet, header_row, req["ticker"], req["strike"])

        if not matches:
            send_telegram_message(
                bot_token, chat_id,
                f"⚠️ No open position found for <b>{req['ticker']}</b> at strike ${req['strike']:.2f}. "
                f"Nothing closed."
            )
            continue

        if len(matches) == 1:
            row_number, record = matches[0]
            exit_cashflow = close_position_row(worksheet, header_row, row_number, record, req["profit"])
            send_telegram_message(
                bot_token, chat_id,
                f"✅ Closed <b>{req['ticker']}</b> strike ${req['strike']:.2f} "
                f"(expiry {record.get('Short Call Expiry', 'N/A')}). "
                f"Profit: ${req['profit']:.2f}, Exit Cashflow: ${exit_cashflow:.2f}."
            )
            print(f"Closed {req['ticker']} strike {req['strike']} (row {row_number}).")
        else:
            # Ambiguous — ask which expiry, and remember this request until the reply arrives
            # (possibly on a LATER run of this workflow).
            state["pending_close"] = {
                "ticker": req["ticker"],
                "strike": req["strike"],
                "profit": req["profit"],
                "candidates": [
                    {"row": row_number, **record} for row_number, record in matches
                ],
            }
            send_telegram_message(bot_token, chat_id, format_disambiguation_message(req["ticker"], req["strike"], matches))
            print(f"Ambiguous close for {req['ticker']} strike {req['strike']} — asked for expiry.")

    # --- Handle a reply resolving a pending disambiguation ---
    pending = state.get("pending_close")
    for reply_text in disambiguation_replies:
        if not pending:
            break  # already resolved by an earlier reply in this same batch

        reply_norm = reply_text.strip().lower()
        matched_candidates = [
            c for c in pending["candidates"]
            if reply_norm == str(c.get("Short Call Expiry", "")).strip().lower()
            or reply_norm in str(c.get("Short Call Expiry", "")).strip().lower()
        ]

        if not matched_candidates:
            send_telegram_message(
                bot_token, chat_id,
                f"⚠️ \"{reply_text}\" didn't match any of the expiry dates for the pending close on "
                f"<b>{pending['ticker']}</b> strike ${pending['strike']:.2f}. "
                f"Please reply with one of the listed dates exactly."
            )
            continue

        for candidate in matched_candidates:
            row_number = candidate["row"]
            close_position_row(worksheet, header_row, row_number, candidate, pending["profit"])
            print(f"Resolved ambiguous close for {pending['ticker']} strike {pending['strike']} (row {row_number}).")

        if len(matched_candidates) == 1:
            candidate = matched_candidates[0]
            # Re-derive exit cashflow just for the message (close_position_row already wrote it).
            try:
                entry_cf = float(candidate.get("Entry Cashflow", 0) or 0)
            except (TypeError, ValueError):
                entry_cf = 0.0
            exit_cashflow = round(pending["profit"] - entry_cf, 2)
            send_telegram_message(
                bot_token, chat_id,
                f"✅ Closed <b>{pending['ticker']}</b> strike ${pending['strike']:.2f} "
                f"(expiry {candidate.get('Short Call Expiry', 'N/A')}). "
                f"Profit: ${pending['profit']:.2f}, Exit Cashflow: ${exit_cashflow:.2f}."
            )
        else:
            # Genuine duplicates: same ticker, strike, AND expiry — there's no way to tell them
            # apart from the reply alone, so close all of them rather than guessing which one
            # was meant, but flag it clearly since one of these closures may be wrong.
            send_telegram_message(
                bot_token, chat_id,
                f"⚠️ {len(matched_candidates)} open positions matched <b>{pending['ticker']}</b> "
                f"strike ${pending['strike']:.2f} at expiry {reply_text.strip()} — all {len(matched_candidates)} "
                f"have been closed with profit ${pending['profit']:.2f}. Please manually review the tracker "
                f"to confirm this is correct."
            )

        pending = None
        state["pending_close"] = None

    save_state(state)


if __name__ == "__main__":
    main()

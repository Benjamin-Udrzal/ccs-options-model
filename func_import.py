######################################
## Importing Packages
######################################

import os.path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from polygon import RESTClient
import requests
from py_vollib.black_scholes import implied_volatility
from py_vollib.black_scholes.greeks import analytical
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


def func_import(
        polygon_api_key: str = None,
        shuffle_training_dataset = True,
        train_test_random_seed = 1,
        run_reason :str = None,
        default_settings :bool = None,
        sample_size :int = None,
        ticker_list_str :str = None,
        test_split :float = None,
        training_dataset = None,
        output_dir: str = None  # if None, falls back to a folder next to this script.
                                 # Pass a path (e.g. "outputs") to control where CSVs land -
                                 # used by the GitHub Actions runner to save inside the repo checkout.







        ):
    ###############################################################################
    # -------------------------- USER CONFIGURATION -------------------------------
    ###############################################################################


    # API key from config
    client = RESTClient(polygon_api_key, retries = 5)
    #API key kept in secrets

    prediction_model_folder = "Model v7"

    ticker_map = {
        "SPY": 1,
        "IWM": 2,
        "QQQ": 3,
        "DIA": 4,
        "XLF": 5,
        "GLD": 6,
        "XLE": 7,
        "XLV": 8,
        "XLI": 9,
        "TLT": 10,
        "HYG": 11,
        "EEM": 12,
        "XLU": 13,
        "XLY": 14,
        "XLB": 15,
        "XLP": 16,
        "FXI": 17,
        "SLV": 18,
    }

    ###############################################################################
    # ------------------------ STRATEGY TIMING CONFIGURATION ----------------------
    ###############################################################################

    front_leg_target_days = 18   # calendar days from entry to target front (short) leg expiry
    back_leg_target_days = 35    # calendar days from entry to target back (long) leg expiry
    exit_business_days_before_front_expiry = 1  # close position this many trading days before front expiry
    option_expiry_search_buffer_days = 21  # extra buffer beyond back-leg target when searching for expirations
    strike_search_window_pct = 0.015  # +/- window around price range when pulling contract strikes


    decision_time_local = time(17, 0)
    decision_tz = ZoneInfo("Europe/London")
    ny_tz = ZoneInfo("America/New_York")

    randomise_entry_weekday= True  # spreads training/testing entry dates across different weekdays
    trade_frequency_days = 7  # calendar days between new entries per ticker (weekly cadence chosen:
                               # ~3 overlapping ~3-week positions per ticker at any given time). Change
                               # this in one place if later want a different cadence — everything
                               # else (dtest/live flow, training data generation) reads from it.

    historical_data_limit_years = 15  # set this to match how far back API plan actually goes
                                   # build_entry_dates() clips to this automatically, so an
                                   # oversized sample_size degrades gracefully (fewer rows, with a
                                   # printed warning) instead of the API erroring on out-of-range dates.

    ###############################################################################
    # -------------------------- FUNCTION (DEFAULT) SETTINGS ------------------------
    ###############################################################################

    samplesize_test_default = 30
    samplesize_train_default = 1500  # ~1 entry/week x ~5 years of history (adjust to match actual
                                 # API data-access window — see HISTORICAL_DATA_LIMIT_YEARS above)
    ticker_list_default = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'GLD', 'XLE', 'XLV', 'XLI', 'TLT', 'HYG', 'EEM',
                        'XLU', 'XLY', 'XLB', 'XLP', 'FXI', 'SLV']

    run_reason_options = ["train", "test", "dtest"]

    test_split_default = 0.20

    combined_stock_data = []
    combined_call_option_data = []
    combined_call_spread_pt1 = []
    combined_call_spread_pt2 = []
    combined_call_option_prices = []
    combined_call_calendar_spread_summary = []
    combined_IntraDayStockPrice = []


    us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    ######################################
    ## Function Inputs and Assigning Variables
    ######################################

    if run_reason == "train":
        if default_settings:
            sample_size = samplesize_train_default
            ticker_list = ticker_list_default
            test_split = test_split_default
        else:
            if not (50 <= sample_size <= 2000):
                ValueError(f"When training the sample size must be between 50 and 2000")
            ticker_list = [item.strip().upper() for item in ticker_list_str.split(",")]
            if not (0 <= test_split <= 0.99):
                ValueError(f"When training the test split must be between 0 and 0.99")


    elif run_reason in ("test", "dtest"):
        if default_settings:
            sample_size = samplesize_test_default
            ticker_list = ticker_list_default
            test_split = 1
        else:
            if not (25 <= sample_size <= 200):
                ValueError(f"When testing the sample size must be between 25 and 200")
            ticker_list = [item.strip().upper() for item in ticker_list_str.split(",")]
            test_split = 1
    else:
        raise ValueError(f"Import function requires one of the following as run_reason: {run_reason_options}")


    ######################################
    ## Helper Functions
    ######################################

    def func_left_join_on_closest(left_df, right_df, left_on, right_on, prefix):
        def find_closest(val):
            if pd.isna(val):
                return pd.NaT
            diffs = (right_df[right_on] - val).abs()
            if diffs.isna().all():
                return pd.NaT
            idx = diffs.idxmin()
            return right_df.loc[idx, right_on]

        matched = pd.Series(
            [find_closest(val) for val in left_df[left_on]],
            name=prefix + right_on
        )
        return pd.concat([left_df.reset_index(drop=True), matched], axis=1)


    def uk_decision_time_to_ny(date_):
        """
        Converts 5pm UK time on a given calendar date into the equivalent US Eastern
        datetime, automatically handling the (slightly offset) UK/US daylight saving
        transitions. This is the cutoff we use for 'what was actually knowable' when
        building decision-time price features.
        """
        uk_dt = datetime.combine(date_, decision_time_local, tzinfo=decision_tz)
        return uk_dt.astimezone(ny_tz)


    def func_left_join_on_decision_price(left_df, right_df, left_on, right_on):
        """
        Left join on calendar date. For each date, finds the LATEST intraday bar at or
        before the NY-time equivalent of 5pm UK on that date (i.e. only data that would
        genuinely have existed at your real decision point). If no bar exists yet at/before
        that time (e.g. a data gap), returns NaN rather than silently using a later bar.
        """
        right_dt = pd.to_datetime(right_df[right_on])
        right_tmp = right_df.copy()
        right_tmp["_date"] = right_dt.dt.date
        right_tmp["_dt"] = right_dt
        grouped = right_tmp.groupby("_date")

        def find_values_for_date(val):
            if pd.isna(val):
                return pd.NaT, np.nan
            if val not in grouped.groups:
                return pd.NaT, np.nan
            day_df = grouped.get_group(val)
            decision_dt_ny = uk_decision_time_to_ny(val)
            eligible = day_df[day_df["_dt"] <= decision_dt_ny]
            if eligible.empty:
                return pd.NaT, np.nan
            idx = eligible["_dt"].idxmax()
            row = right_tmp.loc[idx]
            return row[right_on], row["close"]

        left_dates = left_df[left_on]
        if not np.issubdtype(pd.Series(left_dates).dtype, np.datetime64):
            left_dates = pd.to_datetime(left_dates).dt.date
        else:
            left_dates = left_dates.dt.date

        matched_values = [find_values_for_date(val) for val in left_dates]
        matched = pd.DataFrame(matched_values, columns=[right_on + "_decision", "close_decision"])

        return pd.concat([left_df.reset_index(drop=True), matched], axis=1)


    ######################################
    ## Functions for calculating metrics
    ######################################

    def calculate_iv_and_greeks(tickers, option_data, option_type, r=0.0432):
        results = []
        for ticker in range(len(tickers)):
            data = option_data.iloc[ticker]
            S = data.iloc[1]
            K = data.iloc[2]
            T = data.iloc[3]
            flag = option_type
            option_price = data.iloc[4]

            try:
                iv = implied_volatility.implied_volatility(option_price, S, K, T, r, flag)
                delta = analytical.delta(flag, S, K, T, r, iv)
                gamma = analytical.gamma(flag, S, K, T, r, iv)
                theta = analytical.theta(flag, S, K, T, r, iv)
                vega = analytical.vega(flag, S, K, T, r, iv)
                rho = analytical.rho(flag, S, K, T, r, iv)
                results.append({"implied_volatility": iv, "delta": delta, "gamma": gamma,
                                 "theta": theta, "vega": vega, "rho": rho})
            except Exception as e:
                results.append({"implied_volatility": None, "delta": None, "gamma": None,
                                 "theta": None, "vega": None, "rho": None, "error": str(e)})
        return pd.DataFrame(results)


    def compute_realized_volatility(df_intraday):
        results = []
        for date_, group in df_intraday.groupby('Share_Price_Date'):
            group = group.sort_values(by='Share_Price_Datetime')
            log_returns = np.log(group['close']).diff().dropna()
            daily_vol = 100 * (np.sqrt(np.sum(log_returns ** 2)))

            mid_time = pd.Timestamp(f"{date_} 12:45:00", tz="America/New_York")
            second_half = group[group['Share_Price_Datetime'] >= mid_time]
            log_returns_second_half = np.log(second_half['close']).diff().dropna()
            second_half_vol = 100 * (np.sqrt(np.sum(log_returns_second_half ** 2))) if len(log_returns_second_half) > 0 else np.nan

            results.append({'daily_vol': daily_vol, 'second_half_vol': second_half_vol, 'date': date_})
        return pd.DataFrame(results)


    def linear_regression_slope(y):
        x = np.arange(len(y))
        m, _ = np.polyfit(x, y, 1)
        return m


    def compute_adx(df, period=14):
        high = df['high']
        low = df['low']
        close = df['previous_close']

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return adx, plus_di, minus_di


    def select_leg_expirations(entry_date_, available_expirations, front_target_days, back_target_days):
        """
        Given an entry date and the sorted list of available expiration dates for a
        (ticker, strike), returns (front_expiration, back_expiration) chosen as the
        NEAREST available expiration at/after the target day count. Returns (NaT, NaT)
        or (front, NaT) if a suitable leg can't be found (e.g. not enough listed expiries).
        """
        if not available_expirations:
            return pd.NaT, pd.NaT

        front_candidates = [d for d in available_expirations if d >= entry_date_ + timedelta(days=front_target_days)]
        if not front_candidates:
            return pd.NaT, pd.NaT
        front_expiration = min(front_candidates)

        back_candidates = [d for d in available_expirations
                            if d >= entry_date_ + timedelta(days=back_target_days) and d > front_expiration]
        if not back_candidates:
            return front_expiration, pd.NaT
        back_expiration = min(back_candidates)

        return front_expiration, back_expiration


    def build_entry_dates(sample_size_, mode, seed):
        """
        train/test: builds `sample_size_` entry dates spaced TRADE_FREQUENCY_DAYS apart
        (calendar days), with the weekday randomised within each window (mirrors real
        usage: open a new position on whatever day suits you, roughly every
        TRADE_FREQUENCY_DAYS), which also reduces overlap between adjacent training
        trades' holding periods relative to sampling every single business day.

        dtest: builds a run of consecutive recent business days ending today, so that
        "today" (whatever day you happen to run the script) is guaranteed to be included;
        the calling code later filters this down to just today's row. Live trading isn't
        restricted to a fixed weekday — you run this whenever suits you, roughly every
        TRADE_FREQUENCY_DAYS (see the accompanying explanation for exact guidance).
        """
        today_ = datetime.today().date()

        if mode == "dtest":
            days = pd.date_range(end=today_, periods=sample_size_, freq=us_bday)
            return [d.date() for d in days]

        # Generous pool of business days to draw samples from
        pool = pd.date_range(end=today_, periods=sample_size_ * trade_frequency_days * 2, freq=us_bday)
        pool_df = pd.DataFrame({'d': pool})
        # Bucket by calendar days-since-epoch // TRADE_FREQUENCY_DAYS, rather than a fixed
        # ISO week — this generalises cleanly to any cadence (weekly, fortnightly, etc.),
        # not just exactly-7-day windows.
        epoch_days = (pool_df['d'] - pd.Timestamp("1970-01-01")).dt.days
        pool_df['period_key'] = epoch_days // trade_frequency_days

        if randomise_entry_weekday:
            rng = np.random.default_rng(seed)
            chosen = pool_df.groupby('period_key')['d'].apply(lambda s: rng.choice(s.values))
        else:
            chosen = pool_df.groupby('period_key')['d'].first()

        chosen = sorted(pd.to_datetime(chosen.values))
        chosen = chosen[-sample_size_:]

        earliest_allowed = pd.Timestamp(today_) - pd.Timedelta(days=365 * historical_data_limit_years)
        within_range = [d for d in chosen if d >= earliest_allowed]
        if len(within_range) < len(chosen):
            print(f"WARNING: requested sample_size={sample_size_} needed dates back to "
                  f"{chosen[0].date()}, which exceeds your configured {historical_data_limit_years}-year "
                  f"data access window (earliest allowed: {earliest_allowed.date()}). "
                  f"Dropped {len(chosen) - len(within_range)} entry date(s) rather than risk an API error. "
                  f"Consider lowering sample_size_train_default if you see this often.")

        return [pd.Timestamp(d).date() for d in within_range]


    ######################################
    ## Full Data Import Process
    ######################################

    for ticker in ticker_list:
        print(f"============================")
        print(f"Processing ticker: {ticker}")
        print(f"============================")

        ######################################
        ## Getting list of entry dates for sample
        ######################################

        print("Getting list of sample entry dates...")
        entry_dates = build_entry_dates(sample_size, run_reason, train_test_random_seed)
        if run_reason == 'dtest':
            dtest_actual_date = entry_dates[-1]  # capture the exact date used to build entries, so the
                                                  # final filter below matches it exactly — see fix notes
                                                  # near the dtest filtering section further down
        fetch_start = entry_dates[0]
        fetch_end = entry_dates[-1] + timedelta(days=back_leg_target_days + option_expiry_search_buffer_days)

        ######################################
        # Getting stock price data (covers entry AND potential exit dates)
        ######################################

        print("Getting stock price data...")
        stock_price_data_request = client.get_aggs(ticker=ticker, multiplier=1, timespan='day',
                                                    from_=fetch_start, to=min(fetch_end, datetime.today().date()))
        today = datetime.today()
        today_stock_price_data_request = client.get_snapshot_ticker(ticker=ticker, market_type='stocks')

        stock_price_data = pd.DataFrame(stock_price_data_request)
        stock_price_data['Share_Price_Date'] = pd.to_datetime(stock_price_data['timestamp'], unit='ms').dt.date

        if today.date() not in stock_price_data['Share_Price_Date'].values:
            stock_price_data.loc[len(stock_price_data)] = [
                today_stock_price_data_request.day.open, today_stock_price_data_request.day.high,
                today_stock_price_data_request.day.low, today_stock_price_data_request.day.close,
                today_stock_price_data_request.day.volume, today_stock_price_data_request.day.vwap,
                "", "", "", today.date()
            ]

        ######################################
        ## Getting intraday stock price data
        ######################################

        print("Getting intraday stock prices...")
        stock_price_intra_day_data_request = client.list_aggs(ticker=ticker, multiplier=5, timespan="minute",
                                                                from_=fetch_start, to=min(fetch_end, datetime.today().date()),
                                                                limit=1000000)

        stock_price_intra_day_data = pd.DataFrame(stock_price_intra_day_data_request)
        stock_price_intra_day_data['Share_Price_Datetime'] = pd.to_datetime(
            stock_price_intra_day_data['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')
        stock_price_intra_day_data['Share_Price_Date'] = stock_price_intra_day_data['Share_Price_Datetime'].dt.date

        # Keep pre-market + regular hours; decision time (~noon-ish ET for a 5pm UK check) sits
        # well within this window, but we keep the same upper filter as before for safety.
        stock_price_intra_day_data_filtered = stock_price_intra_day_data[
            stock_price_intra_day_data['Share_Price_Datetime'].dt.hour < 16].copy()

        ######################################
        ## Adding decision-time price to stock price data table
        ######################################

        stock_price_data = func_left_join_on_decision_price(
            left_df=stock_price_data, right_df=stock_price_intra_day_data_filtered,
            left_on='Share_Price_Date', right_on='Share_Price_Datetime')
        stock_price_data.rename(columns={'Share_Price_Datetime_decision': 'Decision_Datetime',
                                          'close_decision': 'Decision_Price'}, inplace=True)

        ######################################
        ## Calculating metrics for stock price data
        ## NOTE: these use each day's own high/low/close, which are only fully known AFTER that
        ## day's market close. They are valid to use as "yesterday's fully-known indicators" via
        ## the Feature_Date shift applied later — do not treat a row's own indicators as knowable
        ## on that same calendar date.
        ######################################

        print("Calculating stock metrics...")

        stock_price_data['previous_close'] = stock_price_data['close'].shift(1)

        stock_price_data['TR'] = stock_price_data[['high', 'low', 'previous_close']].apply(
            lambda row: round(max(
                row['high'] - row['low'],
                abs(row['high'] - row['previous_close']),
                abs(row['low'] - row['previous_close'])
            ), 4), axis=1)
        stock_price_data['Trend_Efficiency'] = (stock_price_data['close'] - stock_price_data['open']).abs() / stock_price_data['TR']
        stock_price_data['ATR3'] = stock_price_data['TR'].rolling(window=3).mean()
        stock_price_data['ATR7'] = stock_price_data['TR'].rolling(window=7).mean()
        stock_price_data['ATR14'] = stock_price_data['TR'].rolling(window=14).mean()

        stock_price_data['Normalised_ATR'] = stock_price_data['ATR3'] / stock_price_data['open']
        stock_price_data['Vol_Expanding'] = (stock_price_data['ATR3'] / stock_price_data['ATR14'] > 1.2).astype(int)

        stock_price_data['MA10'] = stock_price_data['close'].rolling(window=10).mean()
        stock_price_data['MA3'] = stock_price_data['close'].rolling(window=3).mean()
        stock_price_data['STD10'] = stock_price_data['close'].rolling(window=10).std()
        stock_price_data['STD3'] = stock_price_data['close'].rolling(window=3).std()

        stock_price_data['Trend_Distance'] = (stock_price_data['close'] - stock_price_data['MA3']) / stock_price_data['ATR7']
        stock_price_data['Trend_Distance_slope_5'] = (
            stock_price_data["Trend_Distance"].rolling(window=5).apply(linear_regression_slope, raw=True))

        stock_price_data['UpperBand10'] = stock_price_data['MA10'] + (2 * stock_price_data['STD10'])
        stock_price_data['LowerBand10'] = stock_price_data['MA10'] - (2 * stock_price_data['STD10'])
        stock_price_data['UpperBand3'] = stock_price_data['MA3'] + (2 * stock_price_data['STD3'])
        stock_price_data['LowerBand3'] = stock_price_data['MA3'] - (2 * stock_price_data['STD3'])

        stock_price_data['BBW10'] = (stock_price_data['UpperBand10'] - stock_price_data['LowerBand10']) / stock_price_data['MA10']
        stock_price_data['BBW3'] = (stock_price_data['UpperBand3'] - stock_price_data['LowerBand3']) / stock_price_data['MA3']
        stock_price_data = stock_price_data.drop(columns=['UpperBand10', 'LowerBand10', 'UpperBand3', 'LowerBand3'])

        stock_price_data['ADX_Daily'], stock_price_data['Plus_DI_Daily'], stock_price_data['Minus_DI_Daily'] = compute_adx(stock_price_data, period=14)
        stock_price_data['ADX_Daily_slope_5'] = (
            stock_price_data['ADX_Daily'].rolling(window=5).apply(linear_regression_slope, raw=True))

        stock_price_data['Is_Trending'] = (stock_price_data['ADX_Daily'] > 25).astype(int)

        stock_price_data['log_return'] = np.log(stock_price_data['close'] / stock_price_data['close'].shift(1))
        stock_price_data['RV_3'] = stock_price_data['log_return'].rolling(window=3).std().reset_index(level=0, drop=True)
        stock_price_data['RV_5'] = stock_price_data['log_return'].rolling(window=5).std().reset_index(level=0, drop=True)
        stock_price_data['RV_14'] = stock_price_data['log_return'].rolling(window=14).std().reset_index(level=0, drop=True)
        stock_price_data = stock_price_data.drop(columns=['log_return'])
        stock_price_data['Vol_Regime_Ratio'] = stock_price_data['RV_3'] / stock_price_data['RV_14']

        ######################################
        ## Calculating metrics on intraday stock price data
        ######################################

        print("Calculating metrics on intraday stock prices...")
        realised_volatility = compute_realized_volatility(stock_price_intra_day_data_filtered)
        realised_volatility['date'] = pd.to_datetime(realised_volatility["date"]).dt.date
        stock_price_data = pd.merge(stock_price_data, realised_volatility, how='left', left_on='Share_Price_Date', right_on='date')
        stock_price_data.drop(columns=['date'], inplace=True)

        ######################################
        ## Setting up 'Call Calendar Spread' Strategy — ENTRY leg
        ######################################

        print("Setting up Call Calendar Spread strategy (entry)...")

        CCS_day1 = pd.DataFrame(entry_dates, columns=['Share_Price_Date'])
        CCS_day1 = pd.merge(CCS_day1, stock_price_data[['Share_Price_Date', 'Decision_Price']],
                             how='left', left_on='Share_Price_Date', right_on='Share_Price_Date')
        CCS_day1.rename(columns={'Decision_Price': 'Sim_close'}, inplace=True)  # keep 'Sim_close' name downstream for compatibility
        CCS_day1.insert(1, 'Stock_Ticker', ticker)

        drop_idx_no_price = CCS_day1[CCS_day1['Sim_close'].isna()].index
        for idx in drop_idx_no_price:
            print(f"Dropped entry row | index={idx} | date={CCS_day1.loc[idx, 'Share_Price_Date']} (no decision-time price)")
        CCS_day1 = CCS_day1.drop(index=drop_idx_no_price).reset_index(drop=True)

        CCS_day1_len = len(CCS_day1)  # single strike per date now — no x3 multiplier

        ######################################
        ## Getting options contract data (covers full front+back leg search window)
        ######################################

        print("Getting options price contract data...")

        call_option_contract_data_request = []
        for c in client.list_options_contracts(underlying_ticker=ticker, contract_type='call',
                                                expiration_date_gte=CCS_day1['Share_Price_Date'].iloc[0],
                                                expiration_date_lte=fetch_end,
                                                strike_price_gte=stock_price_data['Decision_Price'].min() * (1 - strike_search_window_pct),
                                                strike_price_lte=stock_price_data['Decision_Price'].max() * (1 + strike_search_window_pct),
                                                expired=True, limit=1000):
            call_option_contract_data_request.append(c)

        for c in client.list_options_contracts(underlying_ticker=ticker, contract_type='call',
                                                expiration_date_gte=CCS_day1['Share_Price_Date'].iloc[0],
                                                expiration_date_lte=fetch_end,
                                                strike_price_gte=stock_price_data['Decision_Price'].min() * (1 - strike_search_window_pct),
                                                strike_price_lte=stock_price_data['Decision_Price'].max() * (1 + strike_search_window_pct),
                                                expired=False, limit=1000):
            call_option_contract_data_request.append(c)

        call_option_contract_data = pd.DataFrame(call_option_contract_data_request)
        call_option_contract_data['expiration_date'] = pd.to_datetime(call_option_contract_data['expiration_date']).dt.date

        call_option_contract_data = call_option_contract_data[call_option_contract_data['strike_price'] % 1 == 0]
        call_option_contract_data = call_option_contract_data[call_option_contract_data['shares_per_contract'] == 100]

        exchange_priority = ["BATO", "C2OX", "XBOX", "XBSE"]
        priority_map = {exch: i for i, exch in enumerate(exchange_priority)}
        call_option_contract_data['exchange_priority'] = call_option_contract_data['primary_exchange'].map(priority_map)
        call_option_contract_data['exchange_priority'] = call_option_contract_data['exchange_priority'].fillna(len(exchange_priority))
        call_option_contract_data = call_option_contract_data.sort_values('exchange_priority').drop_duplicates(subset='ticker', keep='first')
        # Guard against duplicate (expiration_date, strike_price) pairs — these can arise from
        # corporate-action-adjusted contract series (e.g. special dividends/spin-offs) that carry
        # a different ticker but the same nominal strike/expiration as the standard contract.
        # Since downstream merges key on (expiration_date, strike_price), any leftover collision
        # here would silently duplicate rows in CCS_day1.
        call_option_contract_data = call_option_contract_data.drop_duplicates(subset=['expiration_date', 'strike_price'], keep='first')

        ######################################
        ## Selecting ATM strike, then front/back leg expirations, then exit date
        ######################################

        print("Selecting strike and front/back leg expirations...")

        CCS_day1 = func_left_join_on_closest(left_df=CCS_day1, right_df=call_option_contract_data,
                                              left_on='Sim_close', right_on='strike_price', prefix='Call_')
        CCS_day1['Call_strike_price'] = pd.to_numeric(CCS_day1['Call_strike_price'], errors='coerce').astype('float64')

        if len(CCS_day1) != CCS_day1_len:
            raise ValueError("CCS_day1 gained rows when joining on strike price")

        # Distinct available expirations, grouped by strike, for fast per-row lookup.
        # RESTRICTED TO FRIDAYS: tickers like SPY/QQQ/IWM/GLD list options expiring on
        # every weekday, but Mon/Tue/Wed/Thu expirations several weeks out are highly
        # illiquid (most of their volume concentrates in the final day or two before
        # expiry) — Polygon's daily aggregate often has no bar at all for these on most
        # days. The standard Friday cycle (which also covers monthly expirations, always
        # the 3rd Friday) is where liquidity actually sits for a multi-week calendar
        # spread, so we only search among Friday expirations.
        RESTRICT_TO_FRIDAY_EXPIRATIONS = True
        expirations_by_strike = call_option_contract_data.groupby('strike_price')['expiration_date'].apply(
            lambda s: sorted({d for d in s if (not RESTRICT_TO_FRIDAY_EXPIRATIONS) or d.weekday() == 4})).to_dict()

        front_expirations = []
        back_expirations = []
        for _, row in CCS_day1.iterrows():
            avail = expirations_by_strike.get(row['Call_strike_price'], [])
            f_exp, b_exp = select_leg_expirations(row['Share_Price_Date'], avail, front_leg_target_days, back_leg_target_days)
            front_expirations.append(f_exp)
            back_expirations.append(b_exp)

        CCS_day1['Call_1_expiration_date'] = front_expirations  # Call_1 = front / short leg
        CCS_day1['Call_2_expiration_date'] = back_expirations   # Call_2 = back / long leg

        # Drop rows where we couldn't find a suitable front or back leg
        missing_legs_idx = CCS_day1[CCS_day1['Call_1_expiration_date'].isna() | CCS_day1['Call_2_expiration_date'].isna()].index
        for idx in missing_legs_idx:
            print(f"Dropped entry row | index={idx} | date={CCS_day1.loc[idx, 'Share_Price_Date']} (no suitable front/back leg found)")
        CCS_day1 = CCS_day1.drop(index=missing_legs_idx).reset_index(drop=True)
        CCS_day1_len = len(CCS_day1)

        CCS_day1['Call_1_Time_to_Expiration'] = (CCS_day1['Call_1_expiration_date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days) / 365
        CCS_day1['Call_2_Time_to_Expiration'] = (CCS_day1['Call_2_expiration_date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days) / 365

        # Exit date = N business days before front leg expiry
        CCS_day1['Exit_Date'] = CCS_day1['Call_1_expiration_date'].apply(
            lambda d: (pd.Timestamp(d) - exit_business_days_before_front_expiry * us_bday).date())

        CCS_day1 = pd.merge(CCS_day1, call_option_contract_data[['expiration_date', 'strike_price', 'ticker']],
                             how='left', left_on=['', 'Call_strike_price'],
                             right_on=['expiration_date', 'strike_price'])
        CCS_day1.drop(columns=['expiration_date', 'strike_price'], inplace=True)
        CCS_day1.rename(columns={'ticker': 'Call_1_Option_Ticker'}, inplace=True)

        CCS_day1 = pd.merge(CCS_day1, call_option_contract_data[['expiration_date', 'strike_price', 'ticker']],
                             how='left', left_on=['Call_2_expiration_date', 'Call_strike_price'],
                             right_on=['expiration_date', 'strike_price'])
        CCS_day1.drop(columns=['expiration_date', 'strike_price'], inplace=True)
        CCS_day1.rename(columns={'ticker': 'Call_2_Option_Ticker'}, inplace=True)

        if len(CCS_day1) != CCS_day1_len:
            raise ValueError("CCS_day1 gained rows when joining on call option contract tickers")

        ######################################
        ## Building the EXIT leg (CCS_day2) from the computed Exit_Date
        ######################################

        print("Setting up Call Calendar Spread strategy (exit)...")

        CCS_day2 = pd.DataFrame({
            'Share_Price_Date': CCS_day1['Exit_Date'],
            'Stock_Ticker': ticker,
            'Call_strike_price': CCS_day1['Call_strike_price'],
            '': CCS_day1['Call_1_expiration_date'],
            'Call_2_expiration_date': CCS_day1['Call_2_expiration_date'],
            'Call_1_Option_Ticker': CCS_day1['Call_1_Option_Ticker'],
            'Call_2_Option_Ticker': CCS_day1['Call_2_Option_Ticker'],
        })
        CCS_day2 = pd.merge(CCS_day2, stock_price_data[['Share_Price_Date', 'Decision_Price']],
                             how='left', left_on='Share_Price_Date', right_on='Share_Price_Date')
        CCS_day2.rename(columns={'Decision_Price': 'Sim_close'}, inplace=True)

        CCS_day2_len = len(CCS_day2)
        if len(CCS_day2) != CCS_day1_len:
            raise ValueError("CCS_day2 length does not match CCS_day1 length after construction")

        ######################################
        ## Getting call option contract prices (entry AND exit dates for each contract)
        ######################################

        option_price_data_request = pd.DataFrame({
            "Options_Ticker": pd.concat([CCS_day1["Call_1_Option_Ticker"], CCS_day1["Call_2_Option_Ticker"]], ignore_index=True),
            "First_Date": pd.concat([CCS_day1["Share_Price_Date"], CCS_day1["Share_Price_Date"]], ignore_index=True),
            "Second_Date": pd.concat([CCS_day1["Exit_Date"], CCS_day1["Exit_Date"]], ignore_index=True)
        }).dropna(subset=["Options_Ticker"])

        print(f"Generating prices for {len(option_price_data_request)} options tickers...")

        call_option_price_data = pd.DataFrame(columns=['Call_Options_Ticker', 'Call_Market_Price', 'Call_Option_Price_Date_UNIX'])

        num_options_tickers = len(option_price_data_request)
        for i in range(num_options_tickers):
            call_price = client.get_aggs(ticker=option_price_data_request['Options_Ticker'].iloc[i],
                                          multiplier=1, timespan='day',
                                          from_=option_price_data_request['First_Date'].iloc[i],
                                          to=option_price_data_request['Second_Date'].iloc[i])

            for x in range(len(call_price)):
                call_option_price_data.loc[len(call_option_price_data)] = [
                    option_price_data_request['Options_Ticker'].iloc[i], call_price[x].close, call_price[x].timestamp]

            if num_options_tickers > 0 and (i % max(1, num_options_tickers // 10) == 0 or i == num_options_tickers - 1):
                percent = (i / num_options_tickers) * 100
                print(f"{percent:.0f}% of options tickers completed")

        call_option_price_data = call_option_price_data.drop_duplicates()
        call_option_price_data['Call_Option_Price_Date'] = pd.to_datetime(
            call_option_price_data['Call_Option_Price_Date_UNIX'], unit='ms').dt.date

        # --- Decision-time intraday lookup for TODAY's option price (dtest/live runs) ---
        # Daily aggregate bars only finalize after market close (~9pm UK), but a dtest run at 5pm UK
        # happens well before that, so client.get_aggs(timespan='day') has NO bar yet for today's
        # entry-leg option price. The fix isn't "grab whatever the price is right now" (that would be
        # a 5:05pm price if you happen to run 5 minutes late, not a 5pm one) — it's to give options the
        # EXACT SAME decision-time treatment already used for the underlying stock: fetch 5-minute
        # intraday bars and pick the latest one at/before 5pm UK. This only fetches the handful of
        # contracts actually needed for today (front+back leg per ticker), so it's cheap.
        if run_reason == 'dtest':
            today_actual = datetime.today().date()
            needed_today = option_price_data_request[option_price_data_request['First_Date'] == today_actual]['Options_Ticker'].unique()
            have_today = set(call_option_price_data.loc[
                call_option_price_data['Call_Option_Price_Date'] == today_actual, 'Call_Options_Ticker']) if not call_option_price_data.empty else set()
            missing_today = [t for t in needed_today if t not in have_today]

            if missing_today:
                print(f"Fetching decision-time intraday prices for {len(missing_today)} option contract(s) with no daily bar yet for today...")
                new_rows = []
                for opt_ticker in missing_today:
                    try:
                        opt_intraday_request = client.list_aggs(ticker=opt_ticker, multiplier=5, timespan="minute",
                                                                  from_=today_actual, to=today_actual, limit=1000)
                        opt_intraday = pd.DataFrame(opt_intraday_request)
                        if opt_intraday.empty:
                            print(f"  WARNING: no intraday trades found yet for {opt_ticker} — likely too illiquid "
                                  f"to have printed a trade before the decision time.")
                            continue

                        opt_intraday['bar_dt'] = pd.to_datetime(opt_intraday['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')
                        decision_dt_ny = uk_decision_time_to_ny(today_actual)
                        eligible = opt_intraday[opt_intraday['bar_dt'] <= decision_dt_ny]
                        if eligible.empty:
                            print(f"  WARNING: {opt_ticker} has intraday bars today but none at/before the "
                                  f"decision time yet — try again closer to/after 5pm UK.")
                            continue

                        latest_bar = eligible.loc[eligible['bar_dt'].idxmax()]
                        new_rows.append({'Call_Options_Ticker': opt_ticker, 'Call_Market_Price': latest_bar['close'],
                                          'Call_Option_Price_Date_UNIX': int(latest_bar['timestamp']),
                                          'Call_Option_Price_Date': today_actual})
                    except Exception as e:
                        print(f"  WARNING: intraday fetch failed for {opt_ticker} ({e}).")
                if new_rows:
                    call_option_price_data = pd.concat([call_option_price_data, pd.DataFrame(new_rows)], ignore_index=True)
                    call_option_price_data = call_option_price_data.drop_duplicates(subset=['Call_Options_Ticker', 'Call_Option_Price_Date'])

        Spread = 0.005  # placeholder bid/ask spread assumption — see explanation notes re: refining this
        call_option_price_data['Call_Option_Price_Long'] = round(call_option_price_data['Call_Market_Price'] * (1 + Spread), 3)
        call_option_price_data['Call_Option_Price_Short'] = round(call_option_price_data['Call_Market_Price'] * (1 - Spread), 3)

        ######################################
        ## Joining Call Options Contract prices onto main Call Calendar Spread Tables
        ######################################

        CCS_day1 = pd.merge(CCS_day1, call_option_price_data[['Call_Options_Ticker', 'Call_Option_Price_Date', 'Call_Market_Price', 'Call_Option_Price_Long', 'Call_Option_Price_Short']],
                             how='left', left_on=['Call_1_Option_Ticker', 'Share_Price_Date'], right_on=['Call_Options_Ticker', 'Call_Option_Price_Date'])
        CCS_day1.drop(columns=['Call_Options_Ticker', 'Call_Option_Price_Date'], inplace=True)
        CCS_day1.rename(columns={'Call_Option_Price_Long': 'Call_1_Option_Price_Long', 'Call_Option_Price_Short': 'Call_1_Option_Price_Short', 'Call_Market_Price': 'Call_1_Market_Price'}, inplace=True)

        CCS_day1 = pd.merge(CCS_day1, call_option_price_data[['Call_Options_Ticker', 'Call_Option_Price_Date', 'Call_Market_Price', 'Call_Option_Price_Long', 'Call_Option_Price_Short']],
                             how='left', left_on=['Call_2_Option_Ticker', 'Share_Price_Date'], right_on=['Call_Options_Ticker', 'Call_Option_Price_Date'])
        CCS_day1.drop(columns=['Call_Options_Ticker', 'Call_Option_Price_Date'], inplace=True)
        CCS_day1.rename(columns={'Call_Option_Price_Long': 'Call_2_Option_Price_Long', 'Call_Option_Price_Short': 'Call_2_Option_Price_Short', 'Call_Market_Price': 'Call_2_Market_Price'}, inplace=True)

        CCS_day2 = pd.merge(CCS_day2, call_option_price_data[['Call_Options_Ticker', 'Call_Option_Price_Date', 'Call_Option_Price_Long', 'Call_Option_Price_Short']],
                             how='left', left_on=['Call_1_Option_Ticker', 'Share_Price_Date'], right_on=['Call_Options_Ticker', 'Call_Option_Price_Date'])
        CCS_day2.drop(columns=['Call_Options_Ticker', 'Call_Option_Price_Date'], inplace=True)
        CCS_day2.rename(columns={'Call_Option_Price_Long': 'Call_1_Option_Price_Long', 'Call_Option_Price_Short': 'Call_1_Option_Price_Short'}, inplace=True)

        CCS_day2 = pd.merge(CCS_day2, call_option_price_data[['Call_Options_Ticker', 'Call_Option_Price_Date', 'Call_Option_Price_Long', 'Call_Option_Price_Short']],
                             how='left', left_on=['Call_2_Option_Ticker', 'Share_Price_Date'], right_on=['Call_Options_Ticker', 'Call_Option_Price_Date'])
        CCS_day2.drop(columns=['Call_Options_Ticker', 'Call_Option_Price_Date'], inplace=True)
        CCS_day2.rename(columns={'Call_Option_Price_Long': 'Call_2_Option_Price_Long', 'Call_Option_Price_Short': 'Call_2_Option_Price_Short'}, inplace=True)

        ######################################
        ## Calculating metrics on call option contract prices (IV & Greeks, at ENTRY only)
        ######################################

        print("Calculating implied volatility and greeks for each options contract...")

        call_1_metric_inputs = CCS_day1[['Call_1_Option_Ticker', 'Sim_close', 'Call_strike_price', 'Call_1_Time_to_Expiration', 'Call_1_Market_Price', 'Share_Price_Date']].copy().reset_index(drop=True)
        call_1_metric_inputs = call_1_metric_inputs.dropna(subset=['Call_1_Option_Ticker'])
        call_1_metric_outputs = pd.concat([call_1_metric_inputs[['Share_Price_Date', 'Call_1_Option_Ticker']],
                                            calculate_iv_and_greeks(call_1_metric_inputs['Call_1_Option_Ticker'], call_1_metric_inputs, 'c')], axis=1)
        CCS_day1 = pd.merge(CCS_day1, call_1_metric_outputs, how='left', left_on=['Call_1_Option_Ticker', 'Share_Price_Date'], right_on=['Call_1_Option_Ticker', 'Share_Price_Date'])
        CCS_day1.rename(columns={'implied_volatility': 'Call_1_implied_volatility', 'delta': 'Call_1_delta', 'gamma': 'Call_1_gamma', 'theta': 'Call_1_theta', 'vega': 'Call_1_vega', 'rho': 'Call_1_rho', 'error': 'Call_1_error'}, inplace=True)

        CCS_day1["Call_1_Mean_IV_5"] = CCS_day1["Call_1_implied_volatility"].rolling(window=5).mean()
        CCS_day1["Call_1_IV_momentum_5"] = CCS_day1["Call_1_implied_volatility"] - CCS_day1["Call_1_Mean_IV_5"]
        CCS_day1["Call_1_IV_slope_5"] = (CCS_day1["Call_1_implied_volatility"].rolling(window=5).apply(linear_regression_slope, raw=True))

        call_2_metric_inputs = CCS_day1[['Call_2_Option_Ticker', 'Sim_close', 'Call_strike_price', 'Call_2_Time_to_Expiration', 'Call_2_Market_Price', 'Share_Price_Date']].copy().reset_index(drop=True)
        call_2_metric_inputs = call_2_metric_inputs.dropna(subset=['Call_2_Option_Ticker'])
        call_2_metric_outputs = pd.concat([call_2_metric_inputs[['Share_Price_Date', 'Call_2_Option_Ticker']],
                                            calculate_iv_and_greeks(call_2_metric_inputs['Call_2_Option_Ticker'], call_2_metric_inputs, 'c')], axis=1)
        CCS_day1 = pd.merge(CCS_day1, call_2_metric_outputs, how='left', left_on=['Call_2_Option_Ticker', 'Share_Price_Date'], right_on=['Call_2_Option_Ticker', 'Share_Price_Date'])
        CCS_day1.rename(columns={'implied_volatility': 'Call_2_implied_volatility', 'delta': 'Call_2_delta', 'gamma': 'Call_2_gamma', 'theta': 'Call_2_theta', 'vega': 'Call_2_vega', 'rho': 'Call_2_rho', 'error': 'Call_2_error'}, inplace=True)

        CCS_day1["Call_2_Mean_IV_5"] = CCS_day1["Call_2_implied_volatility"].rolling(window=5).mean()
        CCS_day1["Call_2_IV_momentum_5"] = CCS_day1["Call_2_implied_volatility"] - CCS_day1["Call_2_Mean_IV_5"]
        CCS_day1["Call_2_IV_slope_5"] = (CCS_day1["Call_2_implied_volatility"].rolling(window=5).apply(linear_regression_slope, raw=True))

        CCS_day1['IV_Term_Slope'] = CCS_day1["Call_1_implied_volatility"] - CCS_day1["Call_2_implied_volatility"]
        CCS_day1['IV_Term_Ratio'] = CCS_day1["Call_1_implied_volatility"] / CCS_day1["Call_2_implied_volatility"]
        CCS_day1['IV_Term_Slope_Mean_5'] = CCS_day1['IV_Term_Slope'].rolling(window=5).mean()
        CCS_day1['IV_Term_Slope_STD_5'] = CCS_day1['IV_Term_Slope'].rolling(window=10).std()
        CCS_day1['IV_Term_Slope_Z_Score_5'] = (CCS_day1['IV_Term_Slope'] - CCS_day1['IV_Term_Slope_Mean_5']) / CCS_day1['IV_Term_Slope_STD_5']

        # IV Rank: where does today's front-leg IV sit relative to its own trailing history for this
        # ticker (0 = lowest IV seen in the window, 1 = highest)? This is a standard vol-selling
        # signal — high IV rank means current premium is rich vs. recent norms, which is exactly
        # the kind of context IV_Term_Slope alone doesn't capture (that's front-vs-back on a single
        # day; this is today-vs-history). Window is in ENTRIES, not calendar days — with weekly
        # entries, 26 entries ~= 6 months of trailing history.
        IV_RANK_WINDOW = 26
        CCS_day1['IV_Rank_26'] = CCS_day1['Call_1_implied_volatility'].rolling(
            window=IV_RANK_WINDOW, min_periods=8
        ).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.nan, raw=False)

        CCS_day1['Theta_Ratio'] = np.clip(CCS_day1['Call_1_theta'].abs() / CCS_day1['Call_2_theta'].abs(), 0, 5)
        CCS_day1['Vega_Ratio'] = np.clip(CCS_day1['Call_2_vega'] / CCS_day1['Call_1_vega'], 0, 5)
        CCS_day1['Normalised_Net_Delta'] = (CCS_day1['Call_2_delta'] - CCS_day1['Call_1_delta']) / CCS_day1['Call_2_delta'].abs() + CCS_day1['Call_1_delta'].abs()
        CCS_day1['Gamma_Ratio'] = CCS_day1['Call_1_gamma'] / CCS_day1['Call_2_gamma']
        CCS_day1['Gamma_to_Theta'] = CCS_day1['Gamma_Ratio'].abs() / CCS_day1['Theta_Ratio'].abs()
        CCS_day1['Normalised_Net_Delta_Mean_5'] = CCS_day1['Normalised_Net_Delta'].rolling(window=5).mean()
        CCS_day1['Delta_Flip_Risk'] = (CCS_day1['Normalised_Net_Delta'] - CCS_day1['Normalised_Net_Delta_Mean_5']).abs()

        # New features reflecting the multi-week structure
        CCS_day1['Moneyness'] = (CCS_day1['Call_strike_price'] - CCS_day1['Sim_close']) / CCS_day1['Sim_close']
        CCS_day1['Front_DTE_At_Entry'] = (CCS_day1['Call_1_expiration_date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days)
        CCS_day1['Back_DTE_At_Entry'] = (CCS_day1['Call_2_expiration_date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days)
        CCS_day1['DTE_Gap'] = CCS_day1['Back_DTE_At_Entry'] - CCS_day1['Front_DTE_At_Entry']
        CCS_day1['Days_Held'] = (CCS_day1['Exit_Date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days)
        ######################################
        ## Creating Call Calendar Spread summary file
        ######################################

        print("Calculating Call Calendar Spread strategy profits...")

        CCS_day1 = CCS_day1.reset_index(drop=True)
        CCS_day2 = CCS_day2.reset_index(drop=True)
        stock_price_data = stock_price_data.reset_index(drop=True)

        CCS_summary = CCS_day1[['Share_Price_Date', 'Call_strike_price']].copy()
        CCS_summary['Share_Price_Date'] = pd.to_datetime(CCS_summary['Share_Price_Date'], errors='coerce')
        CCS_summary['Exit_Date'] = pd.to_datetime(CCS_day1['Exit_Date'], errors='coerce')
        # Call_1 = front/short leg, Call_2 = back/long leg (see select_leg_expirations above)
        CCS_summary['Call_1_expiration_date'] = CCS_day1['Call_1_expiration_date']
        CCS_summary['Call_2_expiration_date'] = CCS_day1['Call_2_expiration_date']

        # Entry cashflow: receive premium selling front leg, pay premium buying back leg
        CCS_summary['Day_1_Cashflow'] = round(100 * (CCS_day1['Call_1_Option_Price_Short'] - CCS_day1['Call_2_Option_Price_Long']), 2)
        # Exit cashflow: pay to buy back (close) the front leg, receive premium selling (closing) the back leg
        CCS_summary['Day_2_Cashflow'] = round(100 * (CCS_day2['Call_2_Option_Price_Short'] - CCS_day2['Call_1_Option_Price_Long']), 2)
        CCS_summary['Overall Profit'] = round(CCS_summary['Day_1_Cashflow'] + CCS_summary['Day_2_Cashflow'], 2)
        CCS_summary['Call_1_P&L'] = round(100 * (CCS_day1['Call_1_Option_Price_Short'] - CCS_day2['Call_1_Option_Price_Long']), 2)
        CCS_summary['Call_2_P&L'] = round(100 * (-CCS_day1['Call_2_Option_Price_Long'] + CCS_day2['Call_2_Option_Price_Short']), 2)

        # Merge in Feature_Date-shifted technical indicators to avoid look-ahead bias (see header note):
        # a row's technical indicators come from the PRIOR fully-completed trading day, not the entry
        # day itself (whose high/low/close aren't final at a 5pm-UK decision point).
        # Built directly as a datetime64 series (not .date() objects) so its dtype matches
        # stock_price_data['Share_Price_Date'] below — mixing object-dtype dates with
        # datetime64 dates causes pandas to refuse the merge.
        CCS_summary['Feature_Date'] = CCS_summary['Share_Price_Date'] - us_bday
        stock_price_data['Share_Price_Date'] = pd.to_datetime(stock_price_data['Share_Price_Date'])

        CCS_summary = pd.merge(
            CCS_summary,
            stock_price_data[['Share_Price_Date', 'TR', 'Trend_Efficiency', 'ATR3', 'ATR7', 'ATR14', 'Normalised_ATR',
                               'Vol_Expanding', 'STD10', 'STD3', 'Trend_Distance', 'Trend_Distance_slope_5', 'BBW10',
                               'BBW3', 'ADX_Daily', 'ADX_Daily_slope_5', 'Is_Trending', 'RV_3', 'RV_5', 'RV_14',
                               'Vol_Regime_Ratio', 'daily_vol', 'second_half_vol']],
            how='left', left_on='Feature_Date', right_on='Share_Price_Date', suffixes=('', '_feat'))
        CCS_summary.drop(columns=['Share_Price_Date_feat'], inplace=True, errors='ignore')

        feature_cols = ['Call_1_implied_volatility', 'Call_1_Mean_IV_5', 'Call_1_IV_momentum_5', 'Call_1_IV_slope_5',
                         'Call_2_Mean_IV_5', 'Call_2_IV_momentum_5', 'Call_2_IV_slope_5', 'IV_Term_Slope', 'IV_Term_Ratio',
                         'IV_Term_Slope_Z_Score_5', 'Theta_Ratio', 'Vega_Ratio', 'Normalised_Net_Delta', 'Gamma_Ratio',
                         'Gamma_to_Theta', 'Normalised_Net_Delta_Mean_5', 'Delta_Flip_Risk', 'Moneyness',
                         'Front_DTE_At_Entry', 'Back_DTE_At_Entry', 'DTE_Gap', 'Days_Held', 'IV_Rank_26']
        CCS_summary[feature_cols] = CCS_day1[feature_cols]

        # Realized-vs-implied vol spread: a classic vol-selling signal — positive values mean implied
        # vol is pricing in more movement than has actually been realized recently (favorable for
        # selling premium), negative means the market may be under-pricing realized moves. RV_14 is a
        # daily log-return std, so annualize it (x sqrt(252)) to put it on the same scale as IV.
        CCS_summary['RV14_Annualized'] = CCS_summary['RV_14'] * np.sqrt(252)
        CCS_summary['IV_vs_RV_Spread'] = CCS_summary['Call_1_implied_volatility'] - CCS_summary['RV14_Annualized']

        # --- Days_To_Next_Earnings: via Financial Modeling Prep (FMP) ---
        # Still wrapped defensively — if FMP has no coverage for a ticker (e.g. some ETFs genuinely
        # have no earnings events) or the request fails for any reason, this fills NaN (imputed later,
        # with a global-median fallback) rather than crashing the whole import.
        # The full FMP-based earnings lookup below is disabled for now (no FMP_API_KEY configured).
        # We still need the column to exist though, since it's selected further down when building
        # CCS_summary_df — so fill it with NaN. The "drop columns that are entirely NaN" step later
        # in the script will then quietly remove it before training/imputation.
        CCS_summary['Days_To_Next_Earnings'] = np.nan
        CCS_summary['Days_To_Next_Earnings'] = CCS_summary['Days_To_Next_Earnings'].astype(float)

        # To re-enable the real lookup: set FMP_API_KEY, uncomment the block below, and remove the
        # two lines directly above.
        # if ENABLE_EARNINGS_FEATURE:
        #     try:
        #         fmp_url = f"https://financialmodelingprep.com/stable/earnings?symbol={ticker}&apikey={FMP_API_KEY}"
        #         fmp_response = requests.get(fmp_url, timeout=15)
        #         fmp_response.raise_for_status()
        #         fmp_data = fmp_response.json()
        #
        #         earnings_dates = sorted({
        #             pd.to_datetime(row['date']).date()
        #             for row in fmp_data if isinstance(row, dict) and row.get('date')
        #         })
        #         if not earnings_dates:
        #             raise ValueError("No earnings dates returned by FMP for this ticker — it may genuinely have no "
        #                               "earnings events (common for some ETFs), or FMP's plan tier may not cover it.")
        #
        #         def days_to_next_earnings(entry_date_):
        #             future = [d for d in earnings_dates if d >= entry_date_]
        #             return (min(future) - entry_date_).days if future else np.nan
        #
        #         CCS_summary['Days_To_Next_Earnings'] = CCS_summary['Share_Price_Date'].apply(
        #             lambda d: days_to_next_earnings(d.date()) if pd.notna(d) else np.nan)
        #         CCS_summary['Days_To_Next_Earnings'] = pd.to_numeric(CCS_summary['Days_To_Next_Earnings'], errors='coerce')
        #         print(f"Earnings proximity feature computed for {ticker} from {len(earnings_dates)} earnings dates (FMP).")
        #     except Exception as e:
        #         print(f"WARNING: could not compute Days_To_Next_Earnings for {ticker} ({e}). "
        #               f"Filling with NaN — verify the earnings endpoint for your plan if you need this feature.")
        #         CCS_summary['Days_To_Next_Earnings'] = np.nan
        #         CCS_summary['Days_To_Next_Earnings'] = CCS_summary['Days_To_Next_Earnings'].astype(float)
        # else:
        #     CCS_summary['Days_To_Next_Earnings'] = np.nan
        #     CCS_summary['Days_To_Next_Earnings'] = CCS_summary['Days_To_Next_Earnings'].astype(float)

        CCS_summary['days_since_prev'] = (CCS_summary['Share_Price_Date'] - CCS_summary['Share_Price_Date'].shift(1)).dt.days.fillna(0)
        CCS_summary['Share_Price_Date_Day'] = CCS_summary['Share_Price_Date'].dt.dayofweek

        ######################################
        ## Aggregating Data across tickers
        ######################################

        main_dataframes = [stock_price_data, call_option_contract_data, CCS_day1, CCS_day2, call_option_price_data, CCS_summary, stock_price_intra_day_data_filtered]
        for df_name in main_dataframes:
            df_name.loc[:, "Ticker"] = ticker

        combined_stock_data.append(stock_price_data)
        combined_call_option_data.append(call_option_contract_data)
        combined_call_spread_pt1.append(CCS_day1)
        combined_call_spread_pt2.append(CCS_day2)
        combined_call_option_prices.append(call_option_price_data)
        combined_call_calendar_spread_summary.append(CCS_summary)
        combined_IntraDayStockPrice.append(stock_price_intra_day_data_filtered)

        del stock_price_data, call_option_contract_data, CCS_day1, CCS_day2, call_option_price_data, CCS_summary, stock_price_intra_day_data_filtered



    ######################################
    ## Formatting Dataframes
    ######################################

    print(f"============================")
    print(f"Aggregating datasets to download...")
    print(f"============================")

    CCS_summary_df = pd.concat(combined_call_calendar_spread_summary, ignore_index=True).copy()
    CCS_summary_df = CCS_summary_df[['Ticker', 'Share_Price_Date', 'Exit_Date', 'Overall Profit', 'Call_strike_price',
                                      'Call_1_expiration_date', 'Call_2_expiration_date',
                                      'Day_1_Cashflow', 'Day_2_Cashflow', 'Call_1_P&L', 'Call_2_P&L',
                                      'TR', 'Trend_Efficiency', 'ATR3', 'ATR7', 'ATR14', 'Normalised_ATR', 'Vol_Expanding',
                                      'STD10', 'STD3', 'Trend_Distance', 'Trend_Distance_slope_5', 'BBW10', 'BBW3',
                                      'ADX_Daily', 'ADX_Daily_slope_5', 'Is_Trending', 'RV_3', 'RV_5', 'RV_14',
                                      'Vol_Regime_Ratio', 'daily_vol', 'second_half_vol', 'Call_1_implied_volatility',
                                      'Call_1_Mean_IV_5', 'Call_1_IV_momentum_5', 'Call_1_IV_slope_5', 'Call_2_Mean_IV_5',
                                      'Call_2_IV_momentum_5', 'Call_2_IV_slope_5', 'IV_Term_Slope', 'IV_Term_Ratio',
                                      'IV_Term_Slope_Z_Score_5', 'Theta_Ratio', 'Vega_Ratio', 'Normalised_Net_Delta',
                                      'Gamma_Ratio', 'Gamma_to_Theta', 'Normalised_Net_Delta_Mean_5', 'Delta_Flip_Risk',
                                      'Moneyness', 'Front_DTE_At_Entry', 'Back_DTE_At_Entry', 'DTE_Gap', 'Days_Held',
                                      'IV_Rank_26', 'IV_vs_RV_Spread', 'Days_To_Next_Earnings',
                                      'days_since_prev', 'Share_Price_Date_Day']]

    CCS_summary_df_clean = CCS_summary_df.dropna(subset=['Overall Profit']).copy()
    # NOTE: only rows missing the actual outcome label are dropped here. A row can still have a
    # valid, completed trade (Overall Profit known) while one or two secondary engineered features
    # are NaN (e.g. the IV solver failing to converge on one leg, or a rolling-window feature still
    # warming up in the first ~2-3 weeks of a ticker's history) — those used to cause the WHOLE row
    # to be discarded under a blanket dropna(), which was needlessly shrinking the dataset. We now
    # median-impute those secondary feature gaps instead of throwing the row away.
    non_feature_cols = ['Ticker', 'Share_Price_Date', 'Exit_Date', 'Overall Profit', 'Call_strike_price',
                         'Day_1_Cashflow', 'Day_2_Cashflow', 'Call_1_P&L', 'Call_2_P&L', Call_1_expiration_date, Call_2_expiration_date]
    feature_cols_to_impute = [c for c in CCS_summary_df_clean.columns if c not in non_feature_cols]

    # Drop any feature column that is ENTIRELY NaN across the whole dataset (e.g. Days_To_Next_Earnings
    # when the earnings endpoint isn't available on your plan) BEFORE imputing. A column like this
    # can't be median-imputed (the median of an all-NaN column is itself NaN), and leaving it in would
    # mean the later safety-net dropna() sees every single row as still having a NaN — silently wiping
    # out the ENTIRE training set. Dropping the unusable column instead of the rows is the correct fix.
    fully_empty_cols = [c for c in feature_cols_to_impute if CCS_summary_df_clean[c].notna().sum() == 0]
    if fully_empty_cols:
        print(f"WARNING: dropping columns with NO usable data across the whole dataset (can't be "
              f"imputed, and would otherwise wipe out every row via the safety-net dropna): {fully_empty_cols}")
        CCS_summary_df_clean = CCS_summary_df_clean.drop(columns=fully_empty_cols)
        feature_cols_to_impute = [c for c in feature_cols_to_impute if c not in fully_empty_cols]

    na_before = CCS_summary_df_clean[feature_cols_to_impute].isna().sum().sum()
    # Impute per-ticker so a gap in one ticker's data isn't filled with another ticker's typical values.
    CCS_summary_df_clean[feature_cols_to_impute] = CCS_summary_df_clean.groupby('Ticker')[feature_cols_to_impute].transform(lambda s: s.fillna(s.median()))

    # Second pass: fall back to the GLOBAL (all-ticker) median for anything the per-ticker pass
    # couldn't fill — this happens when a ticker has ZERO valid values for a column (e.g. an
    # earnings-date feature that only some tickers have coverage for), where the per-ticker median
    # is itself NaN. Without this fallback, those rows would silently be wiped out by the safety-net
    # dropna() below even though the row's actual outcome label and other features are perfectly
    # valid — that's exactly what caused row counts to unexpectedly drop in an earlier run.
    remaining_na_cols = [c for c in feature_cols_to_impute if CCS_summary_df_clean[c].isna().any()]
    if remaining_na_cols:
        global_medians = CCS_summary_df_clean[remaining_na_cols].median()
        CCS_summary_df_clean[remaining_na_cols] = CCS_summary_df_clean[remaining_na_cols].fillna(global_medians)
        print(f"Applied global-median fallback (ticker had no valid data for this feature at all) to: {remaining_na_cols}")

    na_after = CCS_summary_df_clean[feature_cols_to_impute].isna().sum().sum()
    print(f"Imputed {na_before - na_after} missing feature values with per-ticker medians (rows kept as long as Overall Profit was known).")
    CCS_summary_df_clean = CCS_summary_df_clean.dropna().copy()  # catches any column that's entirely NaN dataset-wide (shouldn't happen — fully_empty_cols above already handles that case)
    print(f'Number of rows removed for training/testing datasets: {len(CCS_summary_df) - len(CCS_summary_df_clean)} (ignore for dtest)')
    print(f'Training Split: {(1 - test_split) * 100}% \n Testing Split: {test_split * 100}%')

    if run_reason == 'dtest':
        print(f'Script is being run for daily testing so no training dataset will be output')
        # FIXED: previously recomputed "today" independently here with an arbitrary -4 hours offset
        # (dtest_date = (datetime.today() - timedelta(hours=4)).date()), which could silently mismatch
        # the actual date used to build entry_dates earlier in the ticker loop — if they disagreed even
        # by one day, this filter would match ZERO rows for every ticker, producing an empty (headers-
        # only) testing CSV with no error. Now reuses the EXACT date the entries were actually built on.
        dtest_date = dtest_actual_date
        testing_dataset = CCS_summary_df[CCS_summary_df['Share_Price_Date'].dt.date == dtest_date].copy()

        if testing_dataset.empty:
            print(f"WARNING: no ticker produced a valid row for {dtest_date}. Most common reasons: "
                  f"(a) run before the 5pm UK decision time, so today's price bar doesn't exist yet, "
                  f"(b) today isn't a trading day (weekend/holiday), or (c) no suitable Friday "
                  f"expiration was found for any ticker. Check the 'Dropped entry row' warnings above.")
            # Fall back to the most recent PRIOR date that has at least one valid row, rather than
            # returning an empty file. This won't be a genuine "right now" signal — it reflects
            # whatever the last fully-built row was — so it's clearly flagged below and in the
            # output filename/column, not silently substituted.
            available_dates = sorted(CCS_summary_df['Share_Price_Date'].dt.date.dropna().unique())
            available_dates = [d for d in available_dates if d <= dtest_date]
            if available_dates:
                fallback_date = available_dates[-1]
                print(f"Falling back to the most recent available date instead: {fallback_date} "
                      f"(NOT today — treat this as a stale/reference signal, not a live one).")
                testing_dataset = CCS_summary_df[CCS_summary_df['Share_Price_Date'].dt.date == fallback_date].copy()
                testing_dataset['Is_Fallback_Date'] = True
                dtest_date = fallback_date
            else:
                print("No prior date has any valid rows either within the fetched window — nothing to fall back to.")

    elif run_reason == 'test':
        print(f'Script is being run for testing so no training dataset will be output')
        testing_dataset = CCS_summary_df_clean

    elif run_reason == "train":
        # CHRONOLOGICAL split, not random shuffle: with ~3-week overlapping holding periods,
        # a random split would place highly-correlated, overlapping trades on both sides of
        # the train/test boundary, leaking information and inflating reported test performance.
        # The earliest (1 - test_split) fraction of dates becomes training data; the most
        # recent test_split fraction is held out as a genuine forward-looking test set.
        CCS_summary_df_clean_sorted = CCS_summary_df_clean.sort_values(by='Share_Price_Date').reset_index(drop=True)
        split_idx = int((1 - test_split) * len(CCS_summary_df_clean_sorted))

        training_dataset = CCS_summary_df_clean_sorted.iloc[:split_idx].copy()
        testing_dataset = CCS_summary_df_clean_sorted.iloc[split_idx:].copy()

        if shuffle_training_dataset:
            # Shuffling ONLY the training rows (after the chronological cut) is fine — it just
            # affects the order XGBoost sees rows in, not which rows end up in train vs test.
            training_dataset = training_dataset.sample(frac=1, random_state=train_test_random_seed).reset_index(drop=True)

    else:
        raise ValueError(f"Not fully prepared for downloads for the following run reason: {run_reason}")

    ######################################
    ## Saving Files
    ######################################

    # output_dir lets the caller control where files land:
    #   - left as None: falls back to a folder next to this script (old local-only behaviour)
    #   - passed in (e.g. by the GitHub Actions runner as "outputs"): files land inside the repo
    #     checkout so a later workflow step can commit them back to GitHub as real files.
    base_directory = output_dir if output_dir else os.path.dirname(os.path.abspath(__file__))

    datetime_str = datetime.now().strftime("%Y-%m-%d__%H.%M")

    outer_folder_name = f"{datetime_str} {run_reason} Model Run"
    inner_folder_name = f"{run_reason} Import Outputs"

    full_downloads_path = os.path.join(base_directory, prediction_model_folder, outer_folder_name, inner_folder_name)
    os.makedirs(full_downloads_path, exist_ok=True)


    def ts():
        return datetime.today().strftime("%Y-%m-%d__%H.%M")


    stock_price_data_all = pd.concat(combined_stock_data, ignore_index=True)
    call_option_data_all = pd.concat(combined_call_option_data, ignore_index=True)
    call_spread_pt1_all = pd.concat(combined_call_spread_pt1, ignore_index=True)
    call_spread_pt2_all = pd.concat(combined_call_spread_pt2, ignore_index=True)
    call_option_prices_all = pd.concat(combined_call_option_prices, ignore_index=True)
    intraday_stock_price_all = pd.concat(combined_IntraDayStockPrice, ignore_index=True)

    stock_price_data_all.to_csv(os.path.join(full_downloads_path, f"Stock_Price_Data_{ts()}.csv"), index=False)
    call_option_data_all.to_csv(os.path.join(full_downloads_path, f"Call_Options_Contract_Data_{ts()}.csv"), index=False)
    call_spread_pt1_all.to_csv(os.path.join(full_downloads_path, f"4.CCS_Entry_Data_{ts()}.csv"), index=False)
    call_spread_pt2_all.to_csv(os.path.join(full_downloads_path, f"5.CCS_Exit_Data_{ts()}.csv"), index=False)
    call_option_prices_all.to_csv(os.path.join(full_downloads_path, f"Call_Option_Prices_{ts()}.csv"), index=False)
    CCS_summary_df.to_csv(os.path.join(full_downloads_path, f"3.CCS_Summary_Data_{ts()}.csv"), index=False)
    intraday_stock_price_all.to_csv(os.path.join(full_downloads_path, f"Intraday_Stock_Price_Data_{ts()}.csv"), index=False)

    if test_split != 1:
        training_dataset["Tick"] = training_dataset["Ticker"].map(ticker_map)
        training_dataset.to_csv(os.path.join(full_downloads_path, f"2.CCS_Training_Data_{ts()}.csv"), index=False)
    else:
        training_dataset = None

    if 'Is_Fallback_Date' not in testing_dataset.columns:
        testing_dataset['Is_Fallback_Date'] = False
    testing_dataset["Tick"] = testing_dataset["Ticker"].map(ticker_map)
    testing_dataset.to_csv(os.path.join(full_downloads_path, f"1.CCS_Testing_Data_{ts()}.csv"), index=False)

    print("Done.")

    # ------------------------------------------------------------------
    # Return everything as function outputs so other code (e.g. a
    # training script, or another step in the same workflow) can use
    # the data directly, without re-reading the CSVs from disk.
    # ------------------------------------------------------------------
    return {
        "output_dir": full_downloads_path,
        "stock_price_data": stock_price_data_all,
        "call_option_data": call_option_data_all,
        "call_spread_entry": call_spread_pt1_all,
        "call_spread_exit": call_spread_pt2_all,
        "call_option_prices": call_option_prices_all,
        "intraday_stock_price": intraday_stock_price_all,
        "ccs_summary": CCS_summary_df,
        "training_dataset": training_dataset,
        "testing_dataset": testing_dataset,
    }
from py_vollib.black_scholes import implied_volatility
from py_vollib.black_scholes.greeks import analytical
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar


def func_import(
        polygon_api_key: str = None,
        shuffle_training_dataset = True,
        train_test_random_seed = 1,
        run_reason :str = None,
        default_settings :bool = None,
        sample_size :int = None,
        ticker_list_str :str = None,
        test_split :float = None,
        training_dataset = None,
        output_dir: str = None  # if None, falls back to a folder next to this script.
                                 # Pass a path (e.g. "outputs") to control where CSVs land -
                                 # used by the GitHub Actions runner to save inside the repo checkout.







        ):
    ###############################################################################
    # -------------------------- USER CONFIGURATION -------------------------------
    ###############################################################################


    # API key from config
    client = RESTClient(polygon_api_key, retries = 5)
    #API key kept in secrets

    prediction_model_folder = "Model v7"

    ticker_map = {
        "SPY": 1,
        "IWM": 2,
        "QQQ": 3,
        "DIA": 4,
        "XLF": 5,
        "GLD": 6,
        "XLE": 7,
        "XLV": 8,
        "XLI": 9,
        "TLT": 10,
        "HYG": 11,
        "EEM": 12,
        "XLU": 13,
        "XLY": 14,
        "XLB": 15,
        "XLP": 16,
        "FXI": 17,
        "SLV": 18,
    }

    ###############################################################################
    # ------------------------ STRATEGY TIMING CONFIGURATION ----------------------
    ###############################################################################

    front_leg_target_days = 18   # calendar days from entry to target front (short) leg expiry
    back_leg_target_days = 35    # calendar days from entry to target back (long) leg expiry
    exit_business_days_before_front_expiry = 1  # close position this many trading days before front expiry
    option_expiry_search_buffer_days = 21  # extra buffer beyond back-leg target when searching for expirations
    strike_search_window_pct = 0.015  # +/- window around price range when pulling contract strikes


    decision_time_local = time(17, 0)
    decision_tz = ZoneInfo("Europe/London")
    ny_tz = ZoneInfo("America/New_York")

    randomise_entry_weekday= True  # spreads training/testing entry dates across different weekdays
    trade_frequency_days = 7  # calendar days between new entries per ticker (weekly cadence chosen:
                               # ~3 overlapping ~3-week positions per ticker at any given time). Change
                               # this in one place if later want a different cadence — everything
                               # else (dtest/live flow, training data generation) reads from it.

    historical_data_limit_years = 15  # set this to match how far back API plan actually goes
                                   # build_entry_dates() clips to this automatically, so an
                                   # oversized sample_size degrades gracefully (fewer rows, with a
                                   # printed warning) instead of the API erroring on out-of-range dates.

    ###############################################################################
    # -------------------------- FUNCTION (DEFAULT) SETTINGS ------------------------
    ###############################################################################

    samplesize_test_default = 30
    samplesize_train_default = 100  # ~1 entry/week x ~5 years of history (adjust to match actual
                                 # API data-access window — see HISTORICAL_DATA_LIMIT_YEARS above)
    ticker_list_default = ['SPY', 'QQQ', 'IWM', 'DIA', 'XLF', 'GLD', 'XLE', 'XLV', 'XLI', 'TLT', 'HYG', 'EEM',
                        'XLU', 'XLY', 'XLB', 'XLP', 'FXI', 'SLV']

    run_reason_options = ["train", "test", "dtest"]

    test_split_default = 0.20

    combined_stock_data = []
    combined_call_option_data = []
    combined_call_spread_pt1 = []
    combined_call_spread_pt2 = []
    combined_call_option_prices = []
    combined_call_calendar_spread_summary = []
    combined_IntraDayStockPrice = []


    us_bday = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    ######################################
    ## Function Inputs and Assigning Variables
    ######################################

    if run_reason == "train":
        if default_settings:
            sample_size = samplesize_train_default
            ticker_list = ticker_list_default
            test_split = test_split_default
        else:
            if not (50 <= sample_size <= 2000):
                ValueError(f"When training the sample size must be between 50 and 2000")
            ticker_list = [item.strip().upper() for item in ticker_list_str.split(",")]
            if not (0 <= test_split <= 0.99):
                ValueError(f"When training the test split must be between 0 and 0.99")


    elif run_reason in ("test", "dtest"):
        if default_settings:
            sample_size = samplesize_test_default
            ticker_list = ticker_list_default
            test_split = 1
        else:
            if not (25 <= sample_size <= 200):
                ValueError(f"When testing the sample size must be between 25 and 200")
            ticker_list = [item.strip().upper() for item in ticker_list_str.split(",")]
            test_split = 1
    else:
        raise ValueError(f"Import function requires one of the following as run_reason: {run_reason_options}")


    ######################################
    ## Helper Functions
    ######################################

    def func_left_join_on_closest(left_df, right_df, left_on, right_on, prefix):
        def find_closest(val):
            if pd.isna(val):
                return pd.NaT
            diffs = (right_df[right_on] - val).abs()
            if diffs.isna().all():
                return pd.NaT
            idx = diffs.idxmin()
            return right_df.loc[idx, right_on]

        matched = pd.Series(
            [find_closest(val) for val in left_df[left_on]],
            name=prefix + right_on
        )
        return pd.concat([left_df.reset_index(drop=True), matched], axis=1)


    def uk_decision_time_to_ny(date_):
        """
        Converts 5pm UK time on a given calendar date into the equivalent US Eastern
        datetime, automatically handling the (slightly offset) UK/US daylight saving
        transitions. This is the cutoff we use for 'what was actually knowable' when
        building decision-time price features.
        """
        uk_dt = datetime.combine(date_, decision_time_local, tzinfo=decision_tz)
        return uk_dt.astimezone(ny_tz)


    def func_left_join_on_decision_price(left_df, right_df, left_on, right_on):
        """
        Left join on calendar date. For each date, finds the LATEST intraday bar at or
        before the NY-time equivalent of 5pm UK on that date (i.e. only data that would
        genuinely have existed at your real decision point). If no bar exists yet at/before
        that time (e.g. a data gap), returns NaN rather than silently using a later bar.
        """
        right_dt = pd.to_datetime(right_df[right_on])
        right_tmp = right_df.copy()
        right_tmp["_date"] = right_dt.dt.date
        right_tmp["_dt"] = right_dt
        grouped = right_tmp.groupby("_date")

        def find_values_for_date(val):
            if pd.isna(val):
                return pd.NaT, np.nan
            if val not in grouped.groups:
                return pd.NaT, np.nan
            day_df = grouped.get_group(val)
            decision_dt_ny = uk_decision_time_to_ny(val)
            eligible = day_df[day_df["_dt"] <= decision_dt_ny]
            if eligible.empty:
                return pd.NaT, np.nan
            idx = eligible["_dt"].idxmax()
            row = right_tmp.loc[idx]
            return row[right_on], row["close"]

        left_dates = left_df[left_on]
        if not np.issubdtype(pd.Series(left_dates).dtype, np.datetime64):
            left_dates = pd.to_datetime(left_dates).dt.date
        else:
            left_dates = left_dates.dt.date

        matched_values = [find_values_for_date(val) for val in left_dates]
        matched = pd.DataFrame(matched_values, columns=[right_on + "_decision", "close_decision"])

        return pd.concat([left_df.reset_index(drop=True), matched], axis=1)


    ######################################
    ## Functions for calculating metrics
    ######################################

    def calculate_iv_and_greeks(tickers, option_data, option_type, r=0.0432):
        results = []
        for ticker in range(len(tickers)):
            data = option_data.iloc[ticker]
            S = data.iloc[1]
            K = data.iloc[2]
            T = data.iloc[3]
            flag = option_type
            option_price = data.iloc[4]

            try:
                iv = implied_volatility.implied_volatility(option_price, S, K, T, r, flag)
                delta = analytical.delta(flag, S, K, T, r, iv)
                gamma = analytical.gamma(flag, S, K, T, r, iv)
                theta = analytical.theta(flag, S, K, T, r, iv)
                vega = analytical.vega(flag, S, K, T, r, iv)
                rho = analytical.rho(flag, S, K, T, r, iv)
                results.append({"implied_volatility": iv, "delta": delta, "gamma": gamma,
                                 "theta": theta, "vega": vega, "rho": rho})
            except Exception as e:
                results.append({"implied_volatility": None, "delta": None, "gamma": None,
                                 "theta": None, "vega": None, "rho": None, "error": str(e)})
        return pd.DataFrame(results)


    def compute_realized_volatility(df_intraday):
        results = []
        for date_, group in df_intraday.groupby('Share_Price_Date'):
            group = group.sort_values(by='Share_Price_Datetime')
            log_returns = np.log(group['close']).diff().dropna()
            daily_vol = 100 * (np.sqrt(np.sum(log_returns ** 2)))

            mid_time = pd.Timestamp(f"{date_} 12:45:00", tz="America/New_York")
            second_half = group[group['Share_Price_Datetime'] >= mid_time]
            log_returns_second_half = np.log(second_half['close']).diff().dropna()
            second_half_vol = 100 * (np.sqrt(np.sum(log_returns_second_half ** 2))) if len(log_returns_second_half) > 0 else np.nan

            results.append({'daily_vol': daily_vol, 'second_half_vol': second_half_vol, 'date': date_})
        return pd.DataFrame(results)


    def linear_regression_slope(y):
        x = np.arange(len(y))
        m, _ = np.polyfit(x, y, 1)
        return m


    def compute_adx(df, period=14):
        high = df['high']
        low = df['low']
        close = df['previous_close']

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        return adx, plus_di, minus_di


    def select_leg_expirations(entry_date_, available_expirations, front_target_days, back_target_days):
        """
        Given an entry date and the sorted list of available expiration dates for a
        (ticker, strike), returns (front_expiration, back_expiration) chosen as the
        NEAREST available expiration at/after the target day count. Returns (NaT, NaT)
        or (front, NaT) if a suitable leg can't be found (e.g. not enough listed expiries).
        """
        if not available_expirations:
            return pd.NaT, pd.NaT

        front_candidates = [d for d in available_expirations if d >= entry_date_ + timedelta(days=front_target_days)]
        if not front_candidates:
            return pd.NaT, pd.NaT
        front_expiration = min(front_candidates)

        back_candidates = [d for d in available_expirations
                            if d >= entry_date_ + timedelta(days=back_target_days) and d > front_expiration]
        if not back_candidates:
            return front_expiration, pd.NaT
        back_expiration = min(back_candidates)

        return front_expiration, back_expiration


    def build_entry_dates(sample_size_, mode, seed):
        """
        train/test: builds `sample_size_` entry dates spaced TRADE_FREQUENCY_DAYS apart
        (calendar days), with the weekday randomised within each window (mirrors real
        usage: open a new position on whatever day suits you, roughly every
        TRADE_FREQUENCY_DAYS), which also reduces overlap between adjacent training
        trades' holding periods relative to sampling every single business day.

        dtest: builds a run of consecutive recent business days ending today, so that
        "today" (whatever day you happen to run the script) is guaranteed to be included;
        the calling code later filters this down to just today's row. Live trading isn't
        restricted to a fixed weekday — you run this whenever suits you, roughly every
        TRADE_FREQUENCY_DAYS (see the accompanying explanation for exact guidance).
        """
        today_ = datetime.today().date()

        if mode == "dtest":
            days = pd.date_range(end=today_, periods=sample_size_, freq=us_bday)
            return [d.date() for d in days]

        # Generous pool of business days to draw samples from
        pool = pd.date_range(end=today_, periods=sample_size_ * trade_frequency_days * 2, freq=us_bday)
        pool_df = pd.DataFrame({'d': pool})
        # Bucket by calendar days-since-epoch // TRADE_FREQUENCY_DAYS, rather than a fixed
        # ISO week — this generalises cleanly to any cadence (weekly, fortnightly, etc.),
        # not just exactly-7-day windows.
        epoch_days = (pool_df['d'] - pd.Timestamp("1970-01-01")).dt.days
        pool_df['period_key'] = epoch_days // trade_frequency_days

        if randomise_entry_weekday:
            rng = np.random.default_rng(seed)
            chosen = pool_df.groupby('period_key')['d'].apply(lambda s: rng.choice(s.values))
        else:
            chosen = pool_df.groupby('period_key')['d'].first()

        chosen = sorted(pd.to_datetime(chosen.values))
        chosen = chosen[-sample_size_:]

        earliest_allowed = pd.Timestamp(today_) - pd.Timedelta(days=365 * historical_data_limit_years)
        within_range = [d for d in chosen if d >= earliest_allowed]
        if len(within_range) < len(chosen):
            print(f"WARNING: requested sample_size={sample_size_} needed dates back to "
                  f"{chosen[0].date()}, which exceeds your configured {historical_data_limit_years}-year "
                  f"data access window (earliest allowed: {earliest_allowed.date()}). "
                  f"Dropped {len(chosen) - len(within_range)} entry date(s) rather than risk an API error. "
                  f"Consider lowering sample_size_train_default if you see this often.")

        return [pd.Timestamp(d).date() for d in within_range]


    ######################################
    ## Full Data Import Process
    ######################################

    for ticker in ticker_list:
        print(f"============================")
        print(f"Processing ticker: {ticker}")
        print(f"============================")

        ######################################
        ## Getting list of entry dates for sample
        ######################################

        print("Getting list of sample entry dates...")
        entry_dates = build_entry_dates(sample_size, run_reason, train_test_random_seed)
        if run_reason == 'dtest':
            dtest_actual_date = entry_dates[-1]  # capture the exact date used to build entries, so the
                                                  # final filter below matches it exactly — see fix notes
                                                  # near the dtest filtering section further down
        fetch_start = entry_dates[0]
        fetch_end = entry_dates[-1] + timedelta(days=back_leg_target_days + option_expiry_search_buffer_days)

        ######################################
        # Getting stock price data (covers entry AND potential exit dates)
        ######################################

        print("Getting stock price data...")
        stock_price_data_request = client.get_aggs(ticker=ticker, multiplier=1, timespan='day',
                                                    from_=fetch_start, to=min(fetch_end, datetime.today().date()))
        today = datetime.today()
        today_stock_price_data_request = client.get_snapshot_ticker(ticker=ticker, market_type='stocks')

        stock_price_data = pd.DataFrame(stock_price_data_request)
        stock_price_data['Share_Price_Date'] = pd.to_datetime(stock_price_data['timestamp'], unit='ms').dt.date

        if today.date() not in stock_price_data['Share_Price_Date'].values:
            stock_price_data.loc[len(stock_price_data)] = [
                today_stock_price_data_request.day.open, today_stock_price_data_request.day.high,
                today_stock_price_data_request.day.low, today_stock_price_data_request.day.close,
                today_stock_price_data_request.day.volume, today_stock_price_data_request.day.vwap,
                "", "", "", today.date()
            ]

        ######################################
        ## Getting intraday stock price data
        ######################################

        print("Getting intraday stock prices...")
        stock_price_intra_day_data_request = client.list_aggs(ticker=ticker, multiplier=5, timespan="minute",
                                                                from_=fetch_start, to=min(fetch_end, datetime.today().date()),
                                                                limit=1000000)

        stock_price_intra_day_data = pd.DataFrame(stock_price_intra_day_data_request)
        stock_price_intra_day_data['Share_Price_Datetime'] = pd.to_datetime(
            stock_price_intra_day_data['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')
        stock_price_intra_day_data['Share_Price_Date'] = stock_price_intra_day_data['Share_Price_Datetime'].dt.date

        # Keep pre-market + regular hours; decision time (~noon-ish ET for a 5pm UK check) sits
        # well within this window, but we keep the same upper filter as before for safety.
        stock_price_intra_day_data_filtered = stock_price_intra_day_data[
            stock_price_intra_day_data['Share_Price_Datetime'].dt.hour < 16].copy()

        ######################################
        ## Adding decision-time price to stock price data table
        ######################################

        stock_price_data = func_left_join_on_decision_price(
            left_df=stock_price_data, right_df=stock_price_intra_day_data_filtered,
            left_on='Share_Price_Date', right_on='Share_Price_Datetime')
        stock_price_data.rename(columns={'Share_Price_Datetime_decision': 'Decision_Datetime',
                                          'close_decision': 'Decision_Price'}, inplace=True)

        ######################################
        ## Calculating metrics for stock price data
        ## NOTE: these use each day's own high/low/close, which are only fully known AFTER that
        ## day's market close. They are valid to use as "yesterday's fully-known indicators" via
        ## the Feature_Date shift applied later — do not treat a row's own indicators as knowable
        ## on that same calendar date.
        ######################################

        print("Calculating stock metrics...")

        stock_price_data['previous_close'] = stock_price_data['close'].shift(1)

        stock_price_data['TR'] = stock_price_data[['high', 'low', 'previous_close']].apply(
            lambda row: round(max(
                row['high'] - row['low'],
                abs(row['high'] - row['previous_close']),
                abs(row['low'] - row['previous_close'])
            ), 4), axis=1)
        stock_price_data['Trend_Efficiency'] = (stock_price_data['close'] - stock_price_data['open']).abs() / stock_price_data['TR']
        stock_price_data['ATR3'] = stock_price_data['TR'].rolling(window=3).mean()
        stock_price_data['ATR7'] = stock_price_data['TR'].rolling(window=7).mean()
        stock_price_data['ATR14'] = stock_price_data['TR'].rolling(window=14).mean()

        stock_price_data['Normalised_ATR'] = stock_price_data['ATR3'] / stock_price_data['open']
        stock_price_data['Vol_Expanding'] = (stock_price_data['ATR3'] / stock_price_data['ATR14'] > 1.2).astype(int)

        stock_price_data['MA10'] = stock_price_data['close'].rolling(window=10).mean()
        stock_price_data['MA3'] = stock_price_data['close'].rolling(window=3).mean()
        stock_price_data['STD10'] = stock_price_data['close'].rolling(window=10).std()
        stock_price_data['STD3'] = stock_price_data['close'].rolling(window=3).std()

        stock_price_data['Trend_Distance'] = (stock_price_data['close'] - stock_price_data['MA3']) / stock_price_data['ATR7']
        stock_price_data['Trend_Distance_slope_5'] = (
            stock_price_data["Trend_Distance"].rolling(window=5).apply(linear_regression_slope, raw=True))

        stock_price_data['UpperBand10'] = stock_price_data['MA10'] + (2 * stock_price_data['STD10'])
        stock_price_data['LowerBand10'] = stock_price_data['MA10'] - (2 * stock_price_data['STD10'])
        stock_price_data['UpperBand3'] = stock_price_data['MA3'] + (2 * stock_price_data['STD3'])
        stock_price_data['LowerBand3'] = stock_price_data['MA3'] - (2 * stock_price_data['STD3'])

        stock_price_data['BBW10'] = (stock_price_data['UpperBand10'] - stock_price_data['LowerBand10']) / stock_price_data['MA10']
        stock_price_data['BBW3'] = (stock_price_data['UpperBand3'] - stock_price_data['LowerBand3']) / stock_price_data['MA3']
        stock_price_data = stock_price_data.drop(columns=['UpperBand10', 'LowerBand10', 'UpperBand3', 'LowerBand3'])

        stock_price_data['ADX_Daily'], stock_price_data['Plus_DI_Daily'], stock_price_data['Minus_DI_Daily'] = compute_adx(stock_price_data, period=14)
        stock_price_data['ADX_Daily_slope_5'] = (
            stock_price_data['ADX_Daily'].rolling(window=5).apply(linear_regression_slope, raw=True))

        stock_price_data['Is_Trending'] = (stock_price_data['ADX_Daily'] > 25).astype(int)

        stock_price_data['log_return'] = np.log(stock_price_data['close'] / stock_price_data['close'].shift(1))
        stock_price_data['RV_3'] = stock_price_data['log_return'].rolling(window=3).std().reset_index(level=0, drop=True)
        stock_price_data['RV_5'] = stock_price_data['log_return'].rolling(window=5).std().reset_index(level=0, drop=True)
        stock_price_data['RV_14'] = stock_price_data['log_return'].rolling(window=14).std().reset_index(level=0, drop=True)
        stock_price_data = stock_price_data.drop(columns=['log_return'])
        stock_price_data['Vol_Regime_Ratio'] = stock_price_data['RV_3'] / stock_price_data['RV_14']

        ######################################
        ## Calculating metrics on intraday stock price data
        ######################################

        print("Calculating metrics on intraday stock prices...")
        realised_volatility = compute_realized_volatility(stock_price_intra_day_data_filtered)
        realised_volatility['date'] = pd.to_datetime(realised_volatility["date"]).dt.date
        stock_price_data = pd.merge(stock_price_data, realised_volatility, how='left', left_on='Share_Price_Date', right_on='date')
        stock_price_data.drop(columns=['date'], inplace=True)

        ######################################
        ## Setting up 'Call Calendar Spread' Strategy — ENTRY leg
        ######################################

        print("Setting up Call Calendar Spread strategy (entry)...")

        CCS_day1 = pd.DataFrame(entry_dates, columns=['Share_Price_Date'])
        CCS_day1 = pd.merge(CCS_day1, stock_price_data[['Share_Price_Date', 'Decision_Price']],
                             how='left', left_on='Share_Price_Date', right_on='Share_Price_Date')
        CCS_day1.rename(columns={'Decision_Price': 'Sim_close'}, inplace=True)  # keep 'Sim_close' name downstream for compatibility
        CCS_day1.insert(1, 'Stock_Ticker', ticker)

        drop_idx_no_price = CCS_day1[CCS_day1['Sim_close'].isna()].index
        for idx in drop_idx_no_price:
            print(f"Dropped entry row | index={idx} | date={CCS_day1.loc[idx, 'Share_Price_Date']} (no decision-time price)")
        CCS_day1 = CCS_day1.drop(index=drop_idx_no_price).reset_index(drop=True)

        CCS_day1_len = len(CCS_day1)  # single strike per date now — no x3 multiplier

        ######################################
        ## Getting options contract data (covers full front+back leg search window)
        ######################################

        print("Getting options price contract data...")

        call_option_contract_data_request = []
        for c in client.list_options_contracts(underlying_ticker=ticker, contract_type='call',
                                                expiration_date_gte=CCS_day1['Share_Price_Date'].iloc[0],
                                                expiration_date_lte=fetch_end,
                                                strike_price_gte=stock_price_data['Decision_Price'].min() * (1 - strike_search_window_pct),
                                                strike_price_lte=stock_price_data['Decision_Price'].max() * (1 + strike_search_window_pct),
                                                expired=True, limit=1000):
            call_option_contract_data_request.append(c)

        for c in client.list_options_contracts(underlying_ticker=ticker, contract_type='call',
                                                expiration_date_gte=CCS_day1['Share_Price_Date'].iloc[0],
                                                expiration_date_lte=fetch_end,
                                                strike_price_gte=stock_price_data['Decision_Price'].min() * (1 - strike_search_window_pct),
                                                strike_price_lte=stock_price_data['Decision_Price'].max() * (1 + strike_search_window_pct),
                                                expired=False, limit=1000):
            call_option_contract_data_request.append(c)

        call_option_contract_data = pd.DataFrame(call_option_contract_data_request)
        call_option_contract_data['expiration_date'] = pd.to_datetime(call_option_contract_data['expiration_date']).dt.date

        call_option_contract_data = call_option_contract_data[call_option_contract_data['strike_price'] % 1 == 0]
        call_option_contract_data = call_option_contract_data[call_option_contract_data['shares_per_contract'] == 100]

        exchange_priority = ["BATO", "C2OX", "XBOX", "XBSE"]
        priority_map = {exch: i for i, exch in enumerate(exchange_priority)}
        call_option_contract_data['exchange_priority'] = call_option_contract_data['primary_exchange'].map(priority_map)
        call_option_contract_data['exchange_priority'] = call_option_contract_data['exchange_priority'].fillna(len(exchange_priority))
        call_option_contract_data = call_option_contract_data.sort_values('exchange_priority').drop_duplicates(subset='ticker', keep='first')
        # Guard against duplicate (expiration_date, strike_price) pairs — these can arise from
        # corporate-action-adjusted contract series (e.g. special dividends/spin-offs) that carry
        # a different ticker but the same nominal strike/expiration as the standard contract.
        # Since downstream merges key on (expiration_date, strike_price), any leftover collision
        # here would silently duplicate rows in CCS_day1.
        call_option_contract_data = call_option_contract_data.drop_duplicates(subset=['expiration_date', 'strike_price'], keep='first')

        ######################################
        ## Selecting ATM strike, then front/back leg expirations, then exit date
        ######################################

        print("Selecting strike and front/back leg expirations...")

        CCS_day1 = func_left_join_on_closest(left_df=CCS_day1, right_df=call_option_contract_data,
                                              left_on='Sim_close', right_on='strike_price', prefix='Call_')
        CCS_day1['Call_strike_price'] = pd.to_numeric(CCS_day1['Call_strike_price'], errors='coerce').astype('float64')

        if len(CCS_day1) != CCS_day1_len:
            raise ValueError("CCS_day1 gained rows when joining on strike price")

        # Distinct available expirations, grouped by strike, for fast per-row lookup.
        # RESTRICTED TO FRIDAYS: tickers like SPY/QQQ/IWM/GLD list options expiring on
        # every weekday, but Mon/Tue/Wed/Thu expirations several weeks out are highly
        # illiquid (most of their volume concentrates in the final day or two before
        # expiry) — Polygon's daily aggregate often has no bar at all for these on most
        # days. The standard Friday cycle (which also covers monthly expirations, always
        # the 3rd Friday) is where liquidity actually sits for a multi-week calendar
        # spread, so we only search among Friday expirations.
        RESTRICT_TO_FRIDAY_EXPIRATIONS = True
        expirations_by_strike = call_option_contract_data.groupby('strike_price')['expiration_date'].apply(
            lambda s: sorted({d for d in s if (not RESTRICT_TO_FRIDAY_EXPIRATIONS) or d.weekday() == 4})).to_dict()

        front_expirations = []
        back_expirations = []
        for _, row in CCS_day1.iterrows():
            avail = expirations_by_strike.get(row['Call_strike_price'], [])
            f_exp, b_exp = select_leg_expirations(row['Share_Price_Date'], avail, front_leg_target_days, back_leg_target_days)
            front_expirations.append(f_exp)
            back_expirations.append(b_exp)

        CCS_day1['Call_1_expiration_date'] = front_expirations  # Call_1 = front / short leg
        CCS_day1['Call_2_expiration_date'] = back_expirations   # Call_2 = back / long leg

        # Drop rows where we couldn't find a suitable front or back leg
        missing_legs_idx = CCS_day1[CCS_day1['Call_1_expiration_date'].isna() | CCS_day1['Call_2_expiration_date'].isna()].index
        for idx in missing_legs_idx:
            print(f"Dropped entry row | index={idx} | date={CCS_day1.loc[idx, 'Share_Price_Date']} (no suitable front/back leg found)")
        CCS_day1 = CCS_day1.drop(index=missing_legs_idx).reset_index(drop=True)
        CCS_day1_len = len(CCS_day1)

        CCS_day1['Call_1_Time_to_Expiration'] = (CCS_day1['Call_1_expiration_date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days) / 365
        CCS_day1['Call_2_Time_to_Expiration'] = (CCS_day1['Call_2_expiration_date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days) / 365

        # Exit date = N business days before front leg expiry
        CCS_day1['Exit_Date'] = CCS_day1['Call_1_expiration_date'].apply(
            lambda d: (pd.Timestamp(d) - exit_business_days_before_front_expiry * us_bday).date())

        CCS_day1 = pd.merge(CCS_day1, call_option_contract_data[['expiration_date', 'strike_price', 'ticker']],
                             how='left', left_on=['Call_1_expiration_date', 'Call_strike_price'],
                             right_on=['expiration_date', 'strike_price'])
        CCS_day1.drop(columns=['expiration_date', 'strike_price'], inplace=True)
        CCS_day1.rename(columns={'ticker': 'Call_1_Option_Ticker'}, inplace=True)

        CCS_day1 = pd.merge(CCS_day1, call_option_contract_data[['expiration_date', 'strike_price', 'ticker']],
                             how='left', left_on=['Call_2_expiration_date', 'Call_strike_price'],
                             right_on=['expiration_date', 'strike_price'])
        CCS_day1.drop(columns=['expiration_date', 'strike_price'], inplace=True)
        CCS_day1.rename(columns={'ticker': 'Call_2_Option_Ticker'}, inplace=True)

        if len(CCS_day1) != CCS_day1_len:
            raise ValueError("CCS_day1 gained rows when joining on call option contract tickers")

        ######################################
        ## Building the EXIT leg (CCS_day2) from the computed Exit_Date
        ######################################

        print("Setting up Call Calendar Spread strategy (exit)...")

        CCS_day2 = pd.DataFrame({
            'Share_Price_Date': CCS_day1['Exit_Date'],
            'Stock_Ticker': ticker,
            'Call_strike_price': CCS_day1['Call_strike_price'],
            'Call_1_expiration_date': CCS_day1['Call_1_expiration_date'],
            'Call_2_expiration_date': CCS_day1['Call_2_expiration_date'],
            'Call_1_Option_Ticker': CCS_day1['Call_1_Option_Ticker'],
            'Call_2_Option_Ticker': CCS_day1['Call_2_Option_Ticker'],
        })
        CCS_day2 = pd.merge(CCS_day2, stock_price_data[['Share_Price_Date', 'Decision_Price']],
                             how='left', left_on='Share_Price_Date', right_on='Share_Price_Date')
        CCS_day2.rename(columns={'Decision_Price': 'Sim_close'}, inplace=True)

        CCS_day2_len = len(CCS_day2)
        if len(CCS_day2) != CCS_day1_len:
            raise ValueError("CCS_day2 length does not match CCS_day1 length after construction")

        ######################################
        ## Getting call option contract prices (entry AND exit dates for each contract)
        ######################################

        option_price_data_request = pd.DataFrame({
            "Options_Ticker": pd.concat([CCS_day1["Call_1_Option_Ticker"], CCS_day1["Call_2_Option_Ticker"]], ignore_index=True),
            "First_Date": pd.concat([CCS_day1["Share_Price_Date"], CCS_day1["Share_Price_Date"]], ignore_index=True),
            "Second_Date": pd.concat([CCS_day1["Exit_Date"], CCS_day1["Exit_Date"]], ignore_index=True)
        }).dropna(subset=["Options_Ticker"])

        print(f"Generating prices for {len(option_price_data_request)} options tickers...")

        call_option_price_data = pd.DataFrame(columns=['Call_Options_Ticker', 'Call_Market_Price', 'Call_Option_Price_Date_UNIX'])

        num_options_tickers = len(option_price_data_request)
        for i in range(num_options_tickers):
            call_price = client.get_aggs(ticker=option_price_data_request['Options_Ticker'].iloc[i],
                                          multiplier=1, timespan='day',
                                          from_=option_price_data_request['First_Date'].iloc[i],
                                          to=option_price_data_request['Second_Date'].iloc[i])

            for x in range(len(call_price)):
                call_option_price_data.loc[len(call_option_price_data)] = [
                    option_price_data_request['Options_Ticker'].iloc[i], call_price[x].close, call_price[x].timestamp]

            if num_options_tickers > 0 and (i % max(1, num_options_tickers // 10) == 0 or i == num_options_tickers - 1):
                percent = (i / num_options_tickers) * 100
                print(f"{percent:.0f}% of options tickers completed")

        call_option_price_data = call_option_price_data.drop_duplicates()
        call_option_price_data['Call_Option_Price_Date'] = pd.to_datetime(
            call_option_price_data['Call_Option_Price_Date_UNIX'], unit='ms').dt.date

        # --- Decision-time intraday lookup for TODAY's option price (dtest/live runs) ---
        # Daily aggregate bars only finalize after market close (~9pm UK), but a dtest run at 5pm UK
        # happens well before that, so client.get_aggs(timespan='day') has NO bar yet for today's
        # entry-leg option price. The fix isn't "grab whatever the price is right now" (that would be
        # a 5:05pm price if you happen to run 5 minutes late, not a 5pm one) — it's to give options the
        # EXACT SAME decision-time treatment already used for the underlying stock: fetch 5-minute
        # intraday bars and pick the latest one at/before 5pm UK. This only fetches the handful of
        # contracts actually needed for today (front+back leg per ticker), so it's cheap.
        if run_reason == 'dtest':
            today_actual = datetime.today().date()
            needed_today = option_price_data_request[option_price_data_request['First_Date'] == today_actual]['Options_Ticker'].unique()
            have_today = set(call_option_price_data.loc[
                call_option_price_data['Call_Option_Price_Date'] == today_actual, 'Call_Options_Ticker']) if not call_option_price_data.empty else set()
            missing_today = [t for t in needed_today if t not in have_today]

            if missing_today:
                print(f"Fetching decision-time intraday prices for {len(missing_today)} option contract(s) with no daily bar yet for today...")
                new_rows = []
                for opt_ticker in missing_today:
                    try:
                        opt_intraday_request = client.list_aggs(ticker=opt_ticker, multiplier=5, timespan="minute",
                                                                  from_=today_actual, to=today_actual, limit=1000)
                        opt_intraday = pd.DataFrame(opt_intraday_request)
                        if opt_intraday.empty:
                            print(f"  WARNING: no intraday trades found yet for {opt_ticker} — likely too illiquid "
                                  f"to have printed a trade before the decision time.")
                            continue

                        opt_intraday['bar_dt'] = pd.to_datetime(opt_intraday['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')
                        decision_dt_ny = uk_decision_time_to_ny(today_actual)
                        eligible = opt_intraday[opt_intraday['bar_dt'] <= decision_dt_ny]
                        if eligible.empty:
                            print(f"  WARNING: {opt_ticker} has intraday bars today but none at/before the "
                                  f"decision time yet — try again closer to/after 5pm UK.")
                            continue

                        latest_bar = eligible.loc[eligible['bar_dt'].idxmax()]
                        new_rows.append({'Call_Options_Ticker': opt_ticker, 'Call_Market_Price': latest_bar['close'],
                                          'Call_Option_Price_Date_UNIX': int(latest_bar['timestamp']),
                                          'Call_Option_Price_Date': today_actual})
                    except Exception as e:
                        print(f"  WARNING: intraday fetch failed for {opt_ticker} ({e}).")
                if new_rows:
                    call_option_price_data = pd.concat([call_option_price_data, pd.DataFrame(new_rows)], ignore_index=True)
                    call_option_price_data = call_option_price_data.drop_duplicates(subset=['Call_Options_Ticker', 'Call_Option_Price_Date'])

        Spread = 0.005  # placeholder bid/ask spread assumption — see explanation notes re: refining this
        call_option_price_data['Call_Option_Price_Long'] = round(call_option_price_data['Call_Market_Price'] * (1 + Spread), 3)
        call_option_price_data['Call_Option_Price_Short'] = round(call_option_price_data['Call_Market_Price'] * (1 - Spread), 3)

        ######################################
        ## Joining Call Options Contract prices onto main Call Calendar Spread Tables
        ######################################

        CCS_day1 = pd.merge(CCS_day1, call_option_price_data[['Call_Options_Ticker', 'Call_Option_Price_Date', 'Call_Market_Price', 'Call_Option_Price_Long', 'Call_Option_Price_Short']],
                             how='left', left_on=['Call_1_Option_Ticker', 'Share_Price_Date'], right_on=['Call_Options_Ticker', 'Call_Option_Price_Date'])
        CCS_day1.drop(columns=['Call_Options_Ticker', 'Call_Option_Price_Date'], inplace=True)
        CCS_day1.rename(columns={'Call_Option_Price_Long': 'Call_1_Option_Price_Long', 'Call_Option_Price_Short': 'Call_1_Option_Price_Short', 'Call_Market_Price': 'Call_1_Market_Price'}, inplace=True)

        CCS_day1 = pd.merge(CCS_day1, call_option_price_data[['Call_Options_Ticker', 'Call_Option_Price_Date', 'Call_Market_Price', 'Call_Option_Price_Long', 'Call_Option_Price_Short']],
                             how='left', left_on=['Call_2_Option_Ticker', 'Share_Price_Date'], right_on=['Call_Options_Ticker', 'Call_Option_Price_Date'])
        CCS_day1.drop(columns=['Call_Options_Ticker', 'Call_Option_Price_Date'], inplace=True)
        CCS_day1.rename(columns={'Call_Option_Price_Long': 'Call_2_Option_Price_Long', 'Call_Option_Price_Short': 'Call_2_Option_Price_Short', 'Call_Market_Price': 'Call_2_Market_Price'}, inplace=True)

        CCS_day2 = pd.merge(CCS_day2, call_option_price_data[['Call_Options_Ticker', 'Call_Option_Price_Date', 'Call_Option_Price_Long', 'Call_Option_Price_Short']],
                             how='left', left_on=['Call_1_Option_Ticker', 'Share_Price_Date'], right_on=['Call_Options_Ticker', 'Call_Option_Price_Date'])
        CCS_day2.drop(columns=['Call_Options_Ticker', 'Call_Option_Price_Date'], inplace=True)
        CCS_day2.rename(columns={'Call_Option_Price_Long': 'Call_1_Option_Price_Long', 'Call_Option_Price_Short': 'Call_1_Option_Price_Short'}, inplace=True)

        CCS_day2 = pd.merge(CCS_day2, call_option_price_data[['Call_Options_Ticker', 'Call_Option_Price_Date', 'Call_Option_Price_Long', 'Call_Option_Price_Short']],
                             how='left', left_on=['Call_2_Option_Ticker', 'Share_Price_Date'], right_on=['Call_Options_Ticker', 'Call_Option_Price_Date'])
        CCS_day2.drop(columns=['Call_Options_Ticker', 'Call_Option_Price_Date'], inplace=True)
        CCS_day2.rename(columns={'Call_Option_Price_Long': 'Call_2_Option_Price_Long', 'Call_Option_Price_Short': 'Call_2_Option_Price_Short'}, inplace=True)

        ######################################
        ## Calculating metrics on call option contract prices (IV & Greeks, at ENTRY only)
        ######################################

        print("Calculating implied volatility and greeks for each options contract...")

        call_1_metric_inputs = CCS_day1[['Call_1_Option_Ticker', 'Sim_close', 'Call_strike_price', 'Call_1_Time_to_Expiration', 'Call_1_Market_Price', 'Share_Price_Date']].copy().reset_index(drop=True)
        call_1_metric_inputs = call_1_metric_inputs.dropna(subset=['Call_1_Option_Ticker'])
        call_1_metric_outputs = pd.concat([call_1_metric_inputs[['Share_Price_Date', 'Call_1_Option_Ticker']],
                                            calculate_iv_and_greeks(call_1_metric_inputs['Call_1_Option_Ticker'], call_1_metric_inputs, 'c')], axis=1)
        CCS_day1 = pd.merge(CCS_day1, call_1_metric_outputs, how='left', left_on=['Call_1_Option_Ticker', 'Share_Price_Date'], right_on=['Call_1_Option_Ticker', 'Share_Price_Date'])
        CCS_day1.rename(columns={'implied_volatility': 'Call_1_implied_volatility', 'delta': 'Call_1_delta', 'gamma': 'Call_1_gamma', 'theta': 'Call_1_theta', 'vega': 'Call_1_vega', 'rho': 'Call_1_rho', 'error': 'Call_1_error'}, inplace=True)

        CCS_day1["Call_1_Mean_IV_5"] = CCS_day1["Call_1_implied_volatility"].rolling(window=5).mean()
        CCS_day1["Call_1_IV_momentum_5"] = CCS_day1["Call_1_implied_volatility"] - CCS_day1["Call_1_Mean_IV_5"]
        CCS_day1["Call_1_IV_slope_5"] = (CCS_day1["Call_1_implied_volatility"].rolling(window=5).apply(linear_regression_slope, raw=True))

        call_2_metric_inputs = CCS_day1[['Call_2_Option_Ticker', 'Sim_close', 'Call_strike_price', 'Call_2_Time_to_Expiration', 'Call_2_Market_Price', 'Share_Price_Date']].copy().reset_index(drop=True)
        call_2_metric_inputs = call_2_metric_inputs.dropna(subset=['Call_2_Option_Ticker'])
        call_2_metric_outputs = pd.concat([call_2_metric_inputs[['Share_Price_Date', 'Call_2_Option_Ticker']],
                                            calculate_iv_and_greeks(call_2_metric_inputs['Call_2_Option_Ticker'], call_2_metric_inputs, 'c')], axis=1)
        CCS_day1 = pd.merge(CCS_day1, call_2_metric_outputs, how='left', left_on=['Call_2_Option_Ticker', 'Share_Price_Date'], right_on=['Call_2_Option_Ticker', 'Share_Price_Date'])
        CCS_day1.rename(columns={'implied_volatility': 'Call_2_implied_volatility', 'delta': 'Call_2_delta', 'gamma': 'Call_2_gamma', 'theta': 'Call_2_theta', 'vega': 'Call_2_vega', 'rho': 'Call_2_rho', 'error': 'Call_2_error'}, inplace=True)

        CCS_day1["Call_2_Mean_IV_5"] = CCS_day1["Call_2_implied_volatility"].rolling(window=5).mean()
        CCS_day1["Call_2_IV_momentum_5"] = CCS_day1["Call_2_implied_volatility"] - CCS_day1["Call_2_Mean_IV_5"]
        CCS_day1["Call_2_IV_slope_5"] = (CCS_day1["Call_2_implied_volatility"].rolling(window=5).apply(linear_regression_slope, raw=True))

        CCS_day1['IV_Term_Slope'] = CCS_day1["Call_1_implied_volatility"] - CCS_day1["Call_2_implied_volatility"]
        CCS_day1['IV_Term_Ratio'] = CCS_day1["Call_1_implied_volatility"] / CCS_day1["Call_2_implied_volatility"]
        CCS_day1['IV_Term_Slope_Mean_5'] = CCS_day1['IV_Term_Slope'].rolling(window=5).mean()
        CCS_day1['IV_Term_Slope_STD_5'] = CCS_day1['IV_Term_Slope'].rolling(window=10).std()
        CCS_day1['IV_Term_Slope_Z_Score_5'] = (CCS_day1['IV_Term_Slope'] - CCS_day1['IV_Term_Slope_Mean_5']) / CCS_day1['IV_Term_Slope_STD_5']

        # IV Rank: where does today's front-leg IV sit relative to its own trailing history for this
        # ticker (0 = lowest IV seen in the window, 1 = highest)? This is a standard vol-selling
        # signal — high IV rank means current premium is rich vs. recent norms, which is exactly
        # the kind of context IV_Term_Slope alone doesn't capture (that's front-vs-back on a single
        # day; this is today-vs-history). Window is in ENTRIES, not calendar days — with weekly
        # entries, 26 entries ~= 6 months of trailing history.
        IV_RANK_WINDOW = 26
        CCS_day1['IV_Rank_26'] = CCS_day1['Call_1_implied_volatility'].rolling(
            window=IV_RANK_WINDOW, min_periods=8
        ).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.nan, raw=False)

        CCS_day1['Theta_Ratio'] = np.clip(CCS_day1['Call_1_theta'].abs() / CCS_day1['Call_2_theta'].abs(), 0, 5)
        CCS_day1['Vega_Ratio'] = np.clip(CCS_day1['Call_2_vega'] / CCS_day1['Call_1_vega'], 0, 5)
        CCS_day1['Normalised_Net_Delta'] = (CCS_day1['Call_2_delta'] - CCS_day1['Call_1_delta']) / CCS_day1['Call_2_delta'].abs() + CCS_day1['Call_1_delta'].abs()
        CCS_day1['Gamma_Ratio'] = CCS_day1['Call_1_gamma'] / CCS_day1['Call_2_gamma']
        CCS_day1['Gamma_to_Theta'] = CCS_day1['Gamma_Ratio'].abs() / CCS_day1['Theta_Ratio'].abs()
        CCS_day1['Normalised_Net_Delta_Mean_5'] = CCS_day1['Normalised_Net_Delta'].rolling(window=5).mean()
        CCS_day1['Delta_Flip_Risk'] = (CCS_day1['Normalised_Net_Delta'] - CCS_day1['Normalised_Net_Delta_Mean_5']).abs()

        # New features reflecting the multi-week structure
        CCS_day1['Moneyness'] = (CCS_day1['Call_strike_price'] - CCS_day1['Sim_close']) / CCS_day1['Sim_close']
        CCS_day1['Front_DTE_At_Entry'] = (CCS_day1[''] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days)
        CCS_day1['Back_DTE_At_Entry'] = (CCS_day1['Call_2_expiration_date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days)
        CCS_day1['DTE_Gap'] = CCS_day1['Back_DTE_At_Entry'] - CCS_day1['Front_DTE_At_Entry']
        CCS_day1['Days_Held'] = (CCS_day1['Exit_Date'] - CCS_day1['Share_Price_Date']).apply(lambda d: d.days)
        ######################################
        ## Creating Call Calendar Spread summary file
        ######################################

        print("Calculating Call Calendar Spread strategy profits...")

        CCS_day1 = CCS_day1.reset_index(drop=True)
        CCS_day2 = CCS_day2.reset_index(drop=True)
        stock_price_data = stock_price_data.reset_index(drop=True)

        CCS_summary = CCS_day1[['Share_Price_Date', 'Call_strike_price']].copy()
        CCS_summary['Share_Price_Date'] = pd.to_datetime(CCS_summary['Share_Price_Date'], errors='coerce')
        CCS_summary['Exit_Date'] = pd.to_datetime(CCS_day1['Exit_Date'], errors='coerce')

        # Entry cashflow: receive premium selling front leg, pay premium buying back leg
        CCS_summary['Day_1_Cashflow'] = round(100 * (CCS_day1['Call_1_Option_Price_Short'] - CCS_day1['Call_2_Option_Price_Long']), 2)
        # Exit cashflow: pay to buy back (close) the front leg, receive premium selling (closing) the back leg
        CCS_summary['Day_2_Cashflow'] = round(100 * (CCS_day2['Call_2_Option_Price_Short'] - CCS_day2['Call_1_Option_Price_Long']), 2)
        CCS_summary['Overall Profit'] = round(CCS_summary['Day_1_Cashflow'] + CCS_summary['Day_2_Cashflow'], 2)
        CCS_summary['Call_1_P&L'] = round(100 * (CCS_day1['Call_1_Option_Price_Short'] - CCS_day2['Call_1_Option_Price_Long']), 2)
        CCS_summary['Call_2_P&L'] = round(100 * (-CCS_day1['Call_2_Option_Price_Long'] + CCS_day2['Call_2_Option_Price_Short']), 2)

        # Merge in Feature_Date-shifted technical indicators to avoid look-ahead bias (see header note):
        # a row's technical indicators come from the PRIOR fully-completed trading day, not the entry
        # day itself (whose high/low/close aren't final at a 5pm-UK decision point).
        # Built directly as a datetime64 series (not .date() objects) so its dtype matches
        # stock_price_data['Share_Price_Date'] below — mixing object-dtype dates with
        # datetime64 dates causes pandas to refuse the merge.
        CCS_summary['Feature_Date'] = CCS_summary['Share_Price_Date'] - us_bday
        stock_price_data['Share_Price_Date'] = pd.to_datetime(stock_price_data['Share_Price_Date'])

        CCS_summary = pd.merge(
            CCS_summary,
            stock_price_data[['Share_Price_Date', 'TR', 'Trend_Efficiency', 'ATR3', 'ATR7', 'ATR14', 'Normalised_ATR',
                               'Vol_Expanding', 'STD10', 'STD3', 'Trend_Distance', 'Trend_Distance_slope_5', 'BBW10',
                               'BBW3', 'ADX_Daily', 'ADX_Daily_slope_5', 'Is_Trending', 'RV_3', 'RV_5', 'RV_14',
                               'Vol_Regime_Ratio', 'daily_vol', 'second_half_vol']],
            how='left', left_on='Feature_Date', right_on='Share_Price_Date', suffixes=('', '_feat'))
        CCS_summary.drop(columns=['Share_Price_Date_feat'], inplace=True, errors='ignore')

        feature_cols = ['Call_1_implied_volatility', 'Call_1_Mean_IV_5', 'Call_1_IV_momentum_5', 'Call_1_IV_slope_5',
                         'Call_2_Mean_IV_5', 'Call_2_IV_momentum_5', 'Call_2_IV_slope_5', 'IV_Term_Slope', 'IV_Term_Ratio',
                         'IV_Term_Slope_Z_Score_5', 'Theta_Ratio', 'Vega_Ratio', 'Normalised_Net_Delta', 'Gamma_Ratio',
                         'Gamma_to_Theta', 'Normalised_Net_Delta_Mean_5', 'Delta_Flip_Risk', 'Moneyness',
                         'Front_DTE_At_Entry', 'Back_DTE_At_Entry', 'DTE_Gap', 'Days_Held', 'IV_Rank_26']
        CCS_summary[feature_cols] = CCS_day1[feature_cols]

        # Realized-vs-implied vol spread: a classic vol-selling signal — positive values mean implied
        # vol is pricing in more movement than has actually been realized recently (favorable for
        # selling premium), negative means the market may be under-pricing realized moves. RV_14 is a
        # daily log-return std, so annualize it (x sqrt(252)) to put it on the same scale as IV.
        CCS_summary['RV14_Annualized'] = CCS_summary['RV_14'] * np.sqrt(252)
        CCS_summary['IV_vs_RV_Spread'] = CCS_summary['Call_1_implied_volatility'] - CCS_summary['RV14_Annualized']

        # --- Days_To_Next_Earnings: via Financial Modeling Prep (FMP) ---
        # Still wrapped defensively — if FMP has no coverage for a ticker (e.g. some ETFs genuinely
        # have no earnings events) or the request fails for any reason, this fills NaN (imputed later,
        # with a global-median fallback) rather than crashing the whole import.
        # The full FMP-based earnings lookup below is disabled for now (no FMP_API_KEY configured).
        # We still need the column to exist though, since it's selected further down when building
        # CCS_summary_df — so fill it with NaN. The "drop columns that are entirely NaN" step later
        # in the script will then quietly remove it before training/imputation.
        CCS_summary['Days_To_Next_Earnings'] = np.nan
        CCS_summary['Days_To_Next_Earnings'] = CCS_summary['Days_To_Next_Earnings'].astype(float)

        # To re-enable the real lookup: set FMP_API_KEY, uncomment the block below, and remove the
        # two lines directly above.
        # if ENABLE_EARNINGS_FEATURE:
        #     try:
        #         fmp_url = f"https://financialmodelingprep.com/stable/earnings?symbol={ticker}&apikey={FMP_API_KEY}"
        #         fmp_response = requests.get(fmp_url, timeout=15)
        #         fmp_response.raise_for_status()
        #         fmp_data = fmp_response.json()
        #
        #         earnings_dates = sorted({
        #             pd.to_datetime(row['date']).date()
        #             for row in fmp_data if isinstance(row, dict) and row.get('date')
        #         })
        #         if not earnings_dates:
        #             raise ValueError("No earnings dates returned by FMP for this ticker — it may genuinely have no "
        #                               "earnings events (common for some ETFs), or FMP's plan tier may not cover it.")
        #
        #         def days_to_next_earnings(entry_date_):
        #             future = [d for d in earnings_dates if d >= entry_date_]
        #             return (min(future) - entry_date_).days if future else np.nan
        #
        #         CCS_summary['Days_To_Next_Earnings'] = CCS_summary['Share_Price_Date'].apply(
        #             lambda d: days_to_next_earnings(d.date()) if pd.notna(d) else np.nan)
        #         CCS_summary['Days_To_Next_Earnings'] = pd.to_numeric(CCS_summary['Days_To_Next_Earnings'], errors='coerce')
        #         print(f"Earnings proximity feature computed for {ticker} from {len(earnings_dates)} earnings dates (FMP).")
        #     except Exception as e:
        #         print(f"WARNING: could not compute Days_To_Next_Earnings for {ticker} ({e}). "
        #               f"Filling with NaN — verify the earnings endpoint for your plan if you need this feature.")
        #         CCS_summary['Days_To_Next_Earnings'] = np.nan
        #         CCS_summary['Days_To_Next_Earnings'] = CCS_summary['Days_To_Next_Earnings'].astype(float)
        # else:
        #     CCS_summary['Days_To_Next_Earnings'] = np.nan
        #     CCS_summary['Days_To_Next_Earnings'] = CCS_summary['Days_To_Next_Earnings'].astype(float)

        CCS_summary['days_since_prev'] = (CCS_summary['Share_Price_Date'] - CCS_summary['Share_Price_Date'].shift(1)).dt.days.fillna(0)
        CCS_summary['Share_Price_Date_Day'] = CCS_summary['Share_Price_Date'].dt.dayofweek

        ######################################
        ## Aggregating Data across tickers
        ######################################

        main_dataframes = [stock_price_data, call_option_contract_data, CCS_day1, CCS_day2, call_option_price_data, CCS_summary, stock_price_intra_day_data_filtered]
        for df_name in main_dataframes:
            df_name.loc[:, "Ticker"] = ticker

        combined_stock_data.append(stock_price_data)
        combined_call_option_data.append(call_option_contract_data)
        combined_call_spread_pt1.append(CCS_day1)
        combined_call_spread_pt2.append(CCS_day2)
        combined_call_option_prices.append(call_option_price_data)
        combined_call_calendar_spread_summary.append(CCS_summary)
        combined_IntraDayStockPrice.append(stock_price_intra_day_data_filtered)

        del stock_price_data, call_option_contract_data, CCS_day1, CCS_day2, call_option_price_data, CCS_summary, stock_price_intra_day_data_filtered



    ######################################
    ## Formatting Dataframes
    ######################################

    print(f"============================")
    print(f"Aggregating datasets to download...")
    print(f"============================")

    CCS_summary_df = pd.concat(combined_call_calendar_spread_summary, ignore_index=True).copy()
    CCS_summary_df = CCS_summary_df[['Ticker', 'Share_Price_Date', 'Exit_Date', 'Overall Profit', 'Call_strike_price',
                                      'Day_1_Cashflow', 'Day_2_Cashflow', 'Call_1_P&L', 'Call_2_P&L',
                                      'TR', 'Trend_Efficiency', 'ATR3', 'ATR7', 'ATR14', 'Normalised_ATR', 'Vol_Expanding',
                                      'STD10', 'STD3', 'Trend_Distance', 'Trend_Distance_slope_5', 'BBW10', 'BBW3',
                                      'ADX_Daily', 'ADX_Daily_slope_5', 'Is_Trending', 'RV_3', 'RV_5', 'RV_14',
                                      'Vol_Regime_Ratio', 'daily_vol', 'second_half_vol', 'Call_1_implied_volatility',
                                      'Call_1_Mean_IV_5', 'Call_1_IV_momentum_5', 'Call_1_IV_slope_5', 'Call_2_Mean_IV_5',
                                      'Call_2_IV_momentum_5', 'Call_2_IV_slope_5', 'IV_Term_Slope', 'IV_Term_Ratio',
                                      'IV_Term_Slope_Z_Score_5', 'Theta_Ratio', 'Vega_Ratio', 'Normalised_Net_Delta',
                                      'Gamma_Ratio', 'Gamma_to_Theta', 'Normalised_Net_Delta_Mean_5', 'Delta_Flip_Risk',
                                      'Moneyness', 'Front_DTE_At_Entry', 'Back_DTE_At_Entry', 'DTE_Gap', 'Days_Held',
                                      'IV_Rank_26', 'IV_vs_RV_Spread', 'Days_To_Next_Earnings',
                                      'days_since_prev', 'Share_Price_Date_Day']]

    CCS_summary_df_clean = CCS_summary_df.dropna(subset=['Overall Profit']).copy()
    # NOTE: only rows missing the actual outcome label are dropped here. A row can still have a
    # valid, completed trade (Overall Profit known) while one or two secondary engineered features
    # are NaN (e.g. the IV solver failing to converge on one leg, or a rolling-window feature still
    # warming up in the first ~2-3 weeks of a ticker's history) — those used to cause the WHOLE row
    # to be discarded under a blanket dropna(), which was needlessly shrinking the dataset. We now
    # median-impute those secondary feature gaps instead of throwing the row away.
    non_feature_cols = ['Ticker', 'Share_Price_Date', 'Exit_Date', 'Overall Profit', 'Call_strike_price',
                         'Day_1_Cashflow', 'Day_2_Cashflow', 'Call_1_P&L', 'Call_2_P&L']
    feature_cols_to_impute = [c for c in CCS_summary_df_clean.columns if c not in non_feature_cols]

    # Drop any feature column that is ENTIRELY NaN across the whole dataset (e.g. Days_To_Next_Earnings
    # when the earnings endpoint isn't available on your plan) BEFORE imputing. A column like this
    # can't be median-imputed (the median of an all-NaN column is itself NaN), and leaving it in would
    # mean the later safety-net dropna() sees every single row as still having a NaN — silently wiping
    # out the ENTIRE training set. Dropping the unusable column instead of the rows is the correct fix.
    fully_empty_cols = [c for c in feature_cols_to_impute if CCS_summary_df_clean[c].notna().sum() == 0]
    if fully_empty_cols:
        print(f"WARNING: dropping columns with NO usable data across the whole dataset (can't be "
              f"imputed, and would otherwise wipe out every row via the safety-net dropna): {fully_empty_cols}")
        CCS_summary_df_clean = CCS_summary_df_clean.drop(columns=fully_empty_cols)
        feature_cols_to_impute = [c for c in feature_cols_to_impute if c not in fully_empty_cols]

    na_before = CCS_summary_df_clean[feature_cols_to_impute].isna().sum().sum()
    # Impute per-ticker so a gap in one ticker's data isn't filled with another ticker's typical values.
    CCS_summary_df_clean[feature_cols_to_impute] = CCS_summary_df_clean.groupby('Ticker')[feature_cols_to_impute].transform(lambda s: s.fillna(s.median()))

    # Second pass: fall back to the GLOBAL (all-ticker) median for anything the per-ticker pass
    # couldn't fill — this happens when a ticker has ZERO valid values for a column (e.g. an
    # earnings-date feature that only some tickers have coverage for), where the per-ticker median
    # is itself NaN. Without this fallback, those rows would silently be wiped out by the safety-net
    # dropna() below even though the row's actual outcome label and other features are perfectly
    # valid — that's exactly what caused row counts to unexpectedly drop in an earlier run.
    remaining_na_cols = [c for c in feature_cols_to_impute if CCS_summary_df_clean[c].isna().any()]
    if remaining_na_cols:
        global_medians = CCS_summary_df_clean[remaining_na_cols].median()
        CCS_summary_df_clean[remaining_na_cols] = CCS_summary_df_clean[remaining_na_cols].fillna(global_medians)
        print(f"Applied global-median fallback (ticker had no valid data for this feature at all) to: {remaining_na_cols}")

    na_after = CCS_summary_df_clean[feature_cols_to_impute].isna().sum().sum()
    print(f"Imputed {na_before - na_after} missing feature values with per-ticker medians (rows kept as long as Overall Profit was known).")
    CCS_summary_df_clean = CCS_summary_df_clean.dropna().copy()  # catches any column that's entirely NaN dataset-wide (shouldn't happen — fully_empty_cols above already handles that case)
    print(f'Number of rows removed for training/testing datasets: {len(CCS_summary_df) - len(CCS_summary_df_clean)} (ignore for dtest)')
    print(f'Training Split: {(1 - test_split) * 100}% \n Testing Split: {test_split * 100}%')

    if run_reason == 'dtest':
        print(f'Script is being run for daily testing so no training dataset will be output')
        # FIXED: previously recomputed "today" independently here with an arbitrary -4 hours offset
        # (dtest_date = (datetime.today() - timedelta(hours=4)).date()), which could silently mismatch
        # the actual date used to build entry_dates earlier in the ticker loop — if they disagreed even
        # by one day, this filter would match ZERO rows for every ticker, producing an empty (headers-
        # only) testing CSV with no error. Now reuses the EXACT date the entries were actually built on.
        dtest_date = dtest_actual_date
        testing_dataset = CCS_summary_df[CCS_summary_df['Share_Price_Date'].dt.date == dtest_date].copy()

        if testing_dataset.empty:
            print(f"WARNING: no ticker produced a valid row for {dtest_date}. Most common reasons: "
                  f"(a) run before the 5pm UK decision time, so today's price bar doesn't exist yet, "
                  f"(b) today isn't a trading day (weekend/holiday), or (c) no suitable Friday "
                  f"expiration was found for any ticker. Check the 'Dropped entry row' warnings above.")
            # Fall back to the most recent PRIOR date that has at least one valid row, rather than
            # returning an empty file. This won't be a genuine "right now" signal — it reflects
            # whatever the last fully-built row was — so it's clearly flagged below and in the
            # output filename/column, not silently substituted.
            available_dates = sorted(CCS_summary_df['Share_Price_Date'].dt.date.dropna().unique())
            available_dates = [d for d in available_dates if d <= dtest_date]
            if available_dates:
                fallback_date = available_dates[-1]
                print(f"Falling back to the most recent available date instead: {fallback_date} "
                      f"(NOT today — treat this as a stale/reference signal, not a live one).")
                testing_dataset = CCS_summary_df[CCS_summary_df['Share_Price_Date'].dt.date == fallback_date].copy()
                testing_dataset['Is_Fallback_Date'] = True
                dtest_date = fallback_date
            else:
                print("No prior date has any valid rows either within the fetched window — nothing to fall back to.")

    elif run_reason == 'test':
        print(f'Script is being run for testing so no training dataset will be output')
        testing_dataset = CCS_summary_df_clean

    elif run_reason == "train":
        # CHRONOLOGICAL split, not random shuffle: with ~3-week overlapping holding periods,
        # a random split would place highly-correlated, overlapping trades on both sides of
        # the train/test boundary, leaking information and inflating reported test performance.
        # The earliest (1 - test_split) fraction of dates becomes training data; the most
        # recent test_split fraction is held out as a genuine forward-looking test set.
        CCS_summary_df_clean_sorted = CCS_summary_df_clean.sort_values(by='Share_Price_Date').reset_index(drop=True)
        split_idx = int((1 - test_split) * len(CCS_summary_df_clean_sorted))

        training_dataset = CCS_summary_df_clean_sorted.iloc[:split_idx].copy()
        testing_dataset = CCS_summary_df_clean_sorted.iloc[split_idx:].copy()

        if shuffle_training_dataset:
            # Shuffling ONLY the training rows (after the chronological cut) is fine — it just
            # affects the order XGBoost sees rows in, not which rows end up in train vs test.
            training_dataset = training_dataset.sample(frac=1, random_state=train_test_random_seed).reset_index(drop=True)

    else:
        raise ValueError(f"Not fully prepared for downloads for the following run reason: {run_reason}")

    ######################################
    ## Saving Files
    ######################################

    # output_dir lets the caller control where files land:
    #   - left as None: falls back to a folder next to this script (old local-only behaviour)
    #   - passed in (e.g. by the GitHub Actions runner as "outputs"): files land inside the repo
    #     checkout so a later workflow step can commit them back to GitHub as real files.
    base_directory = output_dir if output_dir else os.path.dirname(os.path.abspath(__file__))

    datetime_str = datetime.now().strftime("%Y-%m-%d__%H.%M")

    outer_folder_name = f"{datetime_str} {run_reason} Model Run"
    inner_folder_name = f"{run_reason} Import Outputs"

    full_downloads_path = os.path.join(base_directory, prediction_model_folder, outer_folder_name, inner_folder_name)
    os.makedirs(full_downloads_path, exist_ok=True)


    def ts():
        return datetime.today().strftime("%Y-%m-%d__%H.%M")


    stock_price_data_all = pd.concat(combined_stock_data, ignore_index=True)
    call_option_data_all = pd.concat(combined_call_option_data, ignore_index=True)
    call_spread_pt1_all = pd.concat(combined_call_spread_pt1, ignore_index=True)
    call_spread_pt2_all = pd.concat(combined_call_spread_pt2, ignore_index=True)
    call_option_prices_all = pd.concat(combined_call_option_prices, ignore_index=True)
    intraday_stock_price_all = pd.concat(combined_IntraDayStockPrice, ignore_index=True)

    stock_price_data_all.to_csv(os.path.join(full_downloads_path, f"Stock_Price_Data_{ts()}.csv"), index=False)
    call_option_data_all.to_csv(os.path.join(full_downloads_path, f"Call_Options_Contract_Data_{ts()}.csv"), index=False)
    call_spread_pt1_all.to_csv(os.path.join(full_downloads_path, f"4.CCS_Entry_Data_{ts()}.csv"), index=False)
    call_spread_pt2_all.to_csv(os.path.join(full_downloads_path, f"5.CCS_Exit_Data_{ts()}.csv"), index=False)
    call_option_prices_all.to_csv(os.path.join(full_downloads_path, f"Call_Option_Prices_{ts()}.csv"), index=False)
    CCS_summary_df.to_csv(os.path.join(full_downloads_path, f"3.CCS_Summary_Data_{ts()}.csv"), index=False)
    intraday_stock_price_all.to_csv(os.path.join(full_downloads_path, f"Intraday_Stock_Price_Data_{ts()}.csv"), index=False)

    if test_split != 1:
        training_dataset["Tick"] = training_dataset["Ticker"].map(ticker_map)
        training_dataset.to_csv(os.path.join(full_downloads_path, f"2.CCS_Training_Data_{ts()}.csv"), index=False)
    else:
        training_dataset = None

    if 'Is_Fallback_Date' not in testing_dataset.columns:
        testing_dataset['Is_Fallback_Date'] = False
    testing_dataset["Tick"] = testing_dataset["Ticker"].map(ticker_map)
    testing_dataset.to_csv(os.path.join(full_downloads_path, f"1.CCS_Testing_Data_{ts()}.csv"), index=False)

    print("Done.")

    # ------------------------------------------------------------------
    # Return everything as function outputs so other code (e.g. a
    # training script, or another step in the same workflow) can use
    # the data directly, without re-reading the CSVs from disk.
    # ------------------------------------------------------------------
    return {
        "output_dir": full_downloads_path,
        "stock_price_data": stock_price_data_all,
        "call_option_data": call_option_data_all,
        "call_spread_entry": call_spread_pt1_all,
        "call_spread_exit": call_spread_pt2_all,
        "call_option_prices": call_option_prices_all,
        "intraday_stock_price": intraday_stock_price_all,
        "ccs_summary": CCS_summary_df,
        "training_dataset": training_dataset,
        "testing_dataset": testing_dataset,
    }

# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 312_cost_access_understanding_price\n\npairs:\n  TM: 7203.T\n  SONY: 6758.T\n  HMC: 7267.T\nfx_ticker: JPY=X\nhedge_ticker: EWJ\n\nin_sample:\n  start: 2018-01-01\n  end: 2023-12-31\nout_of_sample:\n  start: 2024-01-01\n  end: 2026-10-31\n\nratio_window: 60\nratio_min_observations: 40\nentry_discount: 0.01\nmax_holding_sessions: 5\nblackout_sessions: 5\nbeta_window: 60\ngross_exposure: 1.0\ntransaction_cost_bps: 10  # Per side, per leg, on traded notional.\nannual_short_borrow_bps: 50\nsessions_per_year: 252\n\n# Exchange ex-dates can precede the issuer's legal effective date.\nscheduled_actions:\n  TM:\n    - {announced: 2021-05-12, start: 2021-09-29, end: 2021-10-01}\n  HMC:\n    - {announced: 2023-08-09, start: 2023-09-28, end: 2023-10-02}\n  SONY:\n    - {announced: 2024-05-14, start: 2024-09-27, end: 2024-10-09}\n    - {announced: 2025-09-08, start: 2025-09-29, end: 2025-10-07, reset_ratio: true}\n\n# Yahoo's SONY cash distributions omit this distribution in kind.\nspin_off:\n  adr: SONY\n  ex_date: 2025-09-29\n  local_distribution_ticker: 8729.T\n  local_shares_per_adr: 1.0\n"
_orig_open = _b.open
def _open(f, *a, **k):
    try:
        if _os.path.basename(str(f)) == 'config.yaml' and not _os.path.exists(f):
            return _io.StringIO(_CONFIG_YAML)
    except TypeError:
        pass
    return _orig_open(f, *a, **k)
_b.open = _open
_io.open = _open
# --- end shim ---
"""Backtest of the C103 Japanese ADR parity signal and its controls."""

import json
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
HOLIDAY_URL = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"


def save_csv(frame, path):
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def history(ticker, start, end, fields):
    path = CACHE / f"{ticker}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    if path.exists():
        return pd.read_csv(path, index_col="Date", parse_dates=True)[fields]

    frame = yf.Ticker(ticker).history(
        start=start.date().isoformat(), end=end.date().isoformat(),
        auto_adjust=False, actions=True, timeout=15,
    )
    if frame.empty or any(name not in frame for name in fields):
        raise RuntimeError(f"Yahoo Finance did not return the required history for {ticker}")
    if frame.index.tz is None:
        raise RuntimeError(f"Missing exchange time zone in the Yahoo history for {ticker}")
    # Preserve each exchange's local session date rather than the UTC date.
    frame.index = frame.index.tz_localize(None).normalize().rename("Date")
    frame = frame[fields].sort_index()
    if not frame.index.is_unique:
        raise RuntimeError(f"Duplicate daily bars for {ticker}")
    save_csv(frame, path)
    return frame


def tokyo_holidays(start, end):
    path = CACHE / "japan_national_holidays.csv"
    if path.exists():
        holiday_dates = pd.read_csv(path, index_col="Date", parse_dates=True).index
    else:
        response = requests.get(HOLIDAY_URL, timeout=20)
        response.raise_for_status()
        dates = pd.read_csv(BytesIO(response.content), encoding="cp932", usecols=[0])
        holiday_dates = pd.DatetimeIndex(pd.to_datetime(dates.iloc[:, 0]), name="Date")
        save_csv(pd.DataFrame(index=holiday_dates), path)
    if not {start.year, end.year}.issubset(set(holiday_dates.year)):
        raise RuntimeError("The cached Japanese national holiday calendar lacks a required year")
    dates = pd.date_range(start, end, freq="B")
    fixed_closure = dates.strftime("%m-%d").isin(["01-01", "01-02", "01-03", "12-31"])
    return dates[dates.isin(holiday_dates) | fixed_closure]


def blackout(sessions, start, end, before, after):
    left = max(0, sessions.searchsorted(start) - before)
    right = sessions.searchsorted(end, side="right") + after
    return sessions[left:right]


def lagged_fx(fx, dates):
    # The previous calendar date's FX bar has ended before the US signal close.
    rates = fx.reindex(dates - pd.Timedelta(days=1), method="ffill", tolerance="4D")
    return rates.where(rates > 0).set_axis(dates)


def close_parity(config, prices, fx, sessions, holidays):
    window = config["ratio_window"]
    min_periods = config["ratio_min_observations"]
    pad = config["blackout_sessions"]
    holiday_block = pd.Series(False, index=sessions)
    for date in holidays:
        holiday_block.loc[blackout(sessions, date, date, pad, pad)] = True

    spreads = pd.DataFrame(index=sessions)
    entry_block = pd.DataFrame(index=sessions)
    for adr, local in config["pairs"].items():
        planned = pd.Series(False, index=sessions)
        regime = pd.Series(0, index=sessions)
        for event in config["scheduled_actions"].get(adr, []):
            first, last = pd.Timestamp(event["start"]), pd.Timestamp(event["end"])
            affected = blackout(sessions, first, last, pad, pad)
            planned.loc[affected[affected > pd.Timestamp(event["announced"])]] = True
            if event.get("reset_ratio", False):
                regime.loc[first:] += 1

        observed = pd.Series(False, index=sessions)
        for ticker in (adr, local):
            splits = prices[ticker]["Stock Splits"]
            for date in splits.index[splits.fillna(0).ne(0)]:
                observed.loc[blackout(sessions, date, date, 0, pad)] = True

        local_close = prices[local]["Close"].reindex(sessions)
        ratio = prices[adr]["Close"].reindex(sessions) * fx / local_close.where(local_close > 0)
        ratio = ratio.mask(planned | observed | sessions.isin(holidays))
        # Exclude today's ratio and isolate explicitly changed price regimes.
        median = ratio.groupby(regime).transform(
            lambda values: values.shift().rolling(window, min_periods=min_periods).median()
        )
        spreads[adr] = ratio.div(median).sub(1).mask(holiday_block)
        entry_block[adr] = planned | holiday_block
    return spreads, entry_block


def entry_signals(spreads, betas, entry_block, discount, confirmation):
    eligible = spreads.lt(-discount)
    if confirmation:
        eligible &= eligible.shift(fill_value=False)
    # The block for the fill date is based on announced events and scheduled holidays.
    eligible &= betas.notna() & ~entry_block.shift(-1, fill_value=True)
    ranked = spreads.where(eligible).to_numpy()
    # Keep the configured ticker order for ties; use -1 when no ticker qualifies.
    best = np.where(np.isnan(ranked), np.inf, ranked).argmin(axis=1)
    best[~eligible.any(axis=1).to_numpy()] = -1
    return pd.Series(best, index=spreads.index).shift(fill_value=-1)


def simulate(config, opens, closes, payouts, previous_spreads, previous_betas, entries, initial_basket=None):
    cash = previous_nav = 1.0
    holdings = np.zeros(opens.shape[1])
    active = None
    opened_on = 0
    fee = config["transaction_cost_bps"] / 10_000
    borrow = config["annual_short_borrow_bps"] / 10_000
    gross = config["gross_exposure"]
    close_signals = previous_spreads.to_numpy()
    fill_betas = previous_betas.to_numpy()
    results = []

    for i, (opening, closing, distribution) in enumerate(
        zip(opens.to_numpy(), closes.to_numpy(), payouts.to_numpy())
    ):
        cash += holdings @ distribution
        open_nav = cash + holdings @ opening
        dollars = None
        trade = 0
        if initial_basket is not None:
            if i == 0:
                dollars = initial_basket * open_nav * gross
                trade = 1
        elif active is not None:
            expired = i - opened_on >= config["max_holding_sessions"]
            at_parity = close_signals[i, active] >= 0
            if expired or at_parity:
                dollars = np.zeros_like(holdings)
                active = None
        elif entries.iat[i] >= 0:
            active = entries.iat[i]
            hedge = fill_betas[i, active]
            dollars = np.zeros_like(holdings)
            dollars[active] = open_nav * gross / (1 + hedge)
            dollars[-1] = -hedge * dollars[active]
            opened_on = i
            trade = 1

        turnover = 0.0
        if dollars is not None:
            change = dollars - holdings * opening
            traded = np.abs(change).sum()
            cash -= change.sum() + fee * traded
            holdings = dollars / opening
            turnover = traded / open_nav
        cash -= max(0.0, -holdings[-1] * closing[-1]) * borrow / config["sessions_per_year"]
        nav = cash + holdings @ closing
        if not np.isfinite(nav) or nav <= 0:
            raise RuntimeError(f"Invalid portfolio equity on {opens.index[i].date()}")
        results.append((nav / previous_nav - 1, turnover, trade))
        previous_nav = nav
    return pd.DataFrame(results, index=opens.index, columns=["return", "turnover", "n_trades"])


def metrics(frame, sessions_per_year):
    returns = frame["return"]
    equity = (1 + returns).cumprod()
    volatility = returns.std(ddof=1)
    drawdown = equity.div(equity.cummax().clip(lower=1)).sub(1)
    result = {
        "cagr": equity.iat[-1] ** (sessions_per_year / len(equity)) - 1,
        "ann_vol": volatility * np.sqrt(sessions_per_year),
        "sharpe": returns.mean() / volatility * np.sqrt(sessions_per_year) if volatility > 0 else None,
        "max_drawdown": drawdown.min(),
        "total_return": equity.iat[-1] - 1,
        "n_trades": int(frame["n_trades"].sum()),
        "turnover": frame["turnover"].sum(),
    }
    return {
        key: value if isinstance(value, int) else float(value) if value is not None and np.isfinite(value) else None
        for key, value in result.items()
    }


def print_results(strategy, periods, runs, sessions_per_year):
    first, last = periods["full"]
    results = {"strategy": strategy, "data_range": f"{first:%Y-%m-%d} to {last:%Y-%m-%d}"}
    print(f"{strategy} | {results['data_range']} | USD, net of modeled costs")
    print(f"{'Portfolio':<24} {'Split':<5} {'CAGR':>9} {'Vol':>9} {'Sharpe':>8}"
          f" {'Max DD':>9} {'Return':>9} {'Trades':>7} {'Turnover':>9}")
    columns = [
        ("cagr", 9, ".2%"), ("ann_vol", 9, ".2%"), ("sharpe", 8, ".2f"),
        ("max_drawdown", 9, ".2%"), ("total_return", 9, ".2%"),
        ("n_trades", 7, "d"), ("turnover", 9, ".2f"),
    ]
    for key, (name, frame) in runs.items():
        stats = {}
        for split, (start, end) in periods.items():
            row = metrics(frame.loc[start:end], sessions_per_year)
            stats[split] = row
            values = []
            for field, width, spec in columns:
                value = format(row[field], spec) if row[field] is not None else "n/a"
                if field == "turnover" and row[field] is not None:
                    value += "x"
                values.append(value.rjust(width))
            print(f"{name:<24} {split.upper():<5} {' '.join(values)}")
        if key == "strategy":
            results.update(stats)
        else:
            results[key] = {"name": name, **stats}
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not is_start <= is_end < oos_start <= oos_end or (oos_start - is_end).days != 1:
        raise ValueError("IS and OOS must be ordered, nonoverlapping, and contiguous")

    start = is_start - pd.Timedelta(days=180)
    end = oos_end + pd.Timedelta(days=15)  # Calendar padding for the final blackout window.
    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yahoo"))
    adrs = list(config["pairs"])
    hedge = config["hedge_ticker"]
    us_tickers = [*adrs, hedge]
    prices = {ticker: history(ticker, start, end, ["Open", "Close", "Dividends", "Stock Splits"])
              for ticker in us_tickers}
    prices.update({ticker: history(ticker, start, end, ["Close", "Stock Splits"])
                   for ticker in config["pairs"].values()})
    sessions = prices[hedge].index
    if sessions.empty or sessions.min() > start + pd.Timedelta(days=10) or sessions.max() < oos_end - pd.Timedelta(days=5):
        raise RuntimeError("EWJ history does not cover the configured backtest")
    holidays = tokyo_holidays(start, end)
    fx_history = history(config["fx_ticker"], start, end, ["Close"])["Close"]
    fx = lagged_fx(fx_history, sessions)
    spreads, entry_block = close_parity(config, prices, fx, sessions, holidays)

    def field(name):
        return pd.DataFrame({ticker: prices[ticker][name].reindex(sessions) for ticker in us_tickers})

    opens, closes = field("Open"), field("Close")
    report_prices = pd.concat([opens.loc[:oos_end], closes.loc[:oos_end]], axis=1)
    if not np.isfinite(report_prices.to_numpy()).all() or (report_prices <= 0).any().any():
        raise RuntimeError("Missing or invalid U.S. OHLC data on the EWJ session calendar")
    payouts = field("Dividends").fillna(0)
    spin = config["spin_off"]
    ex_date = pd.Timestamp(spin["ex_date"])
    if start <= ex_date <= oos_end:
        local_mark = history(spin["local_distribution_ticker"], ex_date,
                             ex_date + pd.Timedelta(days=1), ["Close"])
        payout = spin["local_shares_per_adr"] * local_mark.at[ex_date, "Close"] / fx.at[ex_date]
        if not np.isfinite(payout) or payout <= 0:
            raise RuntimeError("Missing market data for the SONY distribution in kind")
        payouts.at[ex_date, spin["adr"]] += payout

    returns = (closes + payouts).div(closes.shift()).sub(1)
    hedge_returns = returns[hedge]
    variance = hedge_returns.rolling(config["beta_window"]).var()
    covariance = returns[adrs].rolling(config["beta_window"]).cov(hedge_returns)
    raw_betas = covariance.div(variance.where(variance > 0), axis=0)
    betas = raw_betas.clip(lower=0)

    report = sessions[(sessions >= is_start) & (sessions <= oos_end)]
    if not (report < oos_start).any() or not (report >= oos_start).any():
        raise RuntimeError("Each sample must contain at least one EWJ trading session")
    # Build decisions on the warm-up-inclusive calendar before slicing into the report.
    fills = [entry_signals(spreads, betas, entry_block, config["entry_discount"], confirmation)
             .loc[report] for confirmation in (False, True)]
    previous_session = sessions[sessions.get_loc(report[0]) - 1]
    initial_betas = raw_betas.loc[previous_session]
    if not np.isfinite(initial_betas).all():
        raise RuntimeError("Insufficient warm-up to compute the benchmark hedge")
    basket_beta = max(0.0, initial_betas.mean())
    basket = np.full(len(us_tickers), 1 / len(adrs))
    basket[-1] = -basket_beta
    basket /= 1 + basket_beta

    simulation_data = (config, opens.loc[report], closes.loc[report], payouts.loc[report],
                       spreads.shift().loc[report], betas.shift().loc[report])
    runs = {
        "strategy": ("ADR parity", simulate(*simulation_data, fills[0])),
        "benchmark": ("EWJ-hedged ADR basket", simulate(*simulation_data, fills[0], initial_basket=basket)),
        "two_breach": ("Two consecutive breaches", simulate(*simulation_data, fills[1])),
    }
    periods = {
        "is": (report[0], is_end),
        "oos": (oos_start, report[-1]),
        "full": (report[0], report[-1]),
    }
    print_results(config["strategy"], periods, runs, config["sessions_per_year"])


if __name__ == "__main__":
    main()

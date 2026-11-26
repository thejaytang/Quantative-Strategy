"""105_factor_momentum_style_etf
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/105_factor_momentum_style_etf.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e45e81ae3a3f162e8a4d3aa6588411ed39316f30 on 2026-11-26 13:10 Europe/Oslo, exit 0, output sha256 97bd5c7997faa73a24b836225ad22f34c30e54af589ea57c196f6aa3bbe22606
Data source: Yahoo Finance daily ETF opens and closes via yfinance. Style ETFs are VLUE, MTUM, QUAL, USMV and SIZE, with SPY as the market benchmark and BIL for unallocated slots. The loader reads its CSV cache when available and calls Yahoo Finance on a cache miss. The supplied run used the cached CSV.
Data fetched: 2026-11-26 13:10 Europe/Oslo (run time)
Sample period: Evaluated trading sessions were 2016-01-04 through 2025-12-31. The configured price request starts on 2014-10-01 to provide signal warm-up history.   data_range (from output): 2016-01-04 to 2025-12-31
IS/OOS split: Configured IS is 2016-01-01 through 2022-12-31 and configured OOS is 2023-01-01 through 2025-12-31. Actual IS sessions ran 2016-01-04 through 2022-12-30 and OOS sessions ran 2023-01-03 through 2025-12-31. OOS is a historical time holdout. Each portfolio is simulated continuously, with holdings carried across the split and daily results sliced for reporting.
Costs and slippage: The simulation charges 5 bps on each buy and each sell, including initial purchases and BIL trades, using gross traded notional. At each rebalance it solves for post-fee NAV before assigning the target holdings. The same fee rate applies to both benchmarks, and final holdings are marked to market. Slippage is not modelled. The simulation uses the adjusted opening price with no separate spread or market-impact adjustment.
Signal lag: At the last trading session's close of each month, the strategy computes each style ETF's trailing 252-session return minus SPY's trailing return. The portfolio is rebalanced at the next trading session's open, which is the first session of the new month. An initial evaluation session that falls mid-month uses the latest available monthly signal.
Price adjustment: The Yahoo Finance download sets auto_adjust=True and retains the adjusted Open and Close fields. The same adjusted close series is used for signals, and adjusted opens and closes are used for the simulated return legs.
Survivorship bias: The fixed ETF pool uses currently available fund price histories and can have fund-level selection and survivorship bias. The signal uses the funds' own historical price series, with no reconstruction from current constituent stocks.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09175394345342092, ann_vol=0.16653140082207998, sharpe=0.6108937593957354, max_drawdown=-0.33554527864178263, total_return=0.8474419776161672, turnover=35.65626069803559
    oos: cagr=0.17014972208243084, ann_vol=0.13524772948923133, sharpe=1.229679022711243, max_drawdown=-0.14827682249571705, total_return=0.5982367184053481, turnover=8.476995600799574
    full: cagr=0.11463689309321756, ann_vol=0.1578094088618508, sharpe=0.7670643508346912, max_drawdown=-0.33554527864178263, total_return=1.9526496037495646, turnover=44.133256298835164
    benchmark: 
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 105_factor_momentum_style_etf\nuniverse:\n  style_etfs: [VLUE, MTUM, QUAL, USMV, SIZE]\n  market_etf: SPY\n  cash_etf: BIL\nlookback_sessions: 252\nslots: 2\ntransaction_cost_bps: 5.0\nannualization_sessions: 252\ndata_start: 2014-10-01\nin_sample:\n  start: 2016-01-01\n  end: 2022-12-31\nout_of_sample:\n  start: 2023-01-01\n  end: 2025-12-31\n'
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
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
DAY = pd.Timedelta(days=1)


def load_prices(tickers, start, end):
    cache_path = CACHE / f"yahoo_adjusted_{'_'.join(tickers)}_{start.date()}_{end.date()}.csv"
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers])
    if cache_path.exists():
        prices = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True,
                             float_precision="round_trip")
    else:
        CACHE.mkdir(exist_ok=True)
        yf.set_tz_cache_location(str(CACHE / "yfinance"))
        prices = yf.download(
            tickers, start=str(start.date()), end=str((end + DAY).date()),
            auto_adjust=True, interval="1d", ignore_tz=True, progress=False, timeout=15,
        )

    prices = prices.reindex(columns=columns).sort_index().loc[start:end]
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Yahoo Finance returned no sessions or duplicate dates.")
    if prices.index[-1] < end - 7 * DAY:
        raise ValueError(f"Last available session {prices.index[-1].date()} is too far from {end.date()}.")
    invalid = ~np.isfinite(prices) | prices.le(0)
    if invalid.any().any():
        date = invalid.index[invalid.any(axis=1)][0]
        fields = ", ".join(f"{ticker} {field}" for field, ticker in invalid.columns[invalid.loc[date]])
        raise ValueError(f"Invalid or missing adjusted prices on {date.date()} for {fields}.")

    if not cache_path.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE, suffix=".tmp", delete=False) as handle:
            temporary_path = Path(handle.name)
            try:
                prices.to_csv(handle, index_label="Date")
                handle.close()
                temporary_path.replace(cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return prices


def build_targets(closes, evaluation_dates, styles, market, cash_etf, lookback, slots):
    months = closes.index.to_period("M")
    first_sessions = closes.index[~months.duplicated()]
    trade_dates = first_sessions.intersection(evaluation_dates).union(evaluation_dates[:1])

    trailing_returns = closes.div(closes.shift(lookback)).sub(1)
    scores = trailing_returns[styles].sub(trailing_returns[market], axis=0)
    # At each first session's open, only the previous session's close is known.
    monthly_scores = scores.shift(1).loc[first_sessions].reindex(trade_dates, method="ffill")
    if monthly_scores.isna().any().any():
        raise ValueError("Insufficient history for the first allocation. Move data_start earlier.")

    rotation = pd.DataFrame(0.0, index=trade_dates, columns=closes.columns)
    for date, row in monthly_scores.iterrows():
        winners = row[row.gt(0)].nlargest(slots).index
        rotation.loc[date, winners] = 1 / slots
        rotation.at[date, cash_etf] = (slots - len(winners)) / slots

    equally_weighted = pd.DataFrame(0.0, index=trade_dates, columns=closes.columns)
    equally_weighted.loc[:, styles] = 1 / len(styles)
    market_only = pd.DataFrame(0.0, index=evaluation_dates[:1], columns=closes.columns)
    market_only.loc[:, market] = 1.0
    return {"Rotation": rotation, market: market_only, "Equal-weight styles": equally_weighted}


def simulate(overnight, intraday, targets, fee_rate):
    # Holdings are dollar values. Cash is used for the initial purchase only.
    holdings = np.zeros(intraday.shape[1])
    cash = previous_nav = 1.0
    events = {date: row.to_numpy() for date, row in targets.iterrows()}
    daily = []
    for date, gap, session in zip(intraday.index, overnight.to_numpy(), intraday.to_numpy()):
        holdings *= gap
        turnover = 0.0
        if date in events:
            opening_nav = holdings.sum() + cash
            old_weights = holdings / opening_nav
            target = events[date]

            # Solve for affordable post-fee NAV, charging both buys and sells.
            def gross_turnover(nav_fraction):
                return np.abs(nav_fraction * target - old_weights).sum()

            net_fraction = brentq(lambda fraction: fraction + fee_rate * gross_turnover(fraction) - 1, 0, 1)
            turnover = gross_turnover(net_fraction)
            holdings = opening_nav * net_fraction * target
            cash = 0.0

        holdings *= session
        nav = holdings.sum() + cash
        daily.append((nav / previous_nav - 1, turnover))
        previous_nav = nav
    return pd.DataFrame(daily, index=intraday.index, columns=["return", "turnover"])


def metrics(daily, sessions_per_year):
    returns = daily["return"]
    equity = returns.add(1).cumprod()
    volatility = returns.std(ddof=1)
    values = {
        "cagr": equity.iloc[-1] ** (sessions_per_year / len(equity)) - 1,
        "ann_vol": volatility * math.sqrt(sessions_per_year),
        "sharpe": returns.mean() / volatility * math.sqrt(sessions_per_year) if volatility > 0 else None,
        "max_drawdown": equity.div(equity.cummax().clip(lower=1)).sub(1).min(),
        "total_return": equity.iloc[-1] - 1,
        "turnover": daily["turnover"].sum(),
    }
    return {key: float(value) if value is not None and math.isfinite(value) else None
            for key, value in values.items()}


def print_results(strategy, periods, performance, market):
    start, end = periods["full"]
    data_range = f"{start.date()} to {end.date()}"
    print(f"{strategy} | Evaluated sessions: {data_range}")
    print(" | ".join(f"{period.upper()}: {first.date()} to {last.date()}"
                     for period, (first, last) in periods.items() if period != "full"))
    print(f"{'Portfolio':<21} {'Period':<6} {'CAGR':>9} {'Ann.Vol':>9} {'Sharpe':>8} "
          f"{'Max.DD':>9} {'Return':>9} {'Turnover':>9}")
    for portfolio, segments in performance.items():
        for period, segment in segments.items():
            row = []
            for metric, value in segment.items():
                display_format = {"sharpe": ".2f", "turnover": ".2f"}.get(metric, ".2%")
                row.append(format(value, display_format) if value is not None else "n/a")
            cagr, vol, sharpe, drawdown, total, turnover = row
            print(f"{portfolio:<21} {period.upper():<6} {cagr:>9} {vol:>9} {sharpe:>8} "
                  f"{drawdown:>9} {total:>9} {turnover:>9}")

    results = {
        "strategy": strategy,
        "data_range": data_range,
        **performance["Rotation"],
        "benchmark": {market: performance[market], "equal_weight_styles": performance["Equal-weight styles"]},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)
    universe = config["universe"]
    styles = universe["style_etfs"]
    market, cash_etf = universe["market_etf"], universe["cash_etf"]
    tickers = [*styles, market, cash_etf]
    slots = config["slots"]
    lookback = config["lookback_sessions"]
    sessions_per_year = config["annualization_sessions"]
    fee_rate = config["transaction_cost_bps"] / 10_000
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][edge]) for edge in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][edge]) for edge in ("start", "end"))

    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Use ordered sample dates with warm-up history and no IS/OOS overlap.")
    if len(set(tickers)) != len(tickers) or not 1 <= slots <= len(styles):
        raise ValueError("Use distinct tickers and between one and the number of style ETFs for slots.")
    if lookback < 1 or sessions_per_year < 1 or not 0 <= fee_rate < 1:
        raise ValueError("Lookback and annualization must be positive, and costs must be below 10,000 bps.")

    prices = load_prices(tickers, data_start, oos_end)
    opens, closes = prices["Open"], prices["Close"]
    windows = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    periods = {}
    for period, (start, end) in windows.items():
        sessions = closes.loc[start:end].index
        if sessions.empty:
            raise ValueError(f"No sessions in the {period.upper()} window.")
        periods[period] = (sessions[0], sessions[-1])

    evaluation_dates = closes.loc[is_start:oos_end].index
    targets = build_targets(closes, evaluation_dates, styles, market, cash_etf, lookback, slots)
    overnight = opens.div(closes.shift(1)).loc[evaluation_dates]
    intraday = closes.div(opens).loc[evaluation_dates]
    performance = {}
    for portfolio, weights in targets.items():
        daily = simulate(overnight, intraday, weights, fee_rate)
        performance[portfolio] = {
            period: metrics(daily.loc[start:end], sessions_per_year)
            for period, (start, end) in periods.items()
        }
    print_results(config["strategy"], periods, performance, market)


if __name__ == "__main__":
    main()

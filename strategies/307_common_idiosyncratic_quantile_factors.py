"""307_common_idiosyncratic_quantile_factors
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/307_common_idiosyncratic_quantile_factors.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit dfc503a96cfc21dd90d6de1640b6ece149e0083f on 2026-11-27 06:54 Europe/Oslo, exit 0, output sha256 ac27d8c5afa64cadb4ad64a8980da76ca2bd7beae2e52607f41aac875525f04a
Data source: Yahoo Finance via yfinance, using daily Open and Close prices. The 24 US equities are AAPL, MSFT, GOOGL, AMZN, NVDA, CSCO, ORCL, IBM, JPM, BAC, GS, WFC, JNJ, PFE, MRK, UNH, XOM, CVX, COP, SLB, PG, KO, PEP, and WMT. The factor ETFs are SPY, IWM, IWD, and IWF. The factor return series are SPY, IWM minus SPY, and IWD minus IWF.
Data fetched: 2026-11-27 06:54 Europe/Oslo (run time)
Sample period: Requested price history including warm-up: 2005-01-01 through 2025-12-31. The run reports observed price history from 2005-01-03 through 2025-12-31 and a backtest from 2008-01-02 through 2025-12-31.   data_range (from output): 2008-01-02 to 2025-12-31
IS/OOS split: IS is 2008-01-01 through 2017-12-31. OOS is 2018-01-01 through 2025-12-31. Both reporting intervals are inclusive, and metrics are calculated by slicing daily returns from a single continuous simulation per portfolio, with holdings carried across the split. The README describes a chronological holdout and states that neither interval was used for tuning.
Costs and slippage: The simulator charges 10 bps one way on the sum of absolute changes in position notional at each rebalance, and target positions are sized on equity after that charge. Short borrow costs 50 bps annually, divided by 252 sessions and charged each session on absolute short exposure after any opening rebalance. Slippage is not separately modelled. Fills use adjusted opening prices, and spread and market impact are omitted.
Signal lag: At the last trading session close of each month, formation uses returns through that close. Both signal portfolios rebalance at the next trading session’s open, on the first observed session of the new month. Old positions earn the overnight move to that open, and the new positions earn the open-to-close move. The SPY buy-and-hold benchmark enters at the first backtest open on a preset target.
Price adjustment: The downloader sets yfinance auto_adjust=True for Open and Close, includes the requested end date by passing the next calendar day to yfinance, and stores the adjusted panel in a CSV. Signals use adjusted close-to-close returns. The simulation values overnight and intraday moves with adjusted opens and closes. A fresh download may produce a different adjusted price snapshot.
Survivorship bias: The fixed US24 stock list was selected retrospectively from surviving large-cap names and introduces survivorship bias. It lacks historical membership changes and delisting returns. Eligibility requires complete returns within each formation window.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.0340466377345372, ann_vol=0.10340599871805388, sharpe=0.3752674421110719, max_drawdown=-0.18681139865732566, total_return=0.3972878102582178, turnover=67.30612428033957, n_trades=1792
    oos: cagr=-0.031020069366245684, ann_vol=0.09341113414494928, sharpe=-0.29061622210257, max_drawdown=-0.2912805791650085, total_return=-0.2223399500134613, turnover=57.498882148389654, n_trades=1455
    full: cagr=0.004632679142801965, ann_vol=0.09910332706579528, sharpe=0.09606686676127542, max_drawdown=-0.3705404867132067, total_return=0.0866149083709844, turnover=124.80500642872921, n_trades=3247
    benchmark  full: cagr=0.10969713599992947, ann_vol=0.19952279396556535, sharpe=0.621601012929016, max_drawdown=-0.5244680050352296, total_return=5.492690388361079, turnover=0.9990009990009991, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 307_common_idiosyncratic_quantile_factors\nuniverse:\n  - AAPL\n  - MSFT\n  - GOOGL\n  - AMZN\n  - NVDA\n  - CSCO\n  - ORCL\n  - IBM\n  - JPM\n  - BAC\n  - GS\n  - WFC\n  - JNJ\n  - PFE\n  - MRK\n  - UNH\n  - XOM\n  - CVX\n  - COP\n  - SLB\n  - PG\n  - KO\n  - PEP\n  - WMT\nfactor_etfs:\n  market: SPY\n  small_cap: IWM\n  value: IWD\n  growth: IWF\ndata_start: 2005-01-01\nin_sample:\n  start: 2008-01-01\n  end: 2017-12-31\nout_of_sample:\n  start: 2018-01-01\n  end: 2025-12-31\nparameters:\n  lookback_sessions: 504\n  calibration_sessions: 252\n  tail_quantile: 0.10\n  names_per_leg: 6\n  gross_exposure: 1.0\n  sessions_per_year: 252\ncosts:\n  transaction_bps: 10\n  short_borrow_bps_per_year: 50\n# LIMITATION (Codex): a survivorship-free public free stock universe is not feasible under the free-public-data rules; the paper uses a commercial database. The current ticker list is a surviving-name proxy. See README.\n'
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
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_FIELDS = ["Open", "Close"]


def load_prices(tickers, factor_tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    key = "_".join(sorted(tickers))
    path = CACHE / f"adjusted_{start.date()}_{end.date()}_{key}.csv"
    columns = pd.MultiIndex.from_product([PRICE_FIELDS, tickers])
    if path.exists():
        panel = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    else:
        yf.set_tz_cache_location(str(CACHE / "yfinance"))
        options = {
            "start": start.date().isoformat(),
            "end": (end + pd.Timedelta(days=1)).date().isoformat(),
            "auto_adjust": True,
            "progress": False,
            "multi_level_index": True,
            "timeout": 12,
        }
        panel = yf.download(tickers, threads=4, **options).reindex(columns=columns)
        for ticker in tickers:
            fields = [(field, ticker) for field in PRICE_FIELDS]
            prices = panel[fields]
            if prices.dropna().empty or (ticker in factor_tickers and prices.isna().any().any()):
                retry = yf.download([ticker], threads=False, **options)
                panel = panel.reindex(panel.index.union(retry.index))
                panel[fields] = retry.reindex(index=panel.index, columns=fields)

    panel = panel.reindex(columns=columns).sort_index()
    panel = panel.where(np.isfinite(panel) & (panel > 0)).dropna(how="all")
    missing = [ticker for ticker in tickers if panel.xs(ticker, axis=1, level=1).dropna().empty]
    if missing:
        raise RuntimeError(f"Yahoo Finance has no adjusted open/close observations for {missing}.")
    if panel.index.has_duplicates:
        raise RuntimeError("Duplicate dates in Yahoo Finance price data.")
    if panel.index[0] > start + pd.Timedelta(days=7) or panel.index[-1] < end - pd.Timedelta(days=7):
        raise RuntimeError("Yahoo Finance price data do not span the requested date range.")
    factor_prices = panel.loc[:, pd.IndexSlice[:, factor_tickers]]
    if factor_prices.isna().any().any():
        raise RuntimeError("Missing factor ETF prices on an observed trading session.")

    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=CACHE, suffix=".csv", delete=False, mode="w") as file:
            temporary_path = Path(file.name)
            try:
                panel.to_csv(file)
                file.flush()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return panel["Open"], panel["Close"]


def slopes(x, y):
    """Column-wise OLS slopes with an intercept, allowing a shared regressor."""
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean(axis=0)
    numerator = (x_centered * y_centered).sum(axis=0)
    denominator = (x_centered**2).sum(axis=0)
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )


def long_short_weights(scores, names_per_leg, gross_exposure):
    weights = np.zeros(len(scores))
    eligible = np.flatnonzero(np.isfinite(scores))
    if len(eligible) >= 2 * names_per_leg:
        ordered = eligible[np.argsort(scores[eligible], kind="stable")]
        leg_weight = gross_exposure / (2 * names_per_leg)
        weights[ordered[-names_per_leg:]] = leg_weight
        weights[ordered[:names_per_leg]] = -leg_weight
    return weights


def formation_scores(stock_returns, factor_returns, calibration_days, quantile):
    stocks = stock_returns.to_numpy()
    factors = factor_returns.to_numpy()
    eligible = np.isfinite(stocks).all(axis=0)
    ciq = np.full(stocks.shape[1], np.nan)
    market_tail = np.full(stocks.shape[1], np.nan)
    if eligible.sum() < 2 or not np.isfinite(factors).all():
        return ciq, market_tail

    stocks = stocks[:, eligible]
    design = np.column_stack((np.ones(len(factors)), factors))
    coefficients = np.linalg.lstsq(design[:calibration_days], stocks[:calibration_days], rcond=None)[0]
    residuals = stocks - design @ coefficients
    thresholds = np.quantile(residuals[:calibration_days], quantile, axis=0)
    flags = residuals <= thresholds
    shares = (flags.sum(axis=1, keepdims=True) - flags) / (stocks.shape[1] - 1)

    # Align outcome on day s with the signed change in peer losses from s-2 to s-1.
    signed_change = shares[calibration_days - 2 : -2] - shares[calibration_days - 1 : -1]
    ciq[eligible] = slopes(signed_change, residuals[calibration_days:])

    market = factors[:, 0]
    market_cutoff = np.quantile(market[:calibration_days], quantile)
    market_tail_days = market[calibration_days:] <= market_cutoff
    if market_tail_days.sum() >= 2:
        x = market[calibration_days:][market_tail_days, None]
        y = stocks[calibration_days:][market_tail_days]
        market_tail[eligible] = slopes(x, y)
    return ciq, market_tail


def monthly_targets(stock_returns, factors, first_date, parameters):
    dates = stock_returns.index
    months = dates.to_period("M")
    lookback = parameters["lookback_sessions"]
    ciq_targets = {}
    tail_targets = {}
    for next_open in np.flatnonzero(months[1:] != months[:-1]) + 1:
        if next_open < lookback or dates[next_open] < first_date:
            continue
        formation = slice(next_open - lookback, next_open)
        ciq, tail = formation_scores(
            stock_returns.iloc[formation],
            factors.iloc[formation],
            parameters["calibration_sessions"],
            parameters["tail_quantile"],
        )
        for scores, targets in ((ciq, ciq_targets), (tail, tail_targets)):
            targets[dates[next_open]] = long_short_weights(
                scores, parameters["names_per_leg"], parameters["gross_exposure"]
            )
    return ciq_targets, tail_targets


def simulate(open_prices, close_prices, dates, targets, transaction_rate, short_daily_rate):
    """Fund rebalances at the next open and mark holdings to market each close."""
    positions = np.zeros(close_prices.shape[1])
    previous_closes = close_prices.shift(1).loc[dates].to_numpy()
    opens = open_prices.loc[dates].to_numpy()
    closes = close_prices.loc[dates].to_numpy()
    records = []
    for date, previous_close, opening, closing in zip(dates, previous_closes, opens, closes):
        held = positions != 0
        if not np.isfinite(opening[held]).all():
            raise RuntimeError(f"Missing opening price for a held position on {date.date()}.")
        open_positions = positions.copy()
        open_positions[held] *= opening[held] / previous_close[held]
        open_equity = 1.0 + (open_positions - positions).sum()
        if open_equity <= 0:
            raise RuntimeError(f"Portfolio equity depleted at the open on {date.date()}.")

        traded = np.zeros_like(positions)
        if date in targets:
            weights = targets[date]
            if transaction_rate * np.abs(open_positions).sum() >= open_equity:
                raise RuntimeError(f"Insufficient equity to rebalance on {date.date()}.")
            # Target weights use equity after paying for the actual changes in positions.
            target_equity = brentq(
                lambda equity: equity
                + transaction_rate * np.abs(weights * equity - open_positions).sum()
                - open_equity,
                0.0,
                open_equity,
            )
            target_positions = weights * target_equity
            traded = target_positions - open_positions
            open_positions = target_positions

        active = open_positions != 0
        if not (np.isfinite(opening[active]).all() and np.isfinite(closing[active]).all()):
            raise RuntimeError(f"Missing price for an active position on {date.date()}.")
        close_positions = open_positions.copy()
        close_positions[active] *= closing[active] / opening[active]
        traded_notional = np.abs(traded).sum()
        cost = transaction_rate * traded_notional
        borrow = -short_daily_rate * np.minimum(open_positions, 0).sum()
        close_equity = open_equity + (close_positions - open_positions).sum() - cost - borrow
        if not np.isfinite(close_equity) or close_equity <= 0:
            raise RuntimeError(f"Invalid closing equity on {date.date()}.")
        records.append((close_equity - 1.0, traded_notional / open_equity, (np.abs(traded) > 1e-12).sum()))
        positions = close_positions / close_equity

    return pd.DataFrame(records, index=dates, columns=["return", "turnover", "n_trades"])


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def metrics(history, sessions_per_year):
    returns = history["return"].to_numpy()
    if not len(returns):
        raise RuntimeError("No sessions in a requested reporting period.")
    equity = np.r_[1.0, np.cumprod(1.0 + returns)]
    years = len(returns) / sessions_per_year
    daily_vol = returns.std(ddof=1) if len(returns) > 1 else np.nan
    return {
        "cagr": finite_or_none(equity[-1] ** (1 / years) - 1),
        "ann_vol": finite_or_none(daily_vol * np.sqrt(sessions_per_year)),
        "sharpe": finite_or_none(returns.mean() / daily_vol * np.sqrt(sessions_per_year))
        if daily_vol > 0
        else None,
        "max_drawdown": finite_or_none((equity / np.maximum.accumulate(equity) - 1).min()),
        "total_return": finite_or_none(equity[-1] - 1),
        "n_trades": int(history["n_trades"].sum()),
        "turnover": finite_or_none(history["turnover"].sum()),
    }


def period_metrics(history, in_sample, out_of_sample, sessions_per_year):
    return {
        "is": metrics(history.loc[slice(*in_sample)], sessions_per_year),
        "oos": metrics(history.loc[slice(*out_of_sample)], sessions_per_year),
        "full": metrics(history, sessions_per_year),
    }


def print_table(portfolios):
    columns = [
        ("CAGR", "cagr", 10, ".2%"),
        ("Ann. vol", "ann_vol", 10, ".2%"),
        ("Sharpe", "sharpe", 8, ".2f"),
        ("Max DD", "max_drawdown", 10, ".2%"),
        ("Total ret.", "total_return", 11, ".2%"),
        ("Trades", "n_trades", 8, "d"),
        ("Turnover", "turnover", 10, ".2f"),
    ]
    headings = " ".join(f"{label:>{width}}" for label, _, width, _ in columns)
    print(f"{'Portfolio':<18} {'Period':<6} {headings}")
    for name, periods in portfolios.items():
        for period, row in periods.items():
            values = []
            for _, key, width, format_spec in columns:
                value = row[key]
                display = "n/a" if value is None else format(value, format_spec)
                values.append(f"{display:>{width}}")
            print(f"{name:<18} {period.upper():<6} {' '.join(values)}")


def main():
    with (ROOT / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    stocks = config["universe"]
    etfs = config["factor_etfs"]
    market = etfs["market"]
    tickers = list(dict.fromkeys([*stocks, *etfs.values()]))
    parameters = config["parameters"]
    costs = config["costs"]
    start = pd.Timestamp(config["data_start"])
    in_sample, out_of_sample = [
        tuple(pd.Timestamp(config[period][side]) for side in ("start", "end"))
        for period in ("in_sample", "out_of_sample")
    ]
    first_date, last_date = in_sample[0], out_of_sample[1]
    if not start < first_date <= in_sample[1] < out_of_sample[0] <= last_date:
        raise ValueError("Invalid date ranges in config.yaml.")
    if not 2 <= parameters["calibration_sessions"] < parameters["lookback_sessions"]:
        raise ValueError("The calibration window must leave a second scoring window.")
    if not 0 < parameters["tail_quantile"] < 0.5:
        raise ValueError("tail_quantile must lie between 0 and 0.5.")
    if not 1 <= parameters["names_per_leg"] <= len(stocks) // 2 or len(set(stocks)) != len(stocks):
        raise ValueError("The universe must contain enough unique tickers for both legs.")
    gross = parameters["gross_exposure"]
    sessions_per_year = parameters["sessions_per_year"]
    if sessions_per_year <= 0:
        raise ValueError("sessions_per_year must be positive.")
    transaction_rate = costs["transaction_bps"] / 10_000
    short_daily_rate = costs["short_borrow_bps_per_year"] / (10_000 * sessions_per_year)
    if gross <= 0 or not 0 <= transaction_rate * max(1, gross) < 1 or short_daily_rate < 0:
        raise ValueError("Invalid exposure or cost parameters in config.yaml.")

    open_prices, close_prices = load_prices(tickers, list(etfs.values()), start, last_date)
    returns = close_prices.pct_change(fill_method=None)
    factors = pd.DataFrame({
        "market": returns[market],
        "size": returns[etfs["small_cap"]] - returns[market],
        "value": returns[etfs["value"]] - returns[etfs["growth"]],
    })
    ciq_targets, tail_targets = monthly_targets(returns[stocks], factors, first_date, parameters)
    dates = close_prices.loc[first_date:last_date].index
    if dates.empty or not any(np.any(weights) for weights in ciq_targets.values()):
        raise RuntimeError("The available data cannot form a strategy portfolio in the requested period.")

    portfolios = {}
    benchmark_name = f"{market} buy & hold"
    for name, symbols, targets in (
        ("Residual tail", stocks, ciq_targets),
        ("Market tail beta", stocks, tail_targets),
        (benchmark_name, [market], {dates[0]: np.ones(1)}),
    ):
        history = simulate(
            open_prices[symbols], close_prices[symbols], dates, targets, transaction_rate, short_daily_rate
        )
        portfolios[name] = period_metrics(history, in_sample, out_of_sample, sessions_per_year)

    data_range = f"{dates[0].date()} to {dates[-1].date()}"
    print(f"Adjusted price history {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    print(f"Backtest {data_range} | IS {in_sample[0].date()} to {in_sample[1].date()} "
          f"| OOS {out_of_sample[0].date()} to {out_of_sample[1].date()}")
    print_table(portfolios)
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **portfolios["Residual tail"],
        "benchmark": {"name": benchmark_name, **portfolios[benchmark_name]},
        "market_tail_beta": portfolios["Market tail beta"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

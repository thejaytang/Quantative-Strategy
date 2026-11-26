# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 309_intangible_value_sixth_factor\ncategory: factor\n\nuniverse:\n  market: SPY\n  intangible_value: ITAN\n  conventional_value: IWD\n  growth: IWF\n\ndata:\n  start: 2021-06-28\n  end: 2026-11-20  # Inclusive. Last complete trading week before the job date.\nin_sample:\n  start: 2021-06-28\n  end: 2023-12-31\nout_of_sample:\n  start: 2024-01-01\n  end: 2026-11-20\n\nsleeve_weight: 0.20  # The remainder is invested in the market ETF.\nhedge_lookback_sessions: 126\ntransaction_cost_bps: 5  # One way, per dollar bought or sold.\ntrading_days_per_year: 252\n'
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
"""Backtest a live intangible-value ETF sleeve and an ETF-hedged spread."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def validate_prices(prices, tickers, start, end):
    prices = prices.loc[start:end, tickers]
    first_common = prices.dropna().index.min()
    if pd.isna(first_common) or first_common > start + pd.Timedelta(days=7):
        raise ValueError("ETF prices are missing near the configured start date")
    prices = prices.loc[first_common:]
    if prices.index[-1] < end - pd.Timedelta(days=4):
        raise ValueError("ETF price history ends before the configured end date")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("ETF price dates must be unique and increasing")
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Missing or invalid adjusted ETF close after the common start")
    return prices


def download_prices(cache_path, tickers, start, end):
    with TemporaryDirectory(dir=CACHE) as directory:
        yf.set_tz_cache_location(str(Path(directory) / "yfinance"))
        for attempt in range(3):
            try:
                history = yf.download(
                    tickers,
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    timeout=10,
                )
                prices = validate_prices(history["Close"], tickers, start, end)
            except (KeyError, ValueError, RuntimeError, OSError) as error:
                if attempt == 2:
                    raise RuntimeError("Unable to obtain complete Yahoo Finance ETF prices") from error
                time.sleep(2 ** attempt)
            else:
                temporary_path = Path(directory) / cache_path.name
                prices.to_csv(temporary_path, index_label="Date")
                temporary_path.replace(cache_path)
                return


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    symbols = "-".join(sorted(tickers))
    cache_path = CACHE / f"yahoo_adjusted_{symbols}_{start.date()}_{end.date()}.csv"
    if not cache_path.is_file():
        download_prices(cache_path, tickers, start, end)
    prices = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
    return validate_prices(prices, tickers, start, end)


def hedge_targets(returns, rebalance_dates, universe, lookback):
    market = universe["market"]
    intangible = universe["intangible_value"]
    value = universe["conventional_value"]
    growth = universe["growth"]
    factors = pd.DataFrame({
        "market": returns[market],
        "growth_minus_value": returns[growth] - returns[value],
    })
    targets = {}
    for date in rebalance_dates:
        end = returns.index.get_loc(date)
        if end < lookback:
            continue
        # The execution day's close-to-close return belongs to the old positions.
        window = slice(end - lookback, end)
        design = np.column_stack((np.ones(lookback), factors.iloc[window].to_numpy()))
        observations = returns[[intangible, value]].iloc[window].to_numpy()
        coefficients, _, rank, _ = np.linalg.lstsq(design, observations, rcond=None)
        if rank < design.shape[1]:
            raise ValueError(f"Hedge regressors are collinear for execution on {date.date()}")
        market_beta, style_beta = coefficients[1:, 0] - coefficients[1:, 1]
        targets[date] = {
            market: -market_beta,
            intangible: 1.0,
            value: -1.0 + style_beta,
            growth: -style_beta,
        }
    return pd.DataFrame.from_dict(targets, orient="index", columns=returns.columns).sort_index()


def simulate(returns, targets, cost_rate):
    """Mark holdings to each close, then rebalance at that close."""
    columns = ["return", "turnover", "n_trades"]
    if targets.empty:
        return pd.DataFrame(columns=columns, index=returns.index[:0])

    # Unallocated NAV is zero-yield cash, including proceeds from short positions.
    weights = np.zeros(len(returns.columns))
    targets_by_date = dict(zip(targets.index, targets.to_numpy()))
    returns = returns.loc[targets.index[0]:]
    records = []
    for row in returns.itertuples():
        date, *asset_returns = row
        asset_returns = np.asarray(asset_returns)
        growth = 1.0 + weights @ asset_returns
        if growth <= 0:
            raise ValueError(f"Nonpositive portfolio NAV on {date.date()}")
        weights = weights * (1.0 + asset_returns) / growth
        traded = np.zeros_like(weights)
        target = targets_by_date.get(date)
        if target is not None:
            # Scale the new positions to NAV after fees; fees depend on these trades.
            def nav_balance(scale):
                return scale + cost_rate * np.abs(scale * target - weights).sum() - 1.0

            scale = brentq(nav_balance, 0.0, 1.0)
            traded = np.abs(scale * target - weights)
            growth *= scale
            weights = target
        records.append((growth - 1.0, traded.sum(), int(np.count_nonzero(traded > 1e-10))))
    return pd.DataFrame(records, index=returns.index, columns=columns)


def metrics(ledger, start, end, periods_per_year):
    segment = ledger.loc[start:end]
    result = dict.fromkeys(["cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "turnover"])
    result.update(n_trades=0, n_days=len(segment), data_range=None)
    if segment.empty:
        return result

    returns = segment["return"]
    wealth = (1.0 + returns).cumprod()
    std = returns.std(ddof=1)
    values = {
        "cagr": wealth.iloc[-1] ** (periods_per_year / len(returns)) - 1.0,
        "ann_vol": std * np.sqrt(periods_per_year),
        "sharpe": returns.mean() / std * np.sqrt(periods_per_year) if std > 0 else np.nan,
        "max_drawdown": (wealth / wealth.cummax().clip(lower=1.0) - 1.0).min(),
        "total_return": wealth.iloc[-1] - 1.0,
        "turnover": segment["turnover"].sum(),
    }
    result.update({name: float(value) if np.isfinite(value) else None for name, value in values.items()})
    result.update(
        n_trades=int(segment["n_trades"].sum()),
        data_range=f"{segment.index[0].date()} to {segment.index[-1].date()}",
    )
    return result


def print_table(reports):
    print(f"{'Portfolio':<25} {'Period':<6} {'CAGR':>9} {'Ann. vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total':>9} {'Turnover':>10} {'Trades':>6} {'Days':>5}")
    for name, split in reports.items():
        for period, row in split.items():
            display = []
            for key in ("cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "turnover"):
                value = row[key]
                display.append("n/a" if value is None else format(value, ".2f" if key == "sharpe" else ".2%"))
            cagr, vol, sharpe, drawdown, total, turnover = display
            print(f"{name:<25} {period.upper():<6} {cagr:>9} {vol:>9} {sharpe:>8} "
                  f"{drawdown:>9} {total:>9} {turnover:>10} {row['n_trades']:6d} {row['n_days']:5d}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data, in_sample, out_of_sample = (config[key] for key in ("data", "in_sample", "out_of_sample"))
    start, end = pd.Timestamp(data["start"]), pd.Timestamp(data["end"])
    is_start, is_end = pd.Timestamp(in_sample["start"]), pd.Timestamp(in_sample["end"])
    oos_start, oos_end = pd.Timestamp(out_of_sample["start"]), pd.Timestamp(out_of_sample["end"])
    if not start <= is_start <= is_end < oos_start <= oos_end <= end:
        raise ValueError("IS and OOS ranges must be ordered, disjoint and within the data range")
    splits = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (start, end)}
    universe = config["universe"]
    market, intangible, value, growth = (
        universe[role] for role in ("market", "intangible_value", "conventional_value", "growth")
    )
    tickers = [market, intangible, value, growth]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Use four distinct ETF tickers")
    sleeve = float(config["sleeve_weight"])
    lookback = int(config["hedge_lookback_sessions"])
    cost_bps = float(config["transaction_cost_bps"])
    periods_per_year = int(config["trading_days_per_year"])
    if not (0 < sleeve < 1 and lookback >= 3 and 0 <= cost_bps < 10000 and periods_per_year > 0):
        raise ValueError("Invalid sleeve weight, lookback, transaction cost or annualization factor")

    prices = load_prices(tickers, start, end)
    returns = prices.pct_change().iloc[1:]
    if returns.empty:
        raise ValueError("At least two common ETF closes are required")
    months = returns.index.to_period("M")
    rebalance_dates = returns.index[:1].union(returns.index[1:][months[1:] != months[:-1]])
    strategy_name = f"{market}/{intangible} sleeve"
    value_name = f"{market}/{value} sleeve"
    market_name = f"{market} buy and hold"
    hedge_name = f"Hedged {intangible}-{value}"
    targets = {
        strategy_name: pd.DataFrame({market: 1 - sleeve, intangible: sleeve}, index=rebalance_dates),
        value_name: pd.DataFrame({market: 1 - sleeve, value: sleeve}, index=rebalance_dates),
        market_name: pd.DataFrame({market: 1.0}, index=rebalance_dates[:1]),
        hedge_name: hedge_targets(returns, rebalance_dates, universe, lookback),
    }
    reports = {}
    for name, weights in targets.items():
        weights = weights.reindex(columns=tickers, fill_value=0.0)
        ledger = simulate(returns, weights, cost_bps / 10000.0)
        reports[name] = {
            period: metrics(ledger, left, right, periods_per_year)
            for period, (left, right) in splits.items()
        }

    print(f"Adjusted closes: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"IS: {is_start.date()} to {is_end.date()} | OOS: {oos_start.date()} to {oos_end.date()}")
    print(f"One-way cost: {cost_bps:g} bps | Hedge evaluated: {reports[hedge_name]['full']['data_range'] or 'n/a'}")
    print_table(reports)
    strategy = reports[strategy_name]
    benchmark = {"name": value_name, **reports[value_name]}
    benchmark["buy_and_hold"] = {"name": market_name, **reports[market_name]}
    results = {
        "strategy": config["strategy"],
        "data_range": strategy["full"]["data_range"],
        **{period: {**row, "hedged_spread": reports[hedge_name][period]} for period, row in strategy.items()},
        "benchmark": benchmark,
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

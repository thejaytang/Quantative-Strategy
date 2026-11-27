# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 187_midterm_cycle\ncategory: seasonality\n\nuniverse:\n  equity: SPY\n  cash_proxy: BIL\n\nin_sample:\n  start: 2008-01-01\n  end: 2015-12-31\nout_of_sample:\n  start: 2016-01-01\n  end: 2025-12-31\n\n# Year modulo 4 is 3 in the calendar year following a U.S. midterm.\npost_midterm_year_mod_4: 3\nstatic_equity_weight: 0.25\n# Charged on each dollar bought or sold, including the BIL leg.\ntransaction_cost_bps: 5.0\nslippage_bps: 1.0\n\nprice_snapshot:\n  sha256: 26b5614e1ac5ff754e4440067087aa082f73c556ddf5c548d4fdf93efefae98e\n  # Original snapshot write time, immediately after the yfinance download (Europe/Oslo).\n  fetched_at: "2026-11-25T12:18:05+01:00"\n'
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
"""Backtest the U.S. post-midterm calendar rule using adjusted ETF opens."""

import hashlib
import json
import math
from datetime import datetime
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
YEAR_TYPES = ("Presidential election", "First year", "Midterm election", "Post-midterm")


def load_opens(tickers, start, end, snapshot):
    """Load aligned adjusted opens, including the two valuation boundary sessions."""
    padding = pd.Timedelta(days=10)
    fetch_start, fetch_end = start - padding, end + padding
    cache_file = CACHE / f"adjusted_open_{fetch_start.date()}_{fetch_end.date()}.csv"
    if cache_file.is_file():
        contents = cache_file.read_bytes()
        if hashlib.sha256(contents).hexdigest() == snapshot["sha256"]:
            prices = pd.read_csv(BytesIO(contents), index_col="date", parse_dates=["date"])
            if tuple(prices.columns) == tickers:
                validate_opens(prices, start, end)
                return prices, snapshot["fetched_at"], "bundled snapshot"

    bars = []
    with TemporaryDirectory(dir=ROOT, prefix=".yfinance-") as cache_dir:
        yf.set_tz_cache_location(cache_dir)
        for ticker in tickers:
            history = yf.Ticker(ticker).history(
                start=fetch_start.date().isoformat(), end=fetch_end.date().isoformat(),
                auto_adjust=True, timeout=20,
            )
            if history.empty:
                raise RuntimeError(f"Yahoo Finance returned no daily bars for {ticker}.")
            opens = history["Open"].rename(ticker)
            opens.index = opens.index.tz_localize(None).normalize()
            bars.append(opens)
        fetched_at = datetime.now(ZoneInfo("Europe/Oslo")).isoformat(timespec="seconds")

    prices = pd.concat(bars, axis=1).sort_index()
    validate_opens(prices, start, end)
    return prices, fetched_at, "fresh download"


def validate_opens(prices, start, end):
    if (
        prices.empty or prices.index.has_duplicates or not prices.index.is_monotonic_increasing
        or not np.isfinite(prices).to_numpy().all() or (prices <= 0).to_numpy().any()
        or prices.index[0] >= start or prices.index[-1] <= end
    ):
        raise ValueError("Need complete, aligned, positive adjusted opens and both boundary sessions.")


def simulate(equity_weight, returns, cost_rate):
    targets = pd.DataFrame(
        {returns.columns[0]: equity_weight, returns.columns[1]: 1.0 - equity_weight},
        index=returns.index,
    )
    drifted = targets.shift(1) * (1.0 + returns.shift(1))
    pretrade = drifted.div(drifted.sum(axis=1), axis=0).fillna(0.0)

    # For two fully invested assets a rebalance has one buy and one sell.
    # Solve fee = rate * sum(abs(target * (1 - fee) - pretrade)), also from initial cash.
    change_before_fee = targets - pretrade
    signed_target = (np.sign(change_before_fee) * targets).sum(axis=1)
    fees = cost_rate * change_before_fee.abs().sum(axis=1) / (1.0 + cost_rate * signed_target)
    traded = (targets.mul(1.0 - fees, axis=0) - pretrade).abs()
    return pd.DataFrame({
        "return": (1.0 - fees) * (1.0 + (targets * returns).sum(axis=1)) - 1.0,
        "turnover": traded.sum(axis=1),
        "n_trades": (traded > 1e-10).sum(axis=1),
        "equity_weight": targets.iloc[:, 0],
    })


def annualized_sharpe(returns):
    std = returns.std(ddof=1)
    return returns.mean() / std * math.sqrt(252) if std > 0 else None


def metrics(portfolio, next_open, cash_returns):
    returns = portfolio["return"]
    nav = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    years = (next_open.loc[returns.index[-1]] - returns.index[0]).days / 365.2425
    std = returns.std(ddof=1)
    values = {
        "cagr": nav[-1] ** (1.0 / years) - 1.0,
        "ann_vol": std * math.sqrt(252),
        "sharpe": annualized_sharpe(returns),
        "sharpe_excess": annualized_sharpe(returns - cash_returns),
        "max_drawdown": (nav / np.maximum.accumulate(nav) - 1.0).min(),
        "total_return": nav[-1] - 1.0,
        "turnover": portfolio["turnover"].sum(),
        "avg_equity_weight": portfolio["equity_weight"].mean(),
    }
    finite = {key: float(value) if value is not None and math.isfinite(value) else None
              for key, value in values.items()}
    return {**finite, "n_trades": int(portfolio["n_trades"].sum())}


def fmt(value, spec):
    return "—" if value is None else format(value, spec)


def print_metrics(report):
    print(f"{'Portfolio':<22} {'Split':<5} {'CAGR':>9} {'Ann. vol':>9} "
          f"{'Excess SR':>9} {'Zero SR':>8} "
          f"{'Max DD':>9} {'Total':>10} {'Trades':>7} {'Turnover':>9} {'Equity':>8}")
    for name, splits in report.items():
        for split, row in splits.items():
            print(f"{name:<22} {split.upper():<5} {fmt(row['cagr'], '.2%'):>9} "
                  f"{fmt(row['ann_vol'], '.2%'):>9} {fmt(row['sharpe_excess'], '.2f'):>9} "
                  f"{fmt(row['sharpe'], '.2f'):>8} "
                  f"{fmt(row['max_drawdown'], '.2%'):>9} {fmt(row['total_return'], '.2%'):>10} "
                  f"{row['n_trades']:>7d} {fmt(row['turnover'], '.2f'):>8}x "
                  f"{fmt(row['avg_equity_weight'], '.2%'):>8}")


def print_diagnostics(periods, portfolios, equity_returns, post_midterm):
    start, end = periods["full"]
    annual = (1.0 + equity_returns).groupby(equity_returns.index.year).prod() - 1.0
    annual = annual.loc[
        start.year + int((start.month, start.day) != (1, 1)):
        end.year - int((end.month, end.day) != (12, 31))
    ]
    print("\nSPY adjusted first-open to first-open returns by year type, before costs")
    print(f"{'Year type':<23} {'Years':>6} {'Mean':>9} {'Median':>9}")
    for phase, label in enumerate(YEAR_TYPES):
        group = annual.loc[annual.index % 4 == phase]
        mean = group.mean() if len(group) else None
        median = group.median() if len(group) else None
        print(f"{label:<23} {len(group):>6d} {fmt(mean, '.2%'):>9} {fmt(median, '.2%'):>9}")

    print("\nObserved election cycles, net cumulative returns")
    print(f"{'Years':<11} {'Coverage':<8} {'Strategy':>10} {'Static mix':>12} {'SPY hold':>10}")
    complete_cycles = []
    for year in range(start.year - start.year % 4, end.year + 1, 4):
        first, last = pd.Timestamp(year, 1, 1), pd.Timestamp(year + 3, 12, 31)
        complete = start <= first and last <= end
        log_returns = [np.log1p(p.loc[first:last, "return"]).sum() for p in portfolios]
        strategy, static, buy_hold = np.expm1(log_returns)
        print(f"{year}-{year + 3}   {'Full' if complete else 'Partial':<8} "
              f"{strategy:>10.2%} {static:>12.2%} {buy_hold:>10.2%}")
        if complete:
            complete_cycles.append((first, last))

    print("\nCycle coverage (descriptive; too few complete cycles for reliable inference)")
    for split, (first, last) in periods.items():
        count = sum(first <= a and b <= last for a, b in complete_cycles)
        holding_years = [str(year) for year in equity_returns.loc[first:last].index.year.unique()
                         if year % 4 == post_midterm]
        print(f"{split.upper():<5} {count} complete cycles | SPY holding years "
              f"{', '.join(holding_years) or 'none'}")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not (is_start <= is_end and is_end + pd.Timedelta(days=1) == oos_start <= oos_end):
        raise ValueError("Use ordered, contiguous IS and OOS date ranges.")
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    tickers = (config["universe"]["equity"], config["universe"]["cash_proxy"])
    fee_bps = config["transaction_cost_bps"]
    slippage_bps = config["slippage_bps"]
    cost = (fee_bps + slippage_bps) / 10000.0
    equity_weight = config["static_equity_weight"]
    post_midterm = config["post_midterm_year_mod_4"]
    if not (fee_bps >= 0 and slippage_bps >= 0 and cost < 1):
        raise ValueError("Use nonnegative transaction costs and slippage with total below 100%.")
    if not (0 <= equity_weight <= 1 and post_midterm in range(4) and tickers[0] != tickers[1]):
        raise ValueError("Use a valid equity weight and election-cycle year, and distinct tickers.")

    opens, fetched_at, price_source = load_opens(tickers, is_start, oos_end, config["price_snapshot"])
    next_open = opens.index.to_series().shift(-1)
    returns = (opens.shift(-1) / opens - 1.0).loc[is_start:oos_end]
    if any(returns.loc[first:last].empty for first, last in periods.values()):
        raise ValueError("Each split must contain at least one trading session.")

    # Today's close signal uses only the known calendar of tomorrow's session.
    # Shifting the signal executes it at that next session's open.
    signal_at_close = next_open.dt.year.mod(4).eq(post_midterm).astype(float)
    equity_at_open = signal_at_close.shift(1).loc[returns.index]
    portfolios = {
        "Midterm cycle": simulate(equity_at_open, returns, cost),
        f"Static {equity_weight:.0%} {tickers[0]}": simulate(equity_weight, returns, cost),
        f"{tickers[0]} buy & hold": simulate(1.0, returns, cost),
    }
    cash_returns = returns[tickers[1]]
    report = {
        name: {split: metrics(portfolio.loc[first:last], next_open, cash_returns.loc[first:last])
               for split, (first, last) in periods.items()}
        for name, portfolio in portfolios.items()
    }
    actual_start, last_session = returns.index[[0, -1]]
    data_range = f"{actual_start.date()} to {last_session.date()}"
    print(f"{config['strategy']} | holding sessions {data_range} | "
          f"{fee_bps:g} bps fees + {slippage_bps:g} bp slippage per leg")
    print(f"Final valuation open {next_open.loc[last_session].date()}")
    print(f"DATA_SOURCE: Yahoo Finance via yfinance ({price_source})")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted opens")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"Excess SR uses daily net portfolio returns minus {tickers[1]} gross open-to-open returns.")
    print_metrics(report)
    print_diagnostics(periods, list(portfolios.values()), returns[tickers[0]], post_midterm)

    strategy, static, _ = report.values()
    output = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **strategy,
        "benchmark": {
            "ticker": tickers[0],
            "equity_weight": equity_weight,
            "cash_proxy": tickers[1],
            **static["full"],
            **static,
        },
    }
    print("RESULTS_JSON: " + json.dumps(output, allow_nan=False))


if __name__ == "__main__":
    main()

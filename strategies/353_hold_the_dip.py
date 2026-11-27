"""353_hold_the_dip
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/353_hold_the_dip.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 92e75bea82090bd8b6723b610a02dabec59e659b on 2026-11-27 07:29 Europe/Oslo, exit 0, output sha256 e7bf88552bb0b26251fdc3026efe7c1445c0703e834f888db232f8d7f88e59c0
Data source: Yahoo Finance via yfinance, daily adjusted ETF closes for SPY (equity) and BIL (cash proxy). The loader uses a dated CSV snapshot in data_cache/ when present and downloads on a cache miss. The run reports a snapshot fetch timestamp of 2026-11-27T05:47:18+00:00.
Data fetched: 2026-11-27T06:29:51+00:00
Sample period: 2008-07-01 through 2026-11-25, inclusive, for both the configured evaluation and the log's actual data range. Earlier closes initialize the signal.   data_range (from output): {'start': '2008-07-01', 'end': '2026-11-25'}
IS/OOS split: IS runs through 2025-12-01. OOS starts on 2025-12-02 and runs through 2026-11-25. The same parameters and continuous simulated portfolio span both periods, with the IS closing position carried into OOS. The README describes the OOS period as a retrospective historical evaluation of the selected rule.
Costs and slippage: Each buy or sell charges 1 bp in transaction costs and 2 bps in modeled fixed slippage, applied together as 3 bps of each leg's executed notional. The initial purchase is one leg. An ETF switch sells the old holding and buys the new one with the remaining proceeds after fees. Holding days incur no charge. The passive SPY benchmark pays the same initial purchase fee, and the simulation values holdings at the final close without forcing liquidation.
Signal lag: The signal compares SPY's adjusted close at session t with the adjusted close on or before the date 12 calendar months earlier. The resulting target is filled at the next session's close (t+1). The old position earns the close-to-close return ending at t+1, and the new position first earns the return ending at t+2. The first evaluation close uses a target initialized from earlier data.
Price adjustment: yfinance auto_adjust=True total-return closes
Survivorship bias: The fixed universe uses the two ETFs' historical fund prices and does not reconstruct historical returns from today's index constituents. The retrospective selection of two surviving ETFs retains fund and market selection bias.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09419677454492215, ann_vol=0.1416875996463747, sharpe=0.7075975878691806, sharpe_excess=0.6255480379407973, max_drawdown=-0.2771256477673928, total_return=3.7971828068039013, turnover=7.518456110447595, n_trades=131
    oos: cagr=0.06608402047291162, ann_vol=0.1498880140403393, sharpe=0.5011535022983999, sharpe_excess=0.25883823919019355, max_drawdown=-0.11459526655375618, total_return=0.06491866654598843, turnover=0.0, n_trades=0
    full: cagr=0.09267657072001478, ann_vol=0.14212221402788594, sharpe=0.6959605518103854, sharpe_excess=0.6048300600614466, max_drawdown=-0.2771256477673928, total_return=4.108609517798955, turnover=7.116860255120905, n_trades=131
    benchmark: cagr=0.11792002594575579, ann_vol=0.19744640019426357, sharpe=0.6642171826134416, sharpe_excess=0.5968212877351021, max_drawdown=-0.47165210814843506, total_return=6.777299986898728, turnover=0.05432717752000691, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 353_hold_the_dip\nuniverse:\n  equity: SPY\n  cash_proxy: BIL\nstart_date: "2008-07-01"\nend_date: "2026-11-25"\noos_start: "2025-12-02"\nlookback_months: 12\nsignal_lag_sessions: 1\ncost_bps: 1.0\nslippage_bps: 2.0\n'
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
"""Run the C004 equity trend allocation against passive equity exposure."""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PARAM_COMBOS_TRIED = 1


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    cache_path = CACHE / f"{'_'.join(tickers)}_{start.date()}_{end.date()}_adjusted.csv"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as cache_file:
            fetched_at = cache_file.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cache_file, index_col="Date", parse_dates=True)
        return validate_prices(prices, tickers), fetched_at

    with TemporaryDirectory(prefix="yfinance_", dir=CACHE) as download_cache:
        yf.set_tz_cache_location(download_cache)
        download = yf.download(
            tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
            auto_adjust=True, progress=False, threads=False, timeout=15,
        )
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if download.empty:
        raise RuntimeError("Yahoo Finance returned no data")
    prices = validate_prices(download["Close"], tickers)
    if end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError(f"Price data ends on {prices.index[-1].date()} before {end.date()}")

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as cache_file:
        temp_path = Path(cache_file.name)
        try:
            cache_file.write(f"{fetched_at}\n")
            prices.to_csv(cache_file, index_label="Date")
            cache_file.close()
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def validate_prices(prices, tickers):
    prices = prices.loc[:, tickers].sort_index()
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Expected daily adjusted closes indexed by date")
    if prices.index.hasnans or not prices.index.is_unique:
        raise ValueError("Adjusted close data contains missing or duplicate dates")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Missing, nonpositive, or nonfinite adjusted close")
    return prices


def equity_target(equity_close, lookback_months, lag_sessions):
    reference_dates = equity_close.index - pd.DateOffset(months=lookback_months)
    reference_close = equity_close.reindex(reference_dates, method="ffill").set_axis(equity_close.index)
    close_signal = equity_close.gt(reference_close).where(reference_close.notna())
    return close_signal.shift(lag_sessions)


def simulate(target, daily_returns, equity, cash, fee_rate):
    # Returns ending at today's close belong to the position set at the prior close.
    prior_target = target.shift(fill_value=False)
    held_return = daily_returns[equity].where(prior_target, daily_returns[cash]).copy()
    held_return.iloc[0] = 0.0

    # Initial entry buys one leg. Every switch sells the old leg and buys the new one.
    trades = target.ne(prior_target).astype(int) * 2
    trades.iloc[0] = 1
    sell_fraction = trades.eq(2).astype(float)
    buy_fraction = (trades.gt(0).astype(float) - fee_rate * sell_fraction) / (1 + fee_rate)
    turnover = sell_fraction + buy_fraction
    net_return = (1 + held_return) * (1 - fee_rate * turnover) - 1
    return pd.DataFrame({"return": net_return, "turnover": turnover, "trades": trades})


def metrics(portfolio, cash_returns, start_close):
    returns = portfolio["return"]
    nav = (1 + returns).cumprod()
    years = (nav.index[-1] - start_close).days / 365.2425
    annualizer = math.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std()
    excess = returns - cash_returns
    excess_vol = excess.std()
    return {
        "cagr": float(nav.iloc[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() / vol * annualizer) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float((nav / nav.cummax().clip(lower=1) - 1).min()),
        "total_return": float(nav.iloc[-1] - 1),
        "turnover": float(portfolio["turnover"].sum() / years),
        "n_trades": int(portfolio["trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    equity, cash = (config["universe"][asset] for asset in ("equity", "cash_proxy"))
    if equity == cash or any(not re.fullmatch(r"[A-Z]{1,5}", ticker) for ticker in (equity, cash)):
        raise ValueError("Use two distinct US ETF ticker symbols")
    start, end, split = (pd.Timestamp(config[key]) for key in ("start_date", "end_date", "oos_start"))
    if not start < split < end:
        raise ValueError("Expected start_date < oos_start < end_date")
    lookback = config["lookback_months"]
    lag = config["signal_lag_sessions"]
    if any(type(value) is not int or value < 1 for value in (lookback, lag)):
        raise ValueError("Both the lookback and execution lag must be positive integers")
    costs = [float(config[key]) for key in ("cost_bps", "slippage_bps")]
    fee_rate = sum(costs) / 10_000
    if not all(math.isfinite(cost) and cost >= 0 for cost in costs) or fee_rate >= 1:
        raise ValueError("Costs and slippage must be nonnegative and total less than 10,000 bps")

    warmup_start = start - pd.DateOffset(months=lookback) - pd.Timedelta(days=14 + 3 * lag)
    prices, fetched_at = load_prices([equity, cash], warmup_start, end)
    returns = prices.pct_change().loc[start:end]
    target = equity_target(prices[equity], lookback, lag).reindex(returns.index)
    if len(target) < 2 or target.isna().any():
        raise ValueError("Insufficient prices to initialize the lagged trend signal")
    if (target.index[0] - start > pd.Timedelta(days=7)
            or end - target.index[-1] > pd.Timedelta(days=7)):
        raise ValueError("Price data does not cover the requested sample")

    target = target.astype(bool)
    portfolio = simulate(target, returns, equity, cash, fee_rate)
    passive_target = pd.Series(True, index=target.index)
    benchmark = simulate(passive_target, returns, equity, cash, fee_rate)
    cash_returns = returns[cash].copy()
    cash_returns.iloc[0] = 0.0

    in_sample = target.index < split
    if in_sample.sum() < 2 or (~in_sample).sum() < 2:
        raise ValueError("IS and OOS must each contain at least two sessions")
    start_close = target.index[0]
    oos_base_close = target.index[in_sample][-1]
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(start_close.date()), "end": str(target.index[-1].date())},
        "is": metrics(portfolio.loc[in_sample], cash_returns.loc[in_sample], start_close),
        "oos": metrics(portfolio.loc[~in_sample], cash_returns.loc[~in_sample], oos_base_close),
        "full": metrics(portfolio, cash_returns, start_close),
        "benchmark": metrics(benchmark, cash_returns, start_close),
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True total-return closes")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
    print(f"BENCHMARK ({equity} buy and hold, full): {json.dumps(results['benchmark'], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
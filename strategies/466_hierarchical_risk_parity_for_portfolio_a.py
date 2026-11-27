"""466_hierarchical_risk_parity_for_portfolio_a
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/466_hierarchical_risk_parity_for_portfolio_a.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit f8add1176d5b1caebb66b01079720314ca29fcc4 on 2026-11-27 08:40 Europe/Oslo, exit 0, output sha256 b3e7307cf3a93a9d495b83da9820b75bc2d0bd29d6e783749159dc8a8045e3df
Data source: Yahoo Finance via yfinance, using daily USD ETF closes. ECH, COLO and EPU are the allocation universe. The README documents that Yahoo's COLO series carries the earlier GXG history. BIL is used only as the cash proxy for the excess Sharpe calculation.
Data fetched: 2026-11-27T07:40:04+00:00
Sample period: Evaluated from 2019-07-01 through 2025-06-30 on available sessions, as recorded in the run log. The download request starts on 2018-06-01 to provide warm-up observations.   data_range (from output): 2019-07-01 to 2025-06-30
IS/OOS split: IS is 2019-07-01 through 2022-06-30. OOS is 2022-07-01 through 2025-06-30. The code simulates one continuous path and slices its daily results at the split, carrying positions across it. The same fixed parameters apply throughout.
Costs and slippage: The simulator charges 5 bps in costs plus 2 bps in modeled slippage per buy or sell, a combined 7 bps on executed notional. It deducts this charge at each rebalance using the sum of absolute position changes from drifted holdings to funded target holdings. The charge applies to the initial allocation from cash and to the equal-weight benchmark.
Signal lag: The rolling return window includes session t's close. Its target is executed at the next session's close, t+1. Existing holdings earn the move into that close, and new holdings first earn returns on the subsequent close-to-close interval.
Price adjustment: yfinance auto_adjust total-return closes
Survivorship bias: The fixed universe consists of three ETFs selected with full histories available retrospectively. The README identifies survivorship and selection bias and says these proxies omit the original study's individual securities and historical index constituent changes.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.09377471684092364, ann_vol=0.2679966866039492, sharpe=-0.23093051813551205, sharpe_excess=-0.24723875433900963, max_drawdown=-0.5022436035062043, total_return=-0.2562235012496161, turnover=4.166770497298774, n_trades=105
    oos: cagr=0.1751330329048688, ann_vol=0.19880171983837897, sharpe=0.9169327754395165, sharpe_excess=0.6936858660292082, max_drawdown=-0.1843974706933894, total_return=0.6229808436176614, turnover=2.368894551125467, n_trades=102
    full: cagr=0.03183479389926003, ann_vol=0.2361328416744233, sharpe=0.25288239015706293, sharpe_excess=0.1501509469457215, max_drawdown=-0.5022436035062043, total_return=0.20713500940488427, turnover=6.535665048424241, n_trades=207
    benchmark: cagr=0.03960770973558647, ann_vol=0.23903729724910497, sharpe=0.28429312471087587, sharpe_excess=0.1828004122723826, max_drawdown=-0.5226835426748071, total_return=0.2627997742808954, turnover=2.9808942525743305, n_trades=207
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '466_hierarchical_risk_parity_for_portfolio_a'\nuniverse: [ECH, COLO, EPU]\nrisk_free_ticker: BIL\ndates:\n  download_start: '2018-06-01'\n  start: '2019-07-01'\n  oos_start: '2022-07-01'\n  end: '2025-06-30'\nlookback_sessions: 260\nrebalance_every_sessions: 22\nlinkage: single\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 2\n"
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
"""Rolling hierarchical risk parity for three NUAM equity market proxies."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import brentq
from scipy.spatial.distance import squareform


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def validate_prices(prices, symbols):
    prices = prices.reindex(columns=symbols).sort_index()
    if prices.empty or not prices.index.is_unique or prices.index.hasnans:
        raise ValueError("Expected nonempty adjusted closes with unique session dates")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        counts = invalid.sum()
        raise ValueError(f"Missing or invalid adjusted closes: {counts[counts > 0].to_dict()}")
    return prices


def load_prices(symbols, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    request_key = json.dumps(["yfinance-adjusted-close", sorted(symbols), start, end])
    cache_name = hashlib.sha256(request_key.encode()).hexdigest()[:16]
    path = CACHE_DIR / f"adjusted_close_{cache_name}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            fetched_at = datetime.fromisoformat(handle.readline().strip()).isoformat()
            prices = pd.read_csv(handle, index_col="date", parse_dates=["date"], float_precision="round_trip")
        return validate_prices(prices, symbols), fetched_at

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    with TemporaryDirectory(prefix=".yfinance-", dir=CACHE_DIR) as session_cache:
        yf.set_tz_cache_location(session_cache)
        bars = yf.download(
            symbols, start=start, end=end_exclusive, interval="1d", auto_adjust=True,
            progress=False, timeout=15, multi_level_index=True,
        )
    if bars.empty or "Close" not in bars.columns.get_level_values(0):
        raise RuntimeError("Yahoo Finance returned no daily close data")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prices = validate_prices(bars["Close"], symbols)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(fetched_at + "\n")
            prices.to_csv(handle, index_label="date")
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return prices, fetched_at


def cluster_variance(covariance, members):
    block = covariance[np.ix_(members, members)]
    inverse_variance = 1.0 / np.diag(block)
    weights = inverse_variance / inverse_variance.sum()
    return max(0.0, float(weights @ block @ weights))


def hrp_weights(log_returns, method):
    covariance = log_returns.cov().to_numpy()
    correlations = log_returns.corr().to_numpy()
    if not np.isfinite(correlations).all() or (np.diag(covariance) <= 0).any():
        raise ValueError("HRP needs positive asset variance and finite pairwise correlations")
    distances = np.sqrt((1.0 - np.clip(correlations, -1.0, 1.0)) / 2.0)
    np.fill_diagonal(distances, 0.0)
    ordered = leaves_list(linkage(squareform(distances), method=method))

    def allocate(members, share):
        if len(members) == 1:
            weights[members[0]] = share
            return
        midpoint = len(members) // 2
        left, right = members[:midpoint], members[midpoint:]
        left_variance = cluster_variance(covariance, left)
        right_variance = cluster_variance(covariance, right)
        combined = left_variance + right_variance
        left_share = right_variance / combined if combined > 0 else 0.5
        allocate(left, share * left_share)
        allocate(right, share * (1.0 - left_share))

    weights = np.empty(len(ordered))
    allocate(ordered, 1.0)
    return weights


def close_targets(closes, first_session, config):
    lookback = config["lookback_sessions"]
    lag = config["signal_lag_sessions"]
    first_signal = first_session - 1
    if first_signal < lookback:
        raise ValueError("Insufficient price history for the first signal; move download_start earlier")

    log_returns = np.log(closes).diff()
    targets = {}
    for signal in range(first_signal, len(closes) - lag, config["rebalance_every_sessions"]):
        window = log_returns.iloc[signal - lookback + 1:signal + 1]
        targets[closes.index[signal + lag]] = hrp_weights(window, config["linkage"])
    if not targets:
        raise ValueError("No signal can execute within the evaluation dates")
    return targets


def simulate(asset_returns, targets, fee_rate):
    weights = np.zeros(asset_returns.shape[1])
    records = []
    for date, asset_move in zip(asset_returns.index, asset_returns.to_numpy()):
        # The old positions earn the move into the execution close.
        gross = 1.0 + weights @ asset_move
        weights = weights * (1.0 + asset_move) / gross
        turnover = 0.0
        trades = 0
        net_fraction = 1.0
        if date in targets:
            target = targets[date]
            # The target weights are measured after paying costs on executed notional.
            def funding_gap(fraction):
                traded = np.abs(fraction * target - weights).sum()
                return fraction + fee_rate * traded - 1.0

            net_fraction = brentq(funding_gap, 0.0, 1.0)
            traded_amounts = np.abs(net_fraction * target - weights)
            turnover = float(traded_amounts.sum())
            trades = int(np.count_nonzero(traded_amounts > 1e-12))
            weights = target
        records.append((gross * net_fraction - 1.0, turnover, trades))
    return pd.DataFrame(records, index=asset_returns.index, columns=["return", "turnover", "n_trades"])


def annualized_sharpe(returns):
    volatility = returns.std(ddof=1)
    return float(returns.mean() / volatility * np.sqrt(SESSIONS_PER_YEAR)) if volatility > 0 else None


def metrics(daily, cash_returns, closes_index):
    returns = daily["return"]
    nav = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    prior_close = closes_index[closes_index.get_loc(daily.index[0]) - 1]
    elapsed_years = (daily.index[-1] - prior_close).days / 365.2425
    return {
        "cagr": float(nav[-1] ** (1.0 / elapsed_years) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": annualized_sharpe(returns),
        "sharpe_excess": annualized_sharpe(returns - cash_returns.loc[daily.index]),
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1.0).min()),
        "total_return": float(nav[-1] - 1.0),
        "turnover": float(daily["turnover"].sum()),
        "n_trades": int(daily["n_trades"].sum()),
    }


def period_reports(daily, cash_returns, closes_index, oos_start):
    before_split = daily.index < oos_start
    periods = {"is": daily[before_split], "oos": daily[~before_split], "full": daily}
    if min(len(period) for period in periods.values()) < 2:
        raise ValueError("Both IS and OOS need at least two daily observations")
    return {label: metrics(period, cash_returns, closes_index) for label, period in periods.items()}


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    cash_symbol = config["risk_free_ticker"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or cash_symbol in universe:
        raise ValueError("Use distinct allocation symbols and a separate cash proxy")
    for key, minimum in [("lookback_sessions", 2), ("rebalance_every_sessions", 1), ("signal_lag_sessions", 1)]:
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    cost_bps, slippage_bps = config["cost_bps"], config["slippage_bps"]
    fee_rate = (cost_bps + slippage_bps) / 10_000
    if not (cost_bps >= 0 and slippage_bps >= 0 and fee_rate < 1):
        raise ValueError("Costs and slippage must be nonnegative and below 100% in total")
    dates = config["dates"]
    start, oos_start, end = map(pd.Timestamp, (dates["start"], dates["oos_start"], dates["end"]))
    if not pd.Timestamp(dates["download_start"]) < start < oos_start <= end:
        raise ValueError("Expected download_start < start < oos_start <= end")

    closes, fetched_at = load_prices([*universe, cash_symbol], dates["download_start"], dates["end"])
    first_session = closes.index.searchsorted(start)
    targets = close_targets(closes[universe], first_session, config)
    returns = closes.pct_change(fill_method=None).iloc[first_session:]
    equal_weight = np.full(len(universe), 1.0 / len(universe))
    benchmark_targets = dict.fromkeys(targets, equal_weight)
    strategy_daily = simulate(returns[universe], targets, fee_rate)
    benchmark_daily = simulate(returns[universe], benchmark_targets, fee_rate)
    cash_returns = returns[cash_symbol]
    strategy_reports = period_reports(strategy_daily, cash_returns, closes.index, oos_start)
    benchmark_reports = period_reports(benchmark_daily, cash_returns, closes.index, oos_start)

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return closes")
    print("PARAM_COMBOS_TRIED: 1")
    for label in strategy_reports:
        print(f"{label.upper()}: {json.dumps(strategy_reports[label], allow_nan=False)}")
        print(f"BENCHMARK_EQUAL_WEIGHT_{label.upper()}: {json.dumps(benchmark_reports[label], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": f"{returns.index[0].date()} to {returns.index[-1].date()}",
        **strategy_reports,
        "benchmark": benchmark_reports["full"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

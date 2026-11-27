"""449_the_mathematics_of_heuristic_portfolio_o
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/449_the_mathematics_of_heuristic_portfolio_o.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 6df04a4d89c55b7a1b162dbdcff0dafc1e2bd682 on 2026-11-27 08:11 Europe/Oslo, exit 0, output sha256 885548188ab704fe0e513e8d022c57b5ca7c7b46ead4c130f782f352c24c6859
Data source: Yahoo Finance free daily ETF prices via yfinance. The allocation universe is SPY, EFA, EEM, IEF, TLT, LQD, GLD, and VNQ. BIL supplies the daily return reference for excess Sharpe. The loader uses a local price cache when available.
Data fetched: 2026-11-27T07:11:21+00:00
Sample period: The reported backtest runs from 2010-01-04 through 2025-12-31. The price request starts on 2007-06-01 to supply pre-sample warm-up data.   data_range (from output): 2010-01-04 to 2025-12-31
IS/OOS split: IS is 2010-01-04 through 2017-12-29 inclusive. OOS is 2018-01-02 through 2025-12-31. HRP selects 126 sessions from candidates of 126 and 252 using IS net excess Sharpe, with ties favoring the longer window. The full simulation uses the selected lookback, and the reporting split carries holdings and rolling history into OOS.
Costs and slippage: Slippage is modelled as a fixed 3 bps charge, combined with a 2 bps fee for 5 bps of each simulated buy or sell amount, including initial entry. At the execution close, the code solves for post-cost NAV so it equals pre-trade NAV less 5 bps of the sum of absolute position changes. It records that traded amount as a fraction of pre-trade NAV for turnover.
Signal lag: A signal uses daily close-to-close returns through the last trading session of the month, t. The target is simulated at the close of the next trading session, t+1. Holdings before the rebalance earn the return from t to t+1. The new weights first earn the close-to-close return from t+1 to t+2.
Price adjustment: yfinance auto_adjust total-return (USD adjusted Close)
Survivorship bias: The fixed ETF universe was chosen retrospectively from known funds with continuous sample histories. The README identifies survivorship and asset-selection bias and describes the OOS split as retrospective.
Parameter combinations tried: 5
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.06165074779018975, ann_vol=0.052452854091570876, sharpe=1.1669154019400765, sharpe_excess=1.1566285661956017, max_drawdown=-0.08589944175475428, total_return=0.6126646058723633, turnover=10.36994727368969, n_trades=768
    oos: cagr=0.030080689628168678, ann_vol=0.07914410399687286, sharpe=0.41409848219896006, sharpe_excess=0.10723292085949285, max_drawdown=-0.23376744920253145, total_return=0.26681904622397146, turnover=10.129412383810028, n_trades=768
    full: cagr=0.04575443705311044, ann_vol=0.06712938486833833, sharpe=0.7001079284400838, sharpe_excess=0.5155618425922752, max_drawdown=-0.23376744920253134, total_return=1.04295423789039, turnover=20.499359657499717, n_trades=1536
    benchmark  full: cagr=0.06968761681401348, ann_vol=0.0948018465305901, sharpe=0.7581750653624204, sharpe_excess=0.6271636813611334, max_drawdown=-0.2443410668951388, total_return=1.9321264746114903, turnover=5.7203723989433914, n_trades=1536
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "449_ the_mathematics_of_heuristic_portfolio_o"\nuniverse: [SPY, EFA, EEM, IEF, TLT, LQD, GLD, VNQ]\nrisk_free_ticker: BIL\n\ndata_start: 2007-06-01\nstart: 2010-01-04\nis_end: 2017-12-29\nend: 2025-12-31\n\nrebalance_frequency: M\nsignal_lag_sessions: 1\nlookback_sessions_candidates: [126, 252]\ncost_bps: 2.0\nslippage_bps: 3.0\n'
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
"""Backtest covariance-based heuristic allocations on a fixed ETF universe."""

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
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust total-return (USD adjusted Close)"
SESSIONS_PER_YEAR = 252


def validate_prices(close, tickers, start, end):
    close = close.reindex(columns=tickers).sort_index()
    close.index = pd.DatetimeIndex(close.index, name="Date")
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    close = close.loc[start:end]
    if close.empty or close.index.has_duplicates:
        raise ValueError("Expected nonempty daily prices with unique session dates.")
    if end - close.index[-1] > pd.Timedelta(days=7):
        raise ValueError(f"Prices stop at {close.index[-1].date()}, before requested end {end.date()}.")
    invalid = (~np.isfinite(close) | (close <= 0)).any(axis=0)
    if invalid.any():
        raise ValueError(f"Missing or invalid adjusted closes for {', '.join(invalid.index[invalid])}.")
    return close


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    request = json.dumps([sorted(tickers), str(start.date()), str(end.date()), PRICE_ADJUSTMENT])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = CACHE / f"adjusted_close_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            close = pd.read_csv(source, parse_dates=["Date"], index_col="Date")
        return validate_prices(close, tickers, start, end), fetched_at

    with TemporaryDirectory(dir=CACHE) as session_cache:
        yf.set_tz_cache_location(session_cache)
        quotes = yf.download(
            tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d", auto_adjust=True, progress=False, timeout=15,
        )
    if quotes.empty or "Close" not in quotes:
        raise RuntimeError("Yahoo Finance returned no adjusted closes.")
    close = validate_prices(quotes["Close"], tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as output:
        temporary_path = Path(output.name)
        try:
            output.write(f"# fetched_at={fetched_at}\n")
            close.to_csv(output)
            output.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return close, fetched_at


def equal_weight(covariance):
    size = len(covariance)
    return np.full(size, 1.0 / size)


def inverse_volatility(covariance):
    inverse_vol = 1.0 / np.sqrt(covariance.diagonal())
    return inverse_vol / inverse_vol.sum()


def equal_risk_contribution(covariance):
    volatility = np.sqrt(covariance.diagonal())
    correlation = covariance / np.outer(volatility, volatility)
    # Solve x_i (R x)_i = 1, then undo the volatility scaling and normalize.
    scaled = np.ones(len(volatility))
    for _ in range(2000):
        for i in range(len(scaled)):
            cross_term = correlation[i] @ scaled - correlation[i, i] * scaled[i]
            discriminant = np.hypot(cross_term, 2.0)
            if cross_term >= 0:
                scaled[i] = 2.0 / (discriminant + cross_term)
            else:
                scaled[i] = (discriminant - cross_term) / 2.0
        if np.max(np.abs(scaled * (correlation @ scaled) - 1.0)) < 1e-8:
            weights = scaled / volatility
            return weights / weights.sum()
    raise RuntimeError("Equal risk contribution solver did not converge.")


def hierarchical_risk_parity(covariance):
    volatility = np.sqrt(covariance.diagonal())
    correlation = covariance / np.outer(volatility, volatility)
    distance = np.sqrt(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance), method="single"))
    weights = np.ones(len(order))

    def cluster_variance(indices):
        block = covariance[np.ix_(indices, indices)]
        within_weights = 1.0 / block.diagonal()
        within_weights /= within_weights.sum()
        return within_weights @ block @ within_weights

    def bisect(indices):
        if len(indices) == 1:
            return
        middle = len(indices) // 2
        left, right = indices[:middle], indices[middle:]
        left_var, right_var = cluster_variance(left), cluster_variance(right)
        left_weight = right_var / (left_var + right_var)
        weights[left] *= left_weight
        weights[right] *= 1.0 - left_weight
        bisect(left)
        bisect(right)

    bisect(order)
    return weights


def execution_targets(returns, first_session, lookback, lag, frequency, allocate):
    index = returns.index
    periods = index.to_period(frequency)
    signals = np.flatnonzero(periods[:-1] != periods[1:])
    targets = {}
    for signal in signals:
        execution = signal + lag
        if not first_session <= execution < len(index):
            continue
        window = returns.iloc[max(0, signal - lookback + 1):signal + 1]
        if len(window) != lookback or window.isna().any().any():
            raise ValueError(f"Insufficient warm-up returns at {index[signal].date()} for {lookback} sessions.")
        covariance = window.cov().to_numpy()
        if (covariance.diagonal() <= 0).any():
            raise ValueError(f"Zero return variance at {index[signal].date()}.")
        targets[execution] = allocate(covariance)
    if not targets:
        raise ValueError("No delayed rebalance fits the reporting period.")
    return targets


def backtest(returns, start, lookback, lag, frequency, rate, allocate):
    first_session = returns.index.searchsorted(start)
    targets = execution_targets(returns, first_session, lookback, lag, frequency, allocate)
    weights = np.zeros(len(returns.columns))
    records = []
    for row, asset_returns in enumerate(returns.iloc[first_session:].to_numpy(), start=first_session):
        positions = weights * (1.0 + asset_returns)
        gross_return = float(weights @ asset_returns)
        weights = positions / (1.0 + gross_return)
        net_fraction, turnover, trades = 1.0, 0.0, 0
        if row in targets:
            target = targets[row]

            def balance(remaining_nav):
                traded_fraction = np.abs(remaining_nav * target - weights).sum()
                return remaining_nav + rate * traded_fraction - 1.0

            # The post-cost target is funded by the same NAV from which costs are paid.
            net_fraction = brentq(balance, 0.0, 1.0) if rate else 1.0
            traded = np.abs(net_fraction * target - weights)
            turnover = traded.sum()
            trades = np.count_nonzero(traded > 1e-10)
            weights = target
        records.append(((1.0 + gross_return) * net_fraction - 1.0, turnover, trades))
    return pd.DataFrame(records, index=returns.index[first_session:], columns=["return", "turnover", "n_trades"])


def sharpe(returns):
    volatility = returns.std(ddof=1)
    if not np.isfinite(volatility) or volatility <= 0:
        return None
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility)


def metrics(daily, cash_returns):
    returns = daily["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period needs at least two return observations.")
    wealth = (1.0 + returns).cumprod()
    return {
        "cagr": float(wealth.iloc[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - cash_returns.loc[returns.index]),
        "max_drawdown": float((wealth / wealth.cummax().clip(lower=1.0) - 1.0).min()),
        "total_return": float(wealth.iloc[-1] - 1.0),
        "turnover": float(daily["turnover"].sum()),
        "n_trades": int(daily["n_trades"].sum()),
    }


def performance(daily, cash_returns, split):
    in_sample = daily.index <= split
    return {
        "is": metrics(daily.loc[in_sample], cash_returns),
        "oos": metrics(daily.loc[~in_sample], cash_returns),
        "full": metrics(daily, cash_returns),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data_start, start, split, end = (pd.Timestamp(config[key]) for key in ("data_start", "start", "is_end", "end"))
    universe = config["universe"]
    rf_ticker = config["risk_free_ticker"]
    lag = config["signal_lag_sessions"]
    frequency = config["rebalance_frequency"]
    lookbacks = config["lookback_sessions_candidates"]
    fee, slippage = config["cost_bps"], config["slippage_bps"]
    cost_rate = (fee + slippage) / 10_000
    if not data_start < start < split < end:
        raise ValueError("Dates must satisfy data_start < start < is_end < end.")
    if len(universe) < 2 or len(set(universe)) != len(universe) or rf_ticker != "BIL" or rf_ticker in universe:
        raise ValueError("Specify at least two distinct ETF tickers and use BIL separately for excess returns.")
    if not isinstance(lag, int) or lag < 1 or frequency not in ("M", "Q"):
        raise ValueError("Use a positive session lag and a monthly (M) or quarterly (Q) schedule.")
    if not lookbacks or len(set(lookbacks)) != len(lookbacks) or any(
        not isinstance(window, int) or window <= len(universe) for window in lookbacks
    ):
        raise ValueError("Use distinct integer lookbacks longer than the number of ETFs.")
    if not (fee >= 0 and slippage >= 0 and cost_rate < 1):
        raise ValueError("Costs and slippage must be nonnegative and total less than 10,000 bps.")

    close, fetched_at = load_prices([*universe, rf_ticker], data_start, end)
    returns = close.pct_change(fill_method=None)
    asset_returns, cash_returns = returns[universe], returns[rf_ticker]
    if not close.index[0] < start:
        raise ValueError("The requested evaluation start must follow the first available close.")
    is_returns = asset_returns.loc[:split]
    scores = {}
    for window in lookbacks:
        trial = backtest(is_returns, start, window, lag, frequency, cost_rate, hierarchical_risk_parity)
        score = sharpe(trial["return"] - cash_returns.loc[trial.index])
        if score is None:
            raise ValueError(f"In-sample excess Sharpe is undefined for {window} sessions.")
        scores[window] = score
    selected = max(scores, key=lambda window: (scores[window], window))

    rules = {
        "HRP": hierarchical_risk_parity,
        "IV": inverse_volatility,
        "ERC": equal_risk_contribution,
        "EW benchmark": equal_weight,
    }
    all_metrics = {}
    for name, allocate in rules.items():
        daily = backtest(asset_returns, start, selected, lag, frequency, cost_rate, allocate)
        all_metrics[name] = performance(daily, cash_returns, split)

    evaluated = asset_returns.loc[start:]
    results = {
        "strategy": config["strategy"],
        "data_range": f"{evaluated.index[0].date()} to {evaluated.index[-1].date()}",
        **all_metrics["HRP"],
        "benchmark": all_metrics["EW benchmark"],
    }
    print(f"PARAM_COMBOS_TRIED: {len(scores) + len(rules) - 1}")
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"HRP lookback selected on IS excess Sharpe: {selected} sessions")
    schedule = "monthly" if frequency == "M" else "quarterly"
    print(f"Benchmark: {schedule} equal weight in the same ETF universe")
    print(f"Excess Sharpe reference: {rf_ticker}")
    for name, periods in all_metrics.items():
        for period, values in periods.items():
            print(f"{name} {period.upper()}: {json.dumps(values, allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
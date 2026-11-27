"""482_dynamic_inclusion_and_bounded_multi_fact
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/482_dynamic_inclusion_and_bounded_multi_fact.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit cc50485fb68601d5f9b01f48262e3a778be06982 on 2026-11-27 09:02 Europe/Oslo, exit 0, output sha256 94f6bd08a38be4bb48ca24711fa282e10aa0f0b97bb5d32aae3d1e29b172c072
Data source: Yahoo Finance daily ETF prices and volume via yfinance, with a local gzip CSV cache. The universe is XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV and XLY. SPY quotes validate the session calendar and BIL daily returns are the risk-free proxy for excess Sharpe.
Data fetched: 2026-11-27T08:02:51+00:00
Sample period: Warm-up history requested from 2008-01-01. The logged evaluation runs from 2010-01-04 through 2025-12-31 inclusive.   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31', 'is_end': '2019-12-31', 'oos_start': '2020-01-02'}
IS/OOS split: IS is 2010-01-04 through 2019-12-31 inclusive. OOS is 2020-01-02 through 2025-12-31 inclusive. The grid is scored on IS net excess Sharpe versus BIL. The selected parameters stay fixed in OOS, which carries forward the portfolio state from IS.
Costs and slippage: Models a 2 bp fee and 3 bp fixed slippage, for 5 bp on each dollar of absolute executed buy or sell notional at rebalance. The simulator solves for the post-cost capital used to size target positions and deducts the charge from portfolio value. The dynamic equal-weight benchmark uses the same friction.
Signal lag: The signal uses daily bars through the prior session close. When the next session is the first session of the month, the simulator rebalances at that session's close after applying its close-to-close return to the old holdings. New weights take effect for the following close-to-close interval. The whole rebalance is skipped if any positive-weight target lacks an execution-day close.
Price adjustment: yfinance Adj Close total-return (auto_adjust=False)
Survivorship bias: The fixed sector ETF list is drawn from funds with data available at the time the strategy was written. The README identifies survivorship and hindsight selection bias because the list omits liquidated funds and gives incomplete coverage of the funds available at historical dates. Monthly eligibility requires 253 consecutive valid closes under the configured lookback.
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.12422071352674213, ann_vol=0.13813527472814777, sharpe=0.9170456025399797, sharpe_excess=0.8864605490623347, max_drawdown=-0.17548049249579634, total_return=2.218908710918463, turnover=0.5616147222262134, n_trades=1124
    oos: cagr=0.07916254904985309, ann_vol=0.1802998713660085, sharpe=0.5133020869541116, sharpe_excess=0.3681176193661763, max_drawdown=-0.34971719565984616, total_return=0.5775967518570646, turnover=0.628678580062533, n_trades=769
    full: cagr=0.10711880725022493, ann_vol=0.15526519073240516, sharpe=0.7334986649896756, sharpe_excess=0.6533481600327368, max_drawdown=-0.34971719565984627, total_return=4.078139926869385, turnover=0.5867470029461861, n_trades=1893
    benchmark: cagr=0.10756036492039112, ann_vol=0.15935410804842093, sharpe=0.7212420784008168, sharpe_excess=0.6431484198882744, max_drawdown=-0.36175993505138093, total_return=4.110577780600812, turnover=0.2444178532406553, n_trades=1893
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 482_dynamic_inclusion_and_bounded_multi_fact\nuniverse: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]\ncalendar_ticker: SPY\nrisk_free: BIL\ndates:\n  history_start: 2008-01-01\n  start: 2010-01-01\n  is_end: 2019-12-31\n  end: 2025-12-31\nsignals:\n  liquidity_lookback_sessions: 60\n  volatility_lookback_sessions: 60\n  momentum_lookback_sessions: 252\n  momentum_skip_sessions: 21\n  min_eligible: 5\n  momentum_weight: 0.50\n  lower_weight_multiple: 0.70\n  upper_weight_multiple: 1.30\n  is_grid:\n    min_average_dollar_volume: [5000000, 10000000]\n    max_annualized_volatility: [0.40, 0.55]\n    tilt_strength: [0.35, 0.65]\nexecution:\n  signal_lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\n'
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
import gzip
import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_FIELDS = ["Adj Close", "Close", "Volume"]
GRID_KEYS = ("min_average_dollar_volume", "max_annualized_volatility", "tilt_strength")


def load_prices(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    key = hashlib.sha256(f"v1|{start}|{end}|{','.join(symbols)}".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"yahoo_{key}.csv.gz"
    if cache_file.exists():
        with gzip.open(cache_file, "rt", encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        return prices, fetched_at

    with TemporaryDirectory(dir=cache_dir) as scratch:
        yf.set_tz_cache_location(scratch)
        for _ in range(2):
            downloaded = yf.download(
                symbols, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d", auto_adjust=False, progress=False, threads=4, timeout=12,
            )
            prices = downloaded.reindex(columns=pd.MultiIndex.from_product([PRICE_FIELDS, symbols]))
            missing = prices.columns[prices.isna().all()].tolist()
            if not missing:
                break
    if missing:
        raise RuntimeError(f"Yahoo Finance returned empty fields for {missing}")

    prices = prices.sort_index()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=cache_dir, suffix=".csv.gz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with gzip.open(temporary_path, "wt", encoding="utf-8") as output:
            output.write(fetched_at + "\n")
            prices.to_csv(output)
        temporary_path.replace(cache_file)
    finally:
        temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def monthly_signals(close, returns, dollar_volume, evaluation_dates, model, lag):
    lookback = model["momentum_lookback_sessions"]
    skip = model["momentum_skip_sessions"]
    has_history = close.rolling(lookback + 1).count().eq(lookback + 1)
    momentum = (close.shift(skip) / close.shift(lookback) - 1).where(has_history)
    volatility = (returns.rolling(model["volatility_lookback_sessions"]).std()
                  * np.sqrt(SESSIONS_PER_YEAR))
    liquidity = dollar_volume.rolling(model["liquidity_lookback_sessions"]).mean()

    months = close.index.to_period("M")
    execution_dates = close.index[~months.duplicated()].intersection(evaluation_dates)
    signals = {}
    for execution_date in execution_dates:
        signal_index = close.index.get_loc(execution_date) - lag
        if signal_index < lookback:
            raise ValueError("Increase the history before the first evaluation month.")
        signals[execution_date] = pd.DataFrame({
            "momentum": momentum.iloc[signal_index],
            "volatility": volatility.iloc[signal_index],
            "liquidity": liquidity.iloc[signal_index],
        })
    return signals


def bounded_multiples(raw, lower, upper):
    # Scale and clip simultaneously so the final multiples retain their bounds.
    left, right = 0.0, upper / raw.min()
    for _ in range(48):
        scale = (left + right) / 2
        multiples = np.clip(scale * raw, lower, upper)
        if multiples.mean() < 1:
            left = scale
        else:
            right = scale
    return np.clip(((left + right) / 2) * raw, lower, upper)


def build_targets(signals, model, params):
    tilted_targets = {}
    equal_targets = {}
    for date, features in signals.items():
        eligible = (features["momentum"].notna()
                    & features["liquidity"].ge(params["min_average_dollar_volume"])
                    & features["volatility"].le(params["max_annualized_volatility"]))
        count = int(eligible.sum())
        equal = np.zeros(len(features))
        tilted = equal.copy()
        if count >= model["min_eligible"]:
            chosen = features.loc[eligible]
            momentum_rank = chosen["momentum"].rank()
            low_volatility_rank = (-chosen["volatility"]).rank()
            mix = model["momentum_weight"]
            rank = mix * momentum_rank + (1 - mix) * low_volatility_rank
            score = 2 * (rank.to_numpy() - 1) / (count - 1) - 1
            raw = 1 + params["tilt_strength"] * score
            multiples = bounded_multiples(raw, model["lower_weight_multiple"],
                                          model["upper_weight_multiple"])
            equal[eligible.to_numpy()] = 1 / count
            tilted[eligible.to_numpy()] = multiples / count
        tilted_targets[date] = tilted
        equal_targets[date] = equal
    return tilted_targets, equal_targets


def rebalance(positions, target, friction):
    # Post-cost capital funds the target weights. Friction uses actual executed notional.
    remaining = 1.0
    for _ in range(20):
        traded = np.abs(remaining * target - positions)
        updated = 1 - friction * traded.sum()
        if abs(updated - remaining) < 1e-14:
            remaining = updated
            break
        remaining = updated
    else:
        raise RuntimeError("Rebalance cost calculation did not converge.")
    return remaining, np.abs(remaining * target - positions)


def simulate(returns, close, targets, friction):
    positions = np.zeros(len(returns.columns))
    daily_results = []
    prices_present = close.notna().to_numpy()
    for index, (date, daily_moves) in enumerate(zip(returns.index, returns.to_numpy())):
        invested = positions > 0
        observed = np.isfinite(daily_moves)
        if np.any(invested & ~observed):
            missing = list(returns.columns[invested & ~observed])
            raise RuntimeError(f"Missing return for active positions on {date.date()}: {missing}")

        moves = np.where(observed, daily_moves, 0.0)
        gross = 1 + positions @ moves
        positions = positions * (1 + moves) / gross
        traded = np.zeros_like(positions)
        target = targets.get(date)
        if target is not None and prices_present[index, target > 0].all():
            remaining, traded = rebalance(positions, target, friction)
            gross *= remaining
            positions = target
        daily_results.append((gross - 1, traded.sum() / 2, np.count_nonzero(traded > 1e-10)))
    return pd.DataFrame(daily_results, index=returns.index,
                        columns=["return", "turnover", "n_trades"])


def sharpe(returns):
    deviation = returns.std(ddof=1)
    if not np.isfinite(deviation) or deviation < 1e-12:
        return None
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / deviation)


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    growth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1 / years) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - risk_free),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def select_params(signals, returns, close, risk_free, model, friction):
    candidates = product(*(model["is_grid"][key] for key in GRID_KEYS))
    best_params, best_score = None, -np.inf
    tried = 0
    for values in candidates:
        params = dict(zip(GRID_KEYS, values))
        targets, _ = build_targets(signals, model, params)
        daily_results = simulate(returns, close, targets, friction)
        score = sharpe(daily_results["return"] - risk_free)
        tried += 1
        if score is not None and score > best_score:
            best_params, best_score = params, score
    if best_params is None:
        raise RuntimeError("The in-sample grid did not produce a defined excess Sharpe ratio.")
    return best_params, tried


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    history_start, start, split, end = (pd.Timestamp(config["dates"][key])
                                       for key in ("history_start", "start", "is_end", "end"))
    assets = config["universe"]
    calendar_ticker, risk_free_ticker = config["calendar_ticker"], config["risk_free"]
    symbols = [*assets, calendar_ticker, risk_free_ticker]
    model, execution = config["signals"], config["execution"]
    lag = execution["signal_lag_sessions"]
    cost, slippage = execution["cost_bps"], execution["slippage_bps"]
    friction = (cost + slippage) / 10_000
    if not history_start < start < split < end or len(set(symbols)) != len(symbols):
        raise ValueError("Use ascending dates and distinct tickers.")
    if risk_free_ticker != "BIL":
        raise ValueError("The excess return calculation expects BIL as the risk-free proxy.")
    if not isinstance(lag, int) or lag < 1 or min(cost, slippage) < 0 or friction >= 0.1:
        raise ValueError("Use a positive session lag and nonnegative costs below 1000 bps in total.")
    if not 2 <= model["min_eligible"] <= len(assets):
        raise ValueError("The breadth floor must be between two and the size of the universe.")
    if not 0 <= model["momentum_skip_sessions"] < model["momentum_lookback_sessions"]:
        raise ValueError("The momentum skip must be shorter than the lookback.")
    if not 0 < model["lower_weight_multiple"] <= 1 <= model["upper_weight_multiple"]:
        raise ValueError("Weight multiples must bracket one and have a positive lower bound.")
    if not 0 <= model["momentum_weight"] <= 1 or any(
        not 0 <= value < 1 for value in model["is_grid"]["tilt_strength"]
    ):
        raise ValueError("Use a convex factor mix and tilt strengths between zero and one.")

    prices, fetched_at = load_prices(symbols, history_start, end)
    adjusted_close = prices["Adj Close"].where(prices["Adj Close"].gt(0))
    if adjusted_close[calendar_ticker].isna().any():
        raise RuntimeError(f"Missing {calendar_ticker} closes in the downloaded session calendar.")
    if adjusted_close.index[-1] < end - pd.Timedelta(days=4):
        raise RuntimeError("Yahoo Finance history stops before the requested evaluation end.")
    close = adjusted_close[assets]
    returns = close.pct_change(fill_method=None)
    dollar_volume = prices["Close"][assets].where(prices["Close"][assets].gt(0)) * (
        prices["Volume"][assets].where(prices["Volume"][assets].gt(0))
    )
    evaluation_dates = close.loc[start:end].index
    is_dates = evaluation_dates[evaluation_dates <= split]
    oos_dates = evaluation_dates[evaluation_dates > split]
    if min(len(is_dates), len(oos_dates)) < 2:
        raise ValueError("Both evaluation periods need at least two sessions.")
    risk_free = adjusted_close[risk_free_ticker].pct_change(fill_method=None).loc[evaluation_dates]
    if not np.isfinite(risk_free).all():
        raise RuntimeError(f"Missing {risk_free_ticker} returns in the evaluation period.")

    signals = monthly_signals(close, returns, dollar_volume, evaluation_dates, model, lag)
    is_signals = {date: values for date, values in signals.items() if date <= split}
    params, tried = select_params(is_signals, returns.loc[is_dates], close.loc[is_dates],
                                  risk_free.loc[is_dates], model, friction)
    strategy_targets, benchmark_targets = build_targets(signals, model, params)
    returns = returns.loc[evaluation_dates]
    close = close.loc[evaluation_dates]
    strategy_results = simulate(returns, close, strategy_targets, friction)
    benchmark_results = simulate(returns, close, benchmark_targets, friction)

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance Adj Close total-return (auto_adjust=False)")
    print(f"PARAM_COMBOS_TRIED: {tried}")
    print(f"IS_SELECTED_PARAMETERS: {json.dumps(params)}")
    result = {"strategy": config["strategy"], "data_range": {
        "start": evaluation_dates[0].date().isoformat(),
        "end": evaluation_dates[-1].date().isoformat(),
        "is_end": is_dates[-1].date().isoformat(),
        "oos_start": oos_dates[0].date().isoformat(),
    }}
    for period, dates in (("is", is_dates), ("oos", oos_dates), ("full", evaluation_dates)):
        result[period] = metrics(strategy_results.loc[dates], risk_free.loc[dates])
        benchmark = metrics(benchmark_results.loc[dates], risk_free.loc[dates])
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
        print(f"BENCHMARK DYNAMIC_EQUAL_WEIGHT {period.upper()}: "
              f"{json.dumps(benchmark, allow_nan=False)}")
        if period == "full":
            result["benchmark"] = benchmark
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
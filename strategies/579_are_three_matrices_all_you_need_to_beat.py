"""579_are_three_matrices_all_you_need_to_beat
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/579_are_three_matrices_all_you_need_to_beat.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:08 Europe/Oslo, exit 0, output sha256 0944e8cf36101d468a88600344f0cc598b1f94376cb5271a00160b07324089ac
Data source: Daily ETF opens and closes come from Yahoo Finance via free yfinance data. The sector ETFs are XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV and XLY. SPY is the benchmark and market-regime input. BIL is the defensive ETF and its returns are the excess-Sharpe reference.
Data fetched: 2026-11-27T18:08:39+00:00
Sample period: The daily price panel spans 2018-07-02 through 2026-07-31. Prices before evaluation provide warm-up history. Evaluation covers 2020-03-02 through 2026-07-31, with 2020-03-01 as the configured start cutoff.   data_range (from output): {'start': '2020-03-02', 'end': '2026-07-31'}
IS/OOS split: IS runs from 2020-03-02 through 2024-12-31 inclusive. OOS runs from the first trading session of 2025 through 2026-07-31 inclusive. Both periods are slices of one continuous simulation, with holdings carried into OOS. The configuration is fixed across the split and transition estimates update on a rolling basis.
Costs and slippage: The official run applies 3 bps of trading cost and 2 bps of modeled slippage to absolute traded notional on each buy or sell. The combined 5 bps is included in the rebalance budget and charged to cash, with fills priced at the adjusted open. The SPY benchmark pays the same rate at its initial purchase.
Signal lag: Targets use data through the close of the last trading session of the preceding month. The configured lag is one trading session, so the target is executed at the open of the first session of the following month. The model assumes the completed prior daily bar is available by that open.
Price adjustment: yfinance auto_adjust=True total-return adjusted Open and Close
Survivorship bias: The backtest uses the fixed list of 11 sector ETFs that the README identifies as the 2018 Select Sector SPDR lineup. The loader requires valid historical prices for all configured tickers. The analysis uses fund-level histories from one issuer's sector lineup, with sector classification changes and vendor revisions identified as limitations.
Parameter combinations tried: 4
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1630467090550929, ann_vol=0.14007917531052247, sharpe=1.1487198515339958, sharpe_excess=0.9815799099465281, max_drawdown=-0.12024980021196263, total_return=1.0751663875232578, turnover=6.501835401230872, n_trades=303
    oos: cagr=0.06373456894440621, ann_vol=0.10956326183330932, sharpe=0.618731754032743, sharpe_excess=0.2614059320303455, max_drawdown=-0.07887878980029084, total_return=0.10169173339753024, turnover=8.090195501982274, n_trades=105
    full: cagr=0.13790092693251776, ann_vol=0.1332455845555412, sharpe=1.0364895207973934, sharpe_excess=0.8316098309966098, max_drawdown=-0.12024980021196263, total_return=1.2861936545587977, turnover=6.890801451941848, n_trades=408
    benchmark: cagr=0.17100603993576824, ann_vol=0.20226121835188282, sharpe=0.8820731278453565, sharpe_excess=0.7469449439556873, max_drawdown=-0.2831904906931897, total_return=1.746836924734799, turnover=0.15615254988733038, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '579_ are_three_matrices_all_you_need_to_beat'\nuniverse: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]\nbenchmark: SPY\ndefensive_etf: BIL\ndates:\n  data_start: '2018-07-02'\n  evaluation_start: '2020-03-01'\n  is_end: '2024-12-31'\n  end: '2026-07-31'\nexecution:\n  lag_sessions: 1\n  cost_bps: 3.0\n  slippage_bps: 2.0\nmodel:\n  momentum_sessions: 126\n  volatility_sessions: 21\n  slow_volatility_sessions: 63\n  distance_sessions: 126\n  regime_months: 3\n  rank_buckets: 5\n  covariate_buckets: 3\n  transition_months: 36\n  minimum_transition_months: 12\n  conditional_prior_count: 20.0\nallocation:\n  sector_count: 4\n  switch_band: 0.08\n  distance_tilt: 0.50\n  down_volatility_weight: 0.75\n  down_equity_weight: 0.40\nrobustness:\n  momentum_sessions: [105, 147]\n  higher_slippage_bps: 7.0\n"
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
"""Monthly Select Sector SPDR adaptation of Igor Halperin's OMD-Portfolio.

Paper (2026): https://arxiv.org/abs/2607.27461
Author's private repository: https://github.com/ighalp/omd_portfolio
QuantConnect Tutorials (Apache-2.0), market regime and momentum reference:
https://github.com/QuantConnect/Tutorials/blob/master/04%20Strategy%20Library/37%20Momentum%20and%20State%20of%20Market%20Filters/02%20Method.html
PyDTMC (MIT), discrete transition estimation reference:
https://pypi.org/project/PyDTMC/
https://pydtmc.readthedocs.io/advanced_topics/simulation_empirical_analysis.html
Related replication report:
https://www.agents-quant.com/blog/%E4%B8%89%E7%9F%A9%E9%98%B5osi%E6%8B%A9%E6%97%B6%E5%9C%A8a%E8%82%A1%E7%9A%84%E5%A4%8D%E7%8E%B0%E4%B8%8E%E8%AF%81%E4%BC%AA/
All implementation code here is original.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
TRADING_DAYS = 252


@dataclass(frozen=True)
class MonthState:
    return_ranks: np.ndarray
    volatility_ranks: np.ndarray
    distance_buckets: np.ndarray
    volatility_cells: np.ndarray
    residual_distances: np.ndarray


@dataclass(frozen=True)
class Target:
    signal_close: pd.Timestamp
    weights: np.ndarray


def read_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe = config["universe"]
    tickers = [*universe, config["benchmark"], config["defensive_etf"]]
    if len(set(tickers)) != len(tickers):
        raise ValueError("Sector, benchmark and defensive symbols must be distinct.")
    dates = config["dates"]
    if not (dates["data_start"] < dates["evaluation_start"] < dates["is_end"] < dates["end"]):
        raise ValueError("Data start, evaluation start, IS end and end must be chronological.")
    model = config["model"]
    for name in ("momentum_sessions", "volatility_sessions", "slow_volatility_sessions", "distance_sessions"):
        if not isinstance(model[name], int) or model[name] < 2:
            raise ValueError(f"{name} must be an integer of at least two sessions.")
    for name in ("rank_buckets", "covariate_buckets"):
        if not 2 <= model[name] <= len(universe):
            raise ValueError(f"{name} must fit the sector universe.")
    if not 1 <= model["minimum_transition_months"] <= model["transition_months"]:
        raise ValueError("Invalid transition history lengths.")
    if model["conditional_prior_count"] <= 0 or model["regime_months"] < 1:
        raise ValueError("Conditional prior and regime window must be positive.")
    allocation = config["allocation"]
    if not 2 <= allocation["sector_count"] <= len(universe):
        raise ValueError("The basket must contain at least two sectors and fit the universe.")
    for name in ("switch_band", "distance_tilt", "down_volatility_weight", "down_equity_weight"):
        if not 0 <= allocation[name] <= 1:
            raise ValueError(f"{name} must lie between zero and one.")
    lag = config["execution"]["lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution must lag by at least one session.")
    return config


def check_prices(prices, columns, start, end):
    if prices.empty:
        raise ValueError("Yahoo returned an empty price panel.")
    prices = prices.reindex(columns=columns).sort_index()
    values = prices.to_numpy()
    bad_columns = (~np.isfinite(values) | (values <= 0)).any(axis=0)
    if bad_columns.any():
        missing = ", ".join(f"{ticker}/{field}" for field, ticker in columns[bad_columns])
        raise ValueError(f"Missing or invalid daily prices for {missing}.")
    if prices.index.has_duplicates or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Expected a unique session index.")
    first, last = prices.index[0], prices.index[-1]
    if not start <= first <= start + pd.Timedelta(days=7) or not end - pd.Timedelta(days=7) <= last <= end:
        raise ValueError(f"Price coverage is incomplete: {first.date()} through {last.date()}.")
    return prices


def load_prices(config):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    tickers = [*config["universe"], config["benchmark"], config["defensive_etf"]]
    start = pd.Timestamp(config["dates"]["data_start"])
    end = pd.Timestamp(config["dates"]["end"])
    columns = pd.MultiIndex.from_product([["Open", "Close"], tickers], names=["Price", "Ticker"])
    key = sha256(repr((sorted(tickers), str(start), str(end), "adjusted_open_close_v1")).encode()).hexdigest()[:16]
    cache_file = cache_dir / f"yahoo_{key}.csv"

    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as source:
            fetched_at = datetime.fromisoformat(source.readline().strip()).isoformat()
            prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        return check_prices(prices, columns, start, end), fetched_at

    with TemporaryDirectory(dir=cache_dir, prefix=".request_") as request_dir:
        yf.set_tz_cache_location(request_dir)
        for attempt in range(3):
            try:
                prices = yf.download(
                    tickers,
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    auto_adjust=True,
                    progress=False,
                    threads=4,
                    timeout=12,
                )
                prices = check_prices(prices, columns, start, end)
                break
            except (ValueError, OSError) as error:
                if attempt == 2:
                    raise RuntimeError("Could not download a complete Yahoo price panel.") from error
                print(f"DATA_RETRY: {error}", flush=True)

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=cache_dir, mode="w", encoding="utf-8", delete=False) as destination:
        temporary = Path(destination.name)
        destination.write(fetched_at + "\n")
        prices.to_csv(destination)
    temporary.replace(cache_file)
    return prices, fetched_at


def rank_buckets(values, bucket_count):
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[order] = np.arange(1, len(values) + 1)
    return (bucket_count * ranks - 1) // len(values)


def angular_distances(returns):
    correlation = np.corrcoef(returns, rowvar=False)
    if not np.isfinite(correlation).all():
        raise ValueError("An ETF has an undefined trailing correlation.")
    distances = np.arccos(np.clip(correlation, -1, 1))
    np.fill_diagonal(distances, 0)
    return distances


def month_state(returns, close_index, model):
    def trailing(sessions):
        return returns[close_index - sessions + 1 : close_index + 1]

    ranks = model["rank_buckets"]
    covariate_buckets = model["covariate_buckets"]
    return_ranks = rank_buckets(-trailing(model["momentum_sessions"]).mean(axis=0), ranks)
    volatility_ranks = rank_buckets(trailing(model["volatility_sessions"]).std(axis=0), ranks)
    slow_volatility = rank_buckets(trailing(model["slow_volatility_sessions"]).std(axis=0), covariate_buckets)
    window = trailing(model["distance_sessions"])
    distance_buckets = rank_buckets(angular_distances(window).mean(axis=1), covariate_buckets)

    # Remove the intercept and the equal-weight sector return from the trailing window.
    centered = window - window.mean(axis=0)
    market_mode = centered.mean(axis=1)
    beta = (market_mode @ centered) / (market_mode @ market_mode)
    residual_distances = angular_distances(centered - np.outer(market_mode, beta))
    return MonthState(
        return_ranks, volatility_ranks, distance_buckets,
        distance_buckets * covariate_buckets + slow_volatility, residual_distances,
    )


def best_rank_probability(ranks, cells, rank_count, cell_count, prior_count):
    # Pair consecutive months along each column so sector paths never join at their ends.
    pairs = (cells[:-1] * rank_count + ranks[:-1]) * rank_count + ranks[1:]
    counts = np.bincount(pairs.ravel(), minlength=cell_count * rank_count**2)
    counts = counts.reshape(cell_count, rank_count, rank_count)
    pooled = counts.sum(axis=0) + 1.0
    pooled /= pooled.sum(axis=1, keepdims=True)
    smoothed = counts + prior_count * pooled
    transition = smoothed / smoothed.sum(axis=2, keepdims=True)
    return transition[cells[-1], ranks[-1], 0]


def select_sectors(scores, distances, baskets, incumbents, allocation):
    sector_count = baskets.shape[1]
    first, second = np.triu_indices(sector_count, k=1)
    average_distance = distances[baskets[:, first], baskets[:, second]].mean(axis=1) / np.pi
    objective = scores[baskets].sum(axis=1) + allocation["distance_tilt"] * sector_count / 2 * average_distance
    if incumbents is not None:
        replacements = sector_count - np.isin(baskets, incumbents).sum(axis=1)
        objective -= allocation["switch_band"] * replacements
    return baskets[np.argmax(objective)]


def make_targets(prices, config):
    close = prices["Close"]
    sector_returns = close[config["universe"]].pct_change(fill_method=None).to_numpy()
    market = close[config["benchmark"]].to_numpy()
    model = config["model"]
    allocation = config["allocation"]
    sector_count = len(config["universe"])
    baskets = np.array(list(combinations(range(sector_count), allocation["sector_count"])))
    month_ids = close.index.to_period("M")
    # The first session of a new month confirms the previous month's closing observation.
    new_months = np.flatnonzero(month_ids[1:] != month_ids[:-1]) + 1
    month_ends = new_months - 1
    required_sessions = max(model[name] for name in (
        "momentum_sessions", "volatility_sessions", "slow_volatility_sessions", "distance_sessions",
    ))
    lag = config["execution"]["lag_sessions"]
    evaluation_start = pd.Timestamp(config["dates"]["evaluation_start"])
    states = []
    targets = {}
    incumbents = None

    for month_number, close_index in enumerate(month_ends):
        if close_index < required_sessions:
            continue
        states.append(month_state(sector_returns, close_index, model))
        if len(states) <= model["minimum_transition_months"] or month_number < model["regime_months"]:
            continue
        fill_index = close_index + lag
        if fill_index >= len(close):
            break
        fill_date = close.index[fill_index]
        if fill_date < evaluation_start:
            continue
        past = states[-(model["transition_months"] + 1) :]
        rank_count = model["rank_buckets"]
        cell_count = model["covariate_buckets"]
        prior_count = model["conditional_prior_count"]
        return_probability = best_rank_probability(
            np.stack([state.return_ranks for state in past]),
            np.stack([state.distance_buckets for state in past]),
            rank_count, cell_count, prior_count,
        )
        calm_probability = best_rank_probability(
            np.stack([state.volatility_ranks for state in past]),
            np.stack([state.volatility_cells for state in past]),
            rank_count, cell_count**2, prior_count,
        )
        earlier_close_index = month_ends[month_number - model["regime_months"]]
        rising = market[close_index] > market[earlier_close_index]
        volatility_weight = 0.0 if rising else allocation["down_volatility_weight"]
        equity_weight = 1.0 if rising else allocation["down_equity_weight"]
        scores = (1 - volatility_weight) * return_probability + volatility_weight * calm_probability
        incumbents = select_sectors(scores, states[-1].residual_distances, baskets, incumbents, allocation)
        weights = np.zeros(sector_count + 1)
        weights[incumbents] = equity_weight / len(incumbents)
        weights[-1] = 1 - equity_weight
        targets[fill_date] = Target(close.index[close_index], weights)

    if not targets:
        raise ValueError("Evaluation span has no eligible monthly rebalances.")
    return targets


def rebalance(opening_prices, positions, cash, weights, fee_rate):
    old_values = positions * opening_prices
    opening_value = cash + old_values.sum()

    def budget_gap(budget):
        return budget + fee_rate * np.abs(budget * weights - old_values).sum() - opening_value

    budget = brentq(budget_gap, 0, opening_value, xtol=opening_value * 1e-14)
    # Leave a rounding reserve so conversions between dollars and units stay self-financing.
    new_positions = (budget * (1 - 1e-12)) * weights / opening_prices
    changes = (new_positions - positions) * opening_prices
    traded = np.abs(changes).sum()
    cash = cash - changes.sum() - fee_rate * traded
    if cash < 0:
        raise ArithmeticError("Rebalance would produce negative cash.")
    trades = np.count_nonzero(np.abs(changes) > opening_value * 1e-12)
    return new_positions, cash, traded / opening_value, trades


def simulate(prices, start, targets, fee_rate):
    prices = prices.loc[start:]
    opening = prices["Open"].to_numpy()
    closing = prices["Close"].to_numpy()
    positions = np.zeros(opening.shape[1])
    cash = previous_value = 1.0
    rows = []
    for index, date in enumerate(prices.index):
        turnover = 0.0
        trades = 0
        if date in targets:
            positions, cash, turnover, trades = rebalance(opening[index], positions, cash, targets[date], fee_rate)
        value = cash + positions @ closing[index]
        rows.append((value / previous_value - 1, turnover, trades, cash))
        previous_value = value
    return pd.DataFrame(rows, index=prices.index, columns=["daily_return", "turnover", "n_trades", "cash"])


def execution_rate(execution):
    cost = float(execution["cost_bps"])
    slippage = float(execution["slippage_bps"])
    if not 0 <= cost or not 0 <= slippage or not cost + slippage < 10_000:
        raise ValueError("Cost and slippage must be nonnegative and sum to less than 10,000 bps.")
    return (cost + slippage) / 10_000


def strategy_results(prices, config, targets=None):
    if targets is None:
        targets = make_targets(prices, config)
    weights = {date: target.weights for date, target in targets.items()}
    if any(not (target.signal_close < date) for date, target in targets.items()):
        raise AssertionError("A signal must predate the fill.")
    tickers = [*config["universe"], config["defensive_etf"]]
    price_subset = prices.loc[:, pd.IndexSlice[:, tickers]]
    return simulate(price_subset, config["dates"]["evaluation_start"], weights, execution_rate(config["execution"]))


def metrics(daily_results, reference_returns):
    returns = daily_results["daily_return"].to_numpy()
    excess = returns - reference_returns.loc[daily_results.index].to_numpy()
    if len(returns) < 2:
        raise ValueError("A reported period needs at least two sessions.")
    curve = np.r_[1.0, np.cumprod(1 + returns)]

    def sharpe(values):
        volatility = values.std(ddof=1)
        return float(np.sqrt(TRADING_DAYS) * values.mean() / volatility) if volatility > 0 else None

    return {
        "cagr": float(curve[-1] ** (TRADING_DAYS / len(returns)) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float((curve / np.maximum.accumulate(curve) - 1).min()),
        "total_return": float(curve[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * TRADING_DAYS),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def audit_timing(prices, config, targets, official, is_only):
    index = prices.index
    for fill_date, target in targets.items():
        lag = index.get_loc(fill_date) - index.get_loc(target.signal_close)
        if lag != config["execution"]["lag_sessions"]:
            raise AssertionError("Fill does not match the configured lag.")
    is_end = config["dates"]["is_end"]
    pd.testing.assert_frame_equal(official.loc[:is_end], is_only)

    first_fill = min(targets)
    changed_close = prices.loc[:first_fill].copy()
    changed_close.loc[first_fill, pd.IndexSlice["Close", :]] *= 1.1
    changed_targets = make_targets(changed_close, config)
    np.testing.assert_array_equal(changed_targets[first_fill].weights, targets[first_fill].weights)
    fill_fields = ["turnover", "n_trades", "cash"]
    changed_result = strategy_results(changed_close, config, changed_targets)
    pd.testing.assert_series_equal(changed_result.loc[first_fill, fill_fields], official.loc[first_fill, fill_fields])

    # A later corporate action rescales each earlier history in the affected symbol's units.
    prefix = prices.loc[:is_end]
    scales = np.linspace(0.4, 1.8, len(prefix["Open"].columns))
    rescaled = prefix * np.tile(scales, 2)
    pd.testing.assert_frame_equal(strategy_results(rescaled, config), is_only, rtol=1e-9, atol=1e-11)

    print("LOOK_AHEAD_SELF_CHECK: signal_close_to_lagged_open=PASS, "
          "fill_day_close_independence=PASS, availability=completed_daily_prices_only, "
          "split_and_dividend_unit_invariance=PASS, is_prefix_matches_full=PASS")


def main():
    config = read_config()
    prices, fetched_at = load_prices(config)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True total-return adjusted Open and Close")
    print(f"PRICE_DATA_RANGE: {prices.index[0].date()} to {prices.index[-1].date()}")

    dates = config["dates"]
    is_end = pd.Timestamp(dates["is_end"])
    start = dates["evaluation_start"]
    defensive = config["defensive_etf"]
    reference_returns = prices["Close"][defensive].pct_change(fill_method=None).loc[start:].copy()
    first_date = reference_returns.index[0]
    reference_open = prices.loc[first_date, ("Open", defensive)]
    reference_returns.loc[first_date] = prices.loc[first_date, ("Close", defensive)] / reference_open - 1

    is_prices = prices.loc[:is_end]
    is_only = strategy_results(is_prices, config)
    print("ROBUSTNESS: IS only, fixed official configuration")
    variants = []
    for window in config["robustness"]["momentum_sessions"]:
        variant = deepcopy(config)
        variant["model"]["momentum_sessions"] = window
        variants.append((f"momentum_sessions={window}", variant))
    variant = deepcopy(config)
    variant["execution"]["slippage_bps"] = config["robustness"]["higher_slippage_bps"]
    variants.append((f"slippage_bps={variant['execution']['slippage_bps']:g}", variant))
    for label, variant in variants:
        variant_metrics = metrics(strategy_results(is_prices, variant), reference_returns)
        print(f"ROBUSTNESS_IS {label}: {json.dumps(variant_metrics, allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(variants)}")

    targets = make_targets(prices, config)
    official = strategy_results(prices, config, targets)
    audit_timing(prices, config, targets, official, is_only)
    benchmark_ticker = config["benchmark"]
    benchmark = simulate(
        prices.loc[:, pd.IndexSlice[:, [benchmark_ticker]]], start,
        {first_date: np.ones(1)}, execution_rate(config["execution"]),
    )
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(first_date.date()), "end": str(official.index[-1].date())},
    }
    for period, mask in (("is", official.index <= is_end), ("oos", official.index > is_end), ("full", slice(None))):
        results[period] = metrics(official.loc[mask], reference_returns)
        baseline = metrics(benchmark.loc[mask], reference_returns)
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"BENCHMARK_{period.upper()} {benchmark_ticker}: {json.dumps(baseline, allow_nan=False)}")
        if period == "full":
            results["benchmark"] = baseline
    if min(results["is"]["n_trades"], results["oos"]["n_trades"]) == 0:
        raise RuntimeError("Expected simulated trades in both evaluation periods.")
    print(f"FILL_SCHEDULE: {len(targets)} monthly rebalances, first {min(targets).date()}, last {max(targets).date()}")
    print("RUN_STATUS: 已回测")
    print("REVIEW_STATUS: 待审，确有交易")
    print("RESULTS_JSON: " + json.dumps(results, ensure_ascii=False, allow_nan=False))


def stop_at_deadline(_signum, _frame):
    raise RuntimeError("Research run reached the 14-minute deadline.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_at_deadline)
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)

"""491_topological_risk_parity
WARNING: known defects (see DEEP_REVIEW_FINDINGS.md); numbers invalid.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/491_topological_risk_parity.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit d2f4ee010416fa7342bf8a9f3404c2adc27e3f7d on 2026-11-27 09:18 Europe/Oslo, exit 0, output sha256 f122249022bd09dea69a898e771e2120f906a40c024fe3f1ed2034ae3c18fe36
Data source: Public Yahoo Finance daily ETF closes via yfinance (`interval="1d"`), with a local CSV cache. SPY is the market label and buy-and-hold benchmark, and BIL adjusted-close returns are the cash proxy. Sector ETFs: XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY. Industry ETFs: SMH, IGV, XBI, IHF, KRE, IAI, XRT, ITB, XOP, XME, IYT, IYZ.
Data fetched: 2026-11-27T08:18:31+00:00
Sample period: Reported evaluation sessions: 2020-01-02 through 2025-12-31, inclusive. Price history was requested from 2019-01-01, with 2019 used for warm-up. data_range (from output): {'start': '2020-01-02', 'end': '2025-12-31'}
IS/OOS split: Retrospective chronological split. IS: 2020-01-02 through 2022-12-30. OOS: 2023-01-03 through 2025-12-31. The default configuration is simulated continuously and the resulting daily returns are split at the inclusive IS end. The seven alternative specifications are evaluated on IS data only.
Costs and slippage: 2 bps one-way transaction cost plus 3 bps one-way slippage, modelled as a fixed 5 bps charge on the absolute dollar value of each ETF buy and sell. Drifted weights are marked at the execution close, then the simulator deducts the charge while solving for post-cost target weights. The initial SPY benchmark buy uses the same rate. Extra friction on cash movements is unmodelled.
Signal lag: The strategy calculates each monthly signal with data through the first trading session’s close t and fills at the next trading session’s close (t+1 under the default one-session lag). The old weights earn the return through the fill close. New weights first earn the close-to-close return from the fill close to the following session’s close.
Price adjustment: yfinance auto_adjust total-return ETF closes (splits and distributions)
Survivorship bias: The fixed list was selected retrospectively from ETFs with available public histories. explicitly identifies survivorship and selection bias and places the evaluation start after all selected ETFs have common price history. The backtest requires valid adjusted closes for all configured tickers across the loaded rows.
Parameter combinations tried: 8
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.08951278418141495, ann_vol=0.2486998574020095, sharpe=0.47038092506815904, sharpe_excess=0.44760080326046864, max_drawdown=-0.33608297430294753, total_return=0.29329319277232235, turnover=5.705305537436436, n_trades=587
    oos: cagr=0.1474535747324024, ann_vol=0.16793536626114278, sharpe=0.9032949003251809, sharpe_excess=0.6245309374437, max_drawdown=-0.2094566056700624, total_return=0.5074995572806313, turnover=6.113904573034962, n_trades=647
    full: cagr=0.11803109480608964, ann_vol=0.21223319275122482, sharpe=0.6327633121732696, sharpe_excess=0.5094196215523761, max_drawdown=-0.33608297430294753, total_return=0.9496389155383314, turnover=5.909063146700421, n_trades=1234
    benchmark: cagr=0.1485220983280331, ann_vol=0.20741315957075898, sharpe=0.7717652998676106, sharpe_excess=0.6455120118504614, max_drawdown=-0.33717257210811413, total_return=1.2902428786454272, turnover=0.1670252406953022, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 491_topological_risk_parity\n\nuniverse:\n  market: SPY\n  cash_proxy: BIL\n  sector_etfs: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]\n  industry_etfs: [SMH, IGV, XBI, IHF, KRE, IAI, XRT, ITB, XOP, XME, IYT, IYZ]\n\ndates:\n  data_start: 2019-01-01\n  sample_start: 2020-01-02\n  is_end: 2022-12-30\n  sample_end: 2025-12-31\n\nallocation:\n  correlation_lookback: 126\n  momentum_lookback: 126\n  activity_lookback: 21\n  minimum_activity: 0.0001\n  signal_threshold: 0.001\n  split_mix: 0.50\n  max_weight: 0.20\n  min_weight: 0.005\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 2\n  slippage_bps: 3\n\nrobustness:\n  correlation_lookbacks: [105, 147]\n  momentum_lookbacks: [105, 147]\n  split_mixes: [0.35, 0.65]\n  topology_off_split_mix: 0.0\n'
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
"""Long-only ETF research adaptation of the paper's semi-supervised TRP."""

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Allocation:
    correlation_lookback: int
    momentum_lookback: int
    activity_lookback: int
    minimum_activity: float
    signal_threshold: float
    split_mix: float
    max_weight: float
    min_weight: float

    def __post_init__(self):
        if min(self.correlation_lookback, self.momentum_lookback, self.activity_lookback) < 2:
            raise ValueError("Lookbacks must be at least two sessions.")
        if not 0 <= self.split_mix <= 1 or not 0 <= self.min_weight <= self.max_weight <= 1:
            raise ValueError("Invalid split mix or weight limits.")
        if not 0 <= self.minimum_activity or not 0 <= self.signal_threshold:
            raise ValueError("Activity and signal thresholds must be nonnegative.")


def validate_prices(prices, symbols, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price data must contain unique, ordered sessions.")
    if not start <= prices.index[0] <= start + pd.Timedelta(days=7):
        raise ValueError(f"Price history does not cover the requested start date {start.date()}.")
    if not end - pd.Timedelta(days=7) <= prices.index[-1] <= end:
        raise ValueError(f"Price history does not cover the requested end date {end.date()}.")
    prices = prices.reindex(columns=symbols)
    valid = np.isfinite(prices) & (prices > 0)
    if not valid.all().all():
        missing = valid.columns[~valid.all()].tolist()
        raise ValueError(f"Missing or invalid adjusted closes for {', '.join(missing)}.")
    return prices


def load_prices(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = json.dumps(["yfinance-auto-adjust-close", symbols, str(start.date()), str(end.date())])
    cache_path = cache_dir / f"prices_{hashlib.sha256(cache_key.encode()).hexdigest()[:16]}.csv"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as saved:
            fetched_at = saved.readline().strip().removeprefix("# fetched_at=")
            if datetime.fromisoformat(fetched_at).utcoffset() is None:
                raise ValueError("Cache fetch timestamp must have a timezone.")
            prices = pd.read_csv(
                saved, index_col="Date", parse_dates=["Date"], converters={symbol: float for symbol in symbols}
            )
        return validate_prices(prices, symbols, start, end), fetched_at

    closes = []
    pending = symbols
    with TemporaryDirectory(dir=cache_dir) as session_cache:
        yf.set_tz_cache_location(session_cache)
        for _ in range(2):
            batch = yf.download(
                pending,
                start=str(start.date()),
                end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d",
                auto_adjust=True,
                threads=6,
                timeout=10,
                progress=False,
            )
            if batch is not None and not batch.empty:
                adjusted_closes = batch["Close"].dropna(axis=1, how="all")
                closes.append(adjusted_closes)
                pending = [symbol for symbol in pending if symbol not in adjusted_closes.columns]
            if not pending:
                break
    if pending:
        raise RuntimeError(f"yfinance returned no prices for {', '.join(pending)}.")

    prices = pd.concat(closes, axis=1).sort_index()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).rename("Date")
    prices = validate_prices(prices, symbols, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete_on_close=False) as saved:
        saved.write(f"# fetched_at={fetched_at}\n")
        prices.to_csv(saved)
        saved.close()
        Path(saved.name).replace(cache_path)
    return prices, fetched_at


def topological_factors(history, n_sectors, split_mix):
    if split_mix == 0:
        return np.ones(history.shape[1])

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.atleast_2d(np.corrcoef(history, rowvar=False))
    correlation = np.nan_to_num(correlation, nan=0.0)
    correlation = np.clip((correlation + correlation.T) / 2, -1, 1)
    distance = np.sqrt((1 - correlation) / 2)
    # Sparse graph zeros denote missing edges, so keep exact ties as positive edges.
    np.maximum(distance, np.finfo(float).eps, out=distance)
    np.fill_diagonal(distance, 0.0)
    tree = minimum_spanning_tree(csr_matrix(distance))
    adjacency = (tree + tree.T).tocsr()
    adjacency.sort_indices()

    # Reserve every sector as a direct child of the dummy market root before DFS.
    visited = np.zeros(history.shape[1], dtype=bool)
    visited[:n_sectors] = True
    factors = np.zeros(history.shape[1])
    factors[:n_sectors] = 1 - split_mix + split_mix / n_sectors
    stack = list(reversed(range(n_sectors)))
    while stack:
        parent = stack.pop()
        neighbors = adjacency.indices[adjacency.indptr[parent]:adjacency.indptr[parent + 1]]
        children = neighbors[~visited[neighbors]]
        if children.size:
            visited[children] = True
            factors[children] = factors[parent] * (1 - split_mix + split_mix / children.size)
            stack.extend(reversed(children))
    if not visited.all():
        raise RuntimeError("The rooted tree did not reach every included ETF.")
    return factors


def monthly_targets(prices, asset_returns, start, n_sectors, allocation, lag):
    close = prices.to_numpy()
    returns = asset_returns.to_numpy()
    warmup = max(allocation.correlation_lookback, allocation.momentum_lookback, allocation.activity_lookback)
    first_session = prices.index.searchsorted(start)
    if first_session < warmup:
        raise ValueError("Insufficient price history before the evaluation start for this specification.")
    months = prices.index.to_period("M")
    signal_rows = np.flatnonzero(months[1:] != months[:-1]) + 1
    targets = {}
    for row in signal_rows[(signal_rows >= first_session) & (signal_rows + lag < len(prices))]:
        signal = close[row] / close[row - allocation.momentum_lookback] - 1
        activity = np.abs(returns[row - allocation.activity_lookback + 1:row + 1]).mean(axis=0)
        active = (signal > allocation.signal_threshold) & (activity > allocation.minimum_activity)
        included = active.copy()
        included[:n_sectors] = True
        history = returns[row - allocation.correlation_lookback + 1:row + 1, included]
        factors = topological_factors(history, n_sectors, allocation.split_mix)
        scores = np.where(active[included], signal[included], 0.0) * factors
        if scores.sum() > 0:
            scores = np.minimum(scores / scores.sum(), allocation.max_weight)
            scores[scores < allocation.min_weight] = 0.0
        weights = np.zeros(close.shape[1])
        weights[included] = scores
        targets[prices.index[row + lag]] = weights
    return targets


def execute_target(current, target, fee_rate):
    # Target weights are fractions of post-cost value. Solve for that value as
    # a fraction of pre-cost value while charging both costs on the risky trades.
    remaining = brentq(
        lambda fraction: fraction + fee_rate * np.abs(fraction * target - current).sum() - 1,
        0.0,
        1.0,
    )
    trades = np.abs(remaining * target - current)
    return remaining, float(trades.sum()), int(np.count_nonzero(trades > 1e-10))


def simulate(asset_returns, cash_returns, targets, fee_rate):
    weights = np.zeros(asset_returns.shape[1])
    daily_results = []
    for date, asset_return, cash_return in zip(
        asset_returns.index, asset_returns.to_numpy(), cash_returns.to_numpy(), strict=True
    ):
        grown_assets = weights * (1 + asset_return)
        growth = grown_assets.sum() + (1 - weights.sum()) * (1 + cash_return)
        weights = grown_assets / growth
        turnover, n_trades = 0.0, 0
        if date in targets:
            target = targets[date]
            remaining, turnover, n_trades = execute_target(weights, target, fee_rate)
            growth *= remaining
            weights = target
        daily_results.append((growth - 1, turnover, n_trades))
    return pd.DataFrame(daily_results, index=asset_returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    excess = returns - cash_returns.loc[returns.index]
    wealth = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    scale = np.sqrt(SESSIONS_PER_YEAR)
    vol, excess_vol = returns.std(), excess.std()
    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(vol * scale),
        "sharpe": float(returns.mean() / vol * scale) if vol > 1e-12 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * scale) if excess_vol > 1e-12 else None,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], cash_returns),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe = config["universe"]
    sectors = universe["sector_etfs"]
    assets = sectors + universe["industry_etfs"]
    market, cash = universe["market"], universe["cash_proxy"]
    symbols = [market, cash, *assets]
    if not sectors or len(set(symbols)) != len(symbols):
        raise ValueError("The universe needs sector anchors and distinct tickers.")

    dates = config["dates"]
    data_start, start, is_end, end = (
        pd.Timestamp(dates[key]) for key in ("data_start", "sample_start", "is_end", "sample_end")
    )
    if not data_start < start <= is_end < end:
        raise ValueError("Invalid history, sample or split dates.")
    execution = config["execution"]
    lag = execution["lag_sessions"]
    cost, slippage = execution["cost_bps"], execution["slippage_bps"]
    fee_rate = (cost + slippage) / 10_000
    if not isinstance(lag, int) or lag < 1 or min(cost, slippage) < 0 or not 0 <= fee_rate < 1:
        raise ValueError("Execution requires a positive session lag and valid one-way costs.")

    official = Allocation(**config["allocation"])
    robustness = config["robustness"]
    alternatives = {
        **{
            f"correlation_lookback={n}": replace(official, correlation_lookback=n)
            for n in robustness["correlation_lookbacks"]
        },
        **{
            f"momentum_lookback={n}": replace(official, momentum_lookback=n)
            for n in robustness["momentum_lookbacks"]
        },
        **{f"split_mix={mix}": replace(official, split_mix=mix) for mix in robustness["split_mixes"]},
        "topology_off": replace(official, split_mix=robustness["topology_off_split_mix"]),
    }

    prices, fetched_at = load_prices(symbols, data_start, end)
    returns = prices.pct_change(fill_method=None)
    cash_returns = returns[cash]
    evaluation = returns.loc[start:]
    if len(evaluation.loc[:is_end]) < 2 or (evaluation.index > is_end).sum() < 2:
        raise ValueError("Each evaluation period needs at least two sessions.")

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return ETF closes (splits and distributions)")
    print(f"RISK_FREE_PROXY: {cash} adjusted-close daily return")
    print("ROBUSTNESS (IS ONLY)")
    is_prices = prices.loc[:is_end, assets]
    is_returns = returns.loc[:is_end, assets]
    tried = {official}
    for label, alternative in alternatives.items():
        if alternative in tried:
            continue
        targets = monthly_targets(is_prices, is_returns, start, len(sectors), alternative, lag)
        daily_results = simulate(is_returns.loc[start:], cash_returns.loc[start:is_end], targets, fee_rate)
        tried.add(alternative)
        stats = metrics(daily_results, cash_returns.loc[:is_end])
        print(f"ROBUSTNESS_IS {label}: {json.dumps(stats, allow_nan=False)}")

    targets = monthly_targets(prices[assets], returns[assets], start, len(sectors), official, lag)
    daily_results = simulate(evaluation[assets], cash_returns.loc[start:], targets, fee_rate)
    benchmark = simulate(evaluation[[market]], cash_returns.loc[start:], {evaluation.index[0]: np.ones(1)}, fee_rate)
    result = period_metrics(daily_results, cash_returns, is_end)
    benchmark_result = period_metrics(benchmark, cash_returns, is_end)
    print(f"PARAM_COMBOS_TRIED: {len(tried)}")
    for period, stats in result.items():
        print(f"{period.upper()}: {json.dumps(stats, allow_nan=False)}")
    print(f"BENCHMARK: {market} buy and hold")
    for period, stats in benchmark_result.items():
        print(f"BENCHMARK {period.upper()}: {json.dumps(stats, allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps({
        "strategy": config["strategy"],
        "data_range": {"start": str(evaluation.index[0].date()), "end": str(evaluation.index[-1].date())},
        **result,
        "benchmark": {"name": market, **benchmark_result["full"]},
    }, allow_nan=False))


if __name__ == "__main__":
    main()
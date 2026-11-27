"""480_topological_risk_parity
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/480_topological_risk_parity.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit cc50485fb68601d5f9b01f48262e3a778be06982 on 2026-11-27 09:02 Europe/Oslo, exit 0, output sha256 37cd5af247fbd3dac9beb24bc49aa4983e0e0020b17e2a73886a5b6cd01394c6
Data source: Public Yahoo Finance daily ETF closes via yfinance. The allocation universe is XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV and XLY. BIL daily returns are used only as the excess-Sharpe reference. The reported data fetch timestamp is 2026-11-27T07:09:38+00:00.
Data fetched: 2026-11-27T08:02:49+00:00
Sample period: The run reports 2019-01-02 through 2025-12-31 inclusive. The configured warm-up data start is 2018-06-20.   data_range (from output): {'start': '2019-01-02', 'end': '2025-12-31'}
IS/OOS split: IS runs from 2019-01-02 through the 2022-12-31 cutoff. OOS uses sessions after that cutoff through 2025-12-31. The lookback is selected on IS net excess Sharpe relative to BIL, with ties going to the shorter window. The selected lookback was 63 sessions and is fixed for the full simulation. The OOS slice inherits the ending IS holdings.
Costs and slippage: Each simulated buy and sell incurs 5 bps of transaction costs and 3 bps of modeled flat slippage on its traded notional, including initial purchases from cash. At each fill the simulator marks existing holdings to the close, solves for the post-friction portfolio value using the summed absolute trade notionals, and debits the combined 8 bps through that value.
Signal lag: The signal uses daily returns through the close of the first session of each month. The target fills at the next session's close. Existing holdings earn the close-to-close return ending at the fill close, and the new holdings first earn the close-to-close return beginning at that close.
Price adjustment: yfinance auto_adjust total-return
Survivorship bias: The universe is a fixed, retrospectively chosen list of 11 extant sector ETFs, and the latest ETF inception constrains the sample start. The code requires valid prices for all symbols on every included date. The study excludes defunct and historical alternative funds and does not reconstruct historical sector membership.
Parameter combinations tried: 2
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.13671182415859695, ann_vol=0.23113430220697076, sharpe=0.6708319767841268, sharpe_excess=0.630770117132093, max_drawdown=-0.37787273916039243, total_return=0.6695579976780794, turnover=19.385054975024566, n_trades=528
    oos: cagr=0.16689941364976213, ann_vol=0.13575067891624554, sharpe=1.2050547293880252, sharpe_excess=0.8597358772578457, max_drawdown=-0.16030249331076818, total_return=0.5850254588015393, turnover=18.279828603002095, n_trades=396
    full: cagr=0.14951340952440417, ann_vol=0.1960913759679985, sharpe=0.8093111274760761, sharpe_excess=0.6802341977691687, max_drawdown=-0.37787273916039243, total_return=1.6462919312654751, turnover=37.66488357802666, n_trades=924
    benchmark  full: cagr=0.1454462070586071, ann_vol=0.18715241899944055, sharpe=0.8198122688853346, sharpe_excess=0.6845879051645858, max_drawdown=-0.36163460081567356, total_return=1.581587104523909, turnover=3.244542910611311, n_trades=924
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 480_topological_risk_parity\nuniverse: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]\nrisk_free_ticker: BIL\n\ndata_start: 2018-06-20\nstart: 2019-01-02\nis_end: 2022-12-31\nend: 2025-12-31\n\ncorrelation_lookback_candidates: [63, 126]\npropagation_rho: 1.0\nexecution_lag_sessions: 1\ncost_bps: 5.0\nslippage_bps: 3.0\n'
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
"""Long-only sector ETF research backtest using the paper's rooted MST allocator."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data_cache"
SESSIONS_PER_YEAR = 252
SECTOR_ETFS = frozenset("XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY".split())


def check_prices(prices, tickers, start, end):
    prices = prices.reindex(columns=tickers)
    if (
        prices.empty
        or not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
        or not prices.index.isin(pd.date_range(start, end)).all()
        or prices.index[0] > pd.Timestamp(start) + pd.Timedelta(days=7)
        or prices.index[-1] < pd.Timestamp(end) - pd.Timedelta(days=7)
    ):
        raise ValueError("Incomplete or invalid date coverage in the ETF close data")
    valid = np.isfinite(prices) & (prices > 0)
    if not valid.to_numpy().all():
        missing = valid.columns[~valid.all()].tolist()
        raise ValueError(f"Missing or invalid adjusted closes for {', '.join(missing)}")
    return prices


def load_prices(tickers, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = f"yfinance-adjusted-close-v1|{start}|{end}|{','.join(tickers)}"
    cache_path = CACHE_DIR / f"closes_{hashlib.sha256(cache_key.encode()).hexdigest()[:16]}.csv"
    if cache_path.is_file():
        with cache_path.open() as source:
            fetched_at = source.readline().removeprefix("# fetched_at=").strip()
            prices = pd.read_csv(
                source, index_col=0, parse_dates=True, converters=dict.fromkeys(tickers, float),
            )
        datetime.fromisoformat(fetched_at)
        return check_prices(prices, tickers, start, end), fetched_at

    exclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date()
    with TemporaryDirectory(dir=CACHE_DIR) as temporary:
        yf.set_tz_cache_location(temporary)
        for attempt in range(2):
            data = yf.download(
                tickers, start=str(start), end=str(exclusive_end), auto_adjust=True,
                multi_level_index=True, interval="1d", ignore_tz=True, progress=False,
                timeout=10,
            )
            try:
                prices = check_prices(data["Close"], tickers, start, end)
                break
            except (KeyError, ValueError) as exc:
                if attempt == 1:
                    raise RuntimeError(f"Could not obtain complete Yahoo Finance closes: {exc}") from exc

        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        staged = Path(temporary) / "closes.csv"
        with staged.open("w") as destination:
            destination.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(destination, index_label="date")
        staged.replace(cache_path)
    return prices, fetched_at


def trp_weights(window, rho):
    corr = window.corr().to_numpy()
    corr = np.clip(np.nan_to_num(corr), -1.0, 1.0)
    distances = np.sqrt((1.0 - corr) / 2.0)
    # Sparse graphs treat zero as an absent edge. Keep perfectly correlated pairs.
    distances = np.maximum(distances, np.finfo(float).eps)
    np.fill_diagonal(distances, 0.0)
    tree = minimum_spanning_tree(csr_matrix(distances))
    adjacency = (tree + tree.T).tocsr()
    root = int(adjacency.getnnz(axis=1).argmax())
    order, parent = breadth_first_order(adjacency, root, directed=False)
    branching = np.bincount(parent[order[1:]], minlength=len(parent))
    factors = np.ones(len(parent))
    for node in order[1:]:
        upstream = parent[node]
        factors[node] = factors[upstream] * (1.0 - rho + rho / branching[upstream])
    # Every ETF has a unit starting signal and the gross target is one.
    return factors / factors.sum()


def rebalance_dates(sessions, lag):
    months = sessions.to_period("M")
    return [
        (sessions[i], sessions[i + lag])
        for i in np.flatnonzero(~months.duplicated())
        if i + lag < len(sessions)
    ]


def strategy_targets(history, schedule, lookback, rho):
    targets = {}
    for signal_date, fill_date in schedule:
        window = history.loc[:signal_date].tail(lookback)
        if len(window) != lookback or window.isna().to_numpy().any():
            raise ValueError(f"Insufficient price warmup through {signal_date.date()}")
        targets[fill_date] = trp_weights(window, rho)
    return targets


def simulate(returns, targets, friction):
    weights = np.zeros(returns.shape[1])
    daily_results = []
    for date, asset_returns in zip(returns.index, returns.to_numpy(), strict=True):
        # Old positions earn the close-to-close move ending at the fill close.
        marked_weights = weights * (1.0 + asset_returns)
        gross = 1.0 + weights @ asset_returns
        weights = marked_weights / gross
        turnover = 0.0
        n_trades = 0
        target = targets.get(date)
        remaining = 1.0
        if target is not None:
            # Scale the target after fees so that resulting positions are fully funded.
            remaining = brentq(
                lambda fraction: fraction + friction * np.abs(fraction * target - weights).sum() - 1.0,
                0.0, 1.0,
            )
            changes = np.abs(remaining * target - weights)
            turnover = changes.sum()
            n_trades = int(np.count_nonzero(changes > 1e-10))
            weights = target
        daily_results.append((gross * remaining - 1.0, turnover, n_trades))
    return pd.DataFrame(daily_results, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"].to_numpy()
    excess = returns - risk_free.loc[daily_results.index].to_numpy()
    growth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    daily_vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(annualizer * daily_vol),
        "sharpe": float(annualizer * returns.mean() / daily_vol) if daily_vol > 0 else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], risk_free),
        "oos": metrics(daily_results.loc[~in_sample], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def main():
    cfg = yaml.safe_load((BASE_DIR / "config.yaml").read_text())
    tickers = sorted(cfg["universe"])
    risk_free_ticker = cfg["risk_free_ticker"]
    windows = cfg["correlation_lookback_candidates"]
    rho = cfg["propagation_rho"]
    lag = cfg["execution_lag_sessions"]
    cost_bps, slippage_bps = cfg["cost_bps"], cfg["slippage_bps"]
    friction = (cost_bps + slippage_bps) / 10_000.0
    start, is_end, end = map(pd.Timestamp, (cfg["start"], cfg["is_end"], cfg["end"]))
    if not pd.Timestamp(cfg["data_start"]) < start < is_end < end:
        raise ValueError("Expected data_start < start < is_end < end")
    if lag < 1 or not 0 <= rho <= 1 or min(cost_bps, slippage_bps) < 0 or not friction < 1:
        raise ValueError("Invalid execution lag, propagation rho, costs, or slippage")
    if not windows or min(windows) < 2 or len(set(windows)) != len(windows):
        raise ValueError("Correlation windows must be distinct and at least two sessions")
    if len(set(tickers)) != len(tickers) or len(tickers) < 2 or not SECTOR_ETFS.issuperset(tickers):
        raise ValueError("Expected at least two distinct Select Sector SPDR ETFs")
    if risk_free_ticker != "BIL":
        raise ValueError("Expected BIL as the price-based excess return reference")

    prices, fetched_at = load_prices(tickers + [risk_free_ticker], cfg["data_start"], cfg["end"])
    daily_returns = prices.pct_change(fill_method=None)
    history = daily_returns[tickers]
    sample = history.loc[start:end]
    risk_free = daily_returns[risk_free_ticker]
    training = sample.loc[:is_end]
    if len(training) < 2 or len(sample) - len(training) < 2:
        raise ValueError("IS and OOS each need at least two sessions")

    training_schedule = rebalance_dates(training.index, lag)
    scores = {}
    for lookback in windows:
        targets = strategy_targets(history.loc[:is_end], training_schedule, lookback, rho)
        results = simulate(training, targets, friction)
        scores[lookback] = metrics(results, risk_free.loc[:is_end])["sharpe_excess"]
    chosen = max(windows, key=lambda days: (scores[days], -days))

    schedule = rebalance_dates(sample.index, lag)
    targets = strategy_targets(history, schedule, chosen, rho)
    strategy_periods = period_metrics(simulate(sample, targets, friction), risk_free, is_end)
    equal_weights = np.full(len(tickers), 1.0 / len(tickers))
    benchmark_targets = dict.fromkeys((fill_date for _, fill_date in schedule), equal_weights)
    benchmark_periods = period_metrics(simulate(sample, benchmark_targets, friction), risk_free, is_end)

    report = {
        "strategy": cfg["strategy"],
        "data_range": {"start": str(sample.index[0].date()), "end": str(sample.index[-1].date())},
        **strategy_periods,
        "benchmark": {"name": "Equal-weight sector ETFs", **benchmark_periods},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print(f"SELECTED_LOOKBACK_SESSIONS: {chosen}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy_periods[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark_periods[period], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

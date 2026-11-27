# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 480_topological_risk_parity\nuniverse: [XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY]\nrisk_free_ticker: BIL\n\ndata_start: 2018-06-20\nstart: 2019-01-02\nis_end: 2022-12-31\nend: 2025-12-31\n\ncorrelation_lookback_candidates: [63, 126]\npropagation_rho: 1.0\n# Signal on the first session's close each month. Fill one session later at the close.\nexecution_lag_sessions: 1\n# Use 95% of prior-close value to fix quantities. Reserve 5% before fees and price moves.\n# The fill uses these same quantities. A cash shortfall stops the simulation.\ncash_buffer_fraction: 0.05\ncost_bps: 5.0\nslippage_bps: 3.0\n"
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
import warnings
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree


BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "data_cache"
SESSIONS_PER_YEAR = 252
SECTOR_ETFS = frozenset("XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY".split())


def exchange_sessions(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if not 2018 <= start.year <= end.year <= 2028:
        raise ValueError("NYSE Arca calendar covers 2018–2028")
    calendar = AbstractHolidayCalendar(rules=[
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ])
    mourning_closures = pd.DatetimeIndex(["2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(calendar.holidays(start, end).union(mourning_closures))


def check_prices(prices, tickers, sessions):
    prices = prices.reindex(columns=tickers)
    if (
        not isinstance(prices.index, pd.DatetimeIndex)
        or not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
    ):
        raise ValueError("Expected unique, increasing dates in the ETF close data")
    if not prices.index.equals(sessions):
        missing = sessions.difference(prices.index).strftime("%Y-%m-%d").tolist()
        unexpected = prices.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise ValueError(
            f"NYSE Arca session mismatch: {len(missing)} missing {missing[:5]}, "
            f"{len(unexpected)} unexpected {unexpected[:5]}"
        )
    valid = np.isfinite(prices) & (prices > 0)
    if not valid.to_numpy().all():
        missing = valid.columns[~valid.all()].tolist()
        raise ValueError(f"Missing or invalid adjusted closes for {', '.join(missing)}")
    return prices


def load_prices(tickers, start, end):
    sessions = exchange_sessions(start, end)
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
        return check_prices(prices, tickers, sessions), fetched_at

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
                prices = check_prices(data["Close"], tickers, sessions)
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
    undefined = ~np.isfinite(corr)
    undefined_pairs = np.count_nonzero(np.triu(undefined, k=1))
    if undefined_pairs:
        warnings.warn(
            f"{window.index[-1]} ({len(window)} sessions): "
            f"using zero for {undefined_pairs} undefined pair correlations",
            RuntimeWarning,
            stacklevel=2,
        )
    corr = np.clip(np.where(undefined, 0.0, corr), -1.0, 1.0)
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


def rebalance_values(fixed_changes, close, cash, friction):
    traded_value = np.abs(fixed_changes) @ close
    net_purchase_value = fixed_changes @ close
    return cash - net_purchase_value - traded_value * friction, traded_value


def plan_rebalance(target, positions, cash, prior_close, cash_buffer, friction, sizing_date):
    """Freeze the next session's share changes using the sizing session's close."""
    prior_close_value = cash + positions @ prior_close
    target_value = prior_close_value * (1.0 - cash_buffer)
    fixed_changes = target_value * target / prior_close - positions
    estimated_cash, _ = rebalance_values(fixed_changes, prior_close, cash, friction)
    if not np.isfinite(estimated_cash) or estimated_cash <= 0.0:
        raise ValueError(
            f"{sizing_date.date()}: prior-close estimate leaves cash "
            f"{estimated_cash:.8g} before the scheduled fill"
        )
    fixed_changes.setflags(write=False)
    return fixed_changes


def settle_rebalance(fixed_changes, fill_close, cash, friction, fill_date):
    """Price the frozen share changes and reject a fill that would require borrowing."""
    cash_after_fill, traded_value = rebalance_values(fixed_changes, fill_close, cash, friction)
    if not np.isfinite(cash_after_fill) or cash_after_fill < 0.0:
        raise ValueError(
            f"{fill_date.date()}: fixed-share rebalance would leave cash {cash_after_fill:.8g} "
            f"(starting cash {cash:.8g}, gross traded value {traded_value:.8g})"
        )
    return cash_after_fill, traded_value


def simulate(closes, targets, friction, cash_buffer):
    positions = np.zeros(closes.shape[1])
    cash = previous_value = 1.0
    fixed_changes = None
    daily_results = []
    for i, close in enumerate(closes.to_numpy()):
        # Old positions earn the close-to-close move ending at the fill close.
        value_before_fill = cash + positions @ close
        closing_value = value_before_fill
        turnover = 0.0
        n_trades = 0
        if fixed_changes is not None:
            cash, traded_value = settle_rebalance(fixed_changes, close, cash, friction, closes.index[i])
            positions += fixed_changes
            closing_value = cash + positions @ close
            turnover = traded_value / value_before_fill
            n_trades = int(np.count_nonzero(fixed_changes))
        daily_results.append((closing_value / previous_value - 1.0, turnover, n_trades))
        previous_value = closing_value

        # Commit quantities at this close for the next session's fill.
        fixed_changes = None
        if i + 1 < len(closes):
            next_target = targets.get(closes.index[i + 1])
            if next_target is not None:
                fixed_changes = plan_rebalance(
                    next_target, positions, cash, close, cash_buffer, friction, closes.index[i],
                )
    return pd.DataFrame(daily_results, index=closes.index, columns=["return", "turnover", "n_trades"])


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
    cash_buffer = cfg["cash_buffer_fraction"]
    cost_bps, slippage_bps = cfg["cost_bps"], cfg["slippage_bps"]
    friction = (cost_bps + slippage_bps) / 10_000.0
    start, is_end, end = map(pd.Timestamp, (cfg["start"], cfg["is_end"], cfg["end"]))
    if not pd.Timestamp(cfg["data_start"]) < start < is_end < end:
        raise ValueError("Expected data_start < start < is_end < end")
    if type(lag) is not int or lag < 1 or not 0 <= rho <= 1 or not 0 <= friction < 1:
        raise ValueError("Invalid execution lag, propagation rho, or total trading friction")
    if not 0 < cash_buffer < 1 or not (cost_bps >= 0 and slippage_bps >= 0):
        raise ValueError("Expected a cash buffer between zero and one and nonnegative trading costs")
    if not windows or min(windows) < 2 or len(set(windows)) != len(windows):
        raise ValueError("Correlation windows must be distinct and at least two sessions")
    if len(set(tickers)) != len(tickers) or len(tickers) < 2 or not SECTOR_ETFS.issuperset(tickers):
        raise ValueError("Expected at least two distinct Select Sector SPDR ETFs")
    if risk_free_ticker != "BIL":
        raise ValueError("Expected BIL as the price-based excess return reference")

    prices, fetched_at = load_prices(tickers + [risk_free_ticker], cfg["data_start"], cfg["end"])
    daily_returns = prices.pct_change(fill_method=None)
    history = daily_returns[tickers]
    sample = prices.loc[start:end, tickers]
    risk_free = daily_returns[risk_free_ticker]
    training = sample.loc[:is_end]
    if len(training) < 2 or len(sample) - len(training) < 2:
        raise ValueError("IS and OOS each need at least two sessions")

    training_schedule = rebalance_dates(training.index, lag)
    scores = {}
    for lookback in windows:
        targets = strategy_targets(history.loc[:is_end], training_schedule, lookback, rho)
        results = simulate(training, targets, friction, cash_buffer)
        scores[lookback] = metrics(results, risk_free.loc[:is_end])["sharpe_excess"]
    chosen = max(windows, key=lambda days: (scores[days] if scores[days] is not None else -np.inf, -days))

    schedule = rebalance_dates(sample.index, lag)
    targets = strategy_targets(history, schedule, chosen, rho)
    strategy_periods = period_metrics(simulate(sample, targets, friction, cash_buffer), risk_free, is_end)
    equal_weights = np.full(len(tickers), 1.0 / len(tickers))
    benchmark_targets = dict.fromkeys((fill_date for _, fill_date in schedule), equal_weights)
    benchmark_periods = period_metrics(
        simulate(sample, benchmark_targets, friction, cash_buffer), risk_free, is_end,
    )

    report = {
        "strategy": cfg["strategy"],
        "data_range": {"start": str(sample.index[0].date()), "end": str(sample.index[-1].date())},
        **strategy_periods,
        "benchmark": {"name": "Equal-weight sector ETFs", **benchmark_periods},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print(f"IS_LOOKBACK_SHARPE_EXCESS: {json.dumps(scores, allow_nan=False)}")
    print(f"SELECTED_LOOKBACK_SESSIONS: {chosen}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy_periods[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark_periods[period], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()

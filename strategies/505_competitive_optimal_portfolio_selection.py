"""505_competitive_optimal_portfolio_selection
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/505_competitive_optimal_portfolio_selection.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:03 Europe/Oslo, exit 0, output sha256 796a432a68a7aaa47eb9ca36b8c4d6d56e51f9a0fc80fee39265cbdaefa375b2
Data source: Yahoo Finance via yfinance, daily adjusted ETF closes for XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV and XLY, plus SPY as benchmark and peer proxy and BIL as the cash and risk-free proxy.
Data fetched: 2026-11-27T09:03:59+00:00
Sample period: Scored sessions from 2008-01-02 through 2025-12-31, inclusive. The price request starts on 2006-07-01 for warm-up.   data_range (from output): {'start': '2008-01-02', 'end': '2025-12-31'}
IS/OOS split: IS covers 2008-01-02 through 2017-12-29. OOS covers 2018-01-02 through 2025-12-31. The cutoff is 2018-01-01. The official strategy is simulated continuously across the split with fixed parameters and rolling estimates. The six sensitivity variants use only IS data and are reported for IS only.
Costs and slippage: The official backtest models 5 bp in transaction costs plus 3 bp in fixed slippage on each buy and sell, for a total of 8 bp of executed value. At the fill close, the simulator solves for the fee deduction and scales the target holdings by capital remaining after fees. Fees reduce that day's portfolio value. The SPY benchmark pays the same rate at its initial purchase. One IS sensitivity uses 10 bp in costs plus 6 bp in slippage.
Signal lag: The monthly signal uses close-to-close returns through the final trading session of the prior month and fills at the next session's close under the configured one-session lag. Existing holdings or the BIL cash proxy earn the return into the fill close. The new weights first earn market returns over the following close-to-close interval.
Price adjustment: yfinance auto_adjust total-return closes (dividends and splits)
Survivorship bias: The portfolio universe is a fixed list of nine long-lived sector ETFs selected for complete historical coverage. The README identifies survivorship bias and notes that later-launched real estate and communication services sector ETFs are excluded.
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.0706042750232514, ann_vol=0.1968608675019537, sharpe=0.4453213722253977, sharpe_excess=0.4321246655551902, max_drawdown=-0.48207127695314445, total_return=0.9772181193717167, turnover=61.993901394323395, n_trades=507
    oos: cagr=0.1626640632110552, ann_vol=0.2151358307037739, sharpe=0.8086612143909738, sharpe_excess=0.695652992726271, max_drawdown=-0.31793193132259945, total_return=2.3291648229401107, turnover=33.6657827217435, n_trades=382
    full: cagr=0.11054555088167772, ann_vol=0.2051711438331366, sharpe=0.614065024335774, sharpe_excess=0.5542192702963923, max_drawdown=-0.48207127695314445, total_return=5.582485010292103, turnover=95.6596841160669, n_trades=889
    benchmark SPY full: cagr=0.11044150416739562, ann_vol=0.19950130355872123, sharpe=0.6250091587557874, sharpe_excess=0.5629270878166472, max_drawdown=-0.5187376000277399, total_return=5.571410146429693, turnover=0.9992006394884093, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 505_competitive_optimal_portfolio_selection\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free: BIL\nhistory_start: 2006-07-01\nstart: 2008-01-02\nend: 2025-12-31\noos_start: 2018-01-01\nsignal_lag_sessions: 1\n\nparameters:\n  lookback_days: 252\n  risk_aversion: 20.0\n  competition_weight: 1.0\n  cost_bps: 5.0\n  slippage_bps: 3.0\n\nrobustness:\n  shorter_lookback: {lookback_days: 210}\n  longer_lookback: {lookback_days: 294}\n  lower_risk_aversion: {risk_aversion: 16.0}\n  higher_risk_aversion: {risk_aversion: 24.0}\n  absolute_variance: {competition_weight: 0.0}\n  doubled_costs: {cost_bps: 10.0, slippage_bps: 6.0}\n'
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
"""Monthly sector allocation with a penalty for variance relative to SPY."""

import json
import signal
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-8
SECTOR_FUNDS = {"XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"}


@dataclass(frozen=True)
class Parameters:
    lookback_days: int
    risk_aversion: float
    competition_weight: float
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if type(self.lookback_days) is not int or self.lookback_days < 2:
            raise ValueError("lookback_days must be an integer of at least two")
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0:
            raise ValueError("risk_aversion must be finite and positive")
        if not 0 <= self.competition_weight <= 1:
            raise ValueError("competition_weight must be between zero and one")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.trade_rate < 1):
            raise ValueError("Cost and slippage must be nonnegative and total under 10000 bps")

    @property
    def trade_rate(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_closes(prices, tickers, risk_free, history_start, start, end):
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.empty:
        raise ValueError("Yahoo Finance returned no dated adjusted closes")
    if prices.index.has_duplicates or not prices.index.is_monotonic_increasing:
        raise ValueError("Expected unique adjusted closes in chronological order")
    first, last = prices.index[[0, -1]]
    week = pd.Timedelta(days=7)
    if not history_start <= first <= history_start + week or not end - week <= last <= end:
        raise ValueError(f"Incomplete Yahoo Finance date coverage: {first.date()} to {last.date()}")
    if prices.index.to_series().diff().gt(week).any():
        raise ValueError("Yahoo Finance history has a gap longer than seven calendar days")

    risk_free_start = prices.index.searchsorted(start) - 1
    if risk_free_start < 0:
        raise ValueError("At least one pre-sample close is required for daily returns")
    for ticker in tickers:
        ticker_closes = prices[ticker]
        if ticker == risk_free:
            ticker_closes = ticker_closes.iloc[risk_free_start:]
        invalid = ~np.isfinite(ticker_closes) | (ticker_closes <= 0)
        if invalid.any():
            date = ticker_closes.index[invalid][0].date()
            raise ValueError(f"Missing or invalid adjusted close for {ticker} on {date}")


def load_closes(tickers, risk_free, history_start, start, end):
    CACHE.mkdir(exist_ok=True)
    symbols = "_".join(tickers)
    path = CACHE / f"yfinance_adjusted_{history_start.date()}_{end.date()}_{symbols}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            fetched_at = stream.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(stream, index_col="date", parse_dates=["date"])
        prices = prices.reindex(columns=tickers)
        check_closes(prices, tickers, risk_free, history_start, start, end)
        return prices, fetched_at

    with TemporaryDirectory(dir=CACHE, prefix="yfinance_") as cache_dir:
        yf.set_tz_cache_location(cache_dir)
        for attempt in range(2):
            data = yf.download(
                tickers,
                start=history_start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                threads=False,
                progress=False,
                timeout=8,
            )
            prices = data["Close"].reindex(columns=tickers) if data is not None and "Close" in data else pd.DataFrame()
            try:
                check_closes(prices, tickers, risk_free, history_start, start, end)
            except ValueError:
                if attempt == 1:
                    raise
            else:
                break
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            stream.write(fetched_at + "\n")
            prices.to_csv(stream, index_label="date")
            stream.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def optimal_weights(relative_window, risk_aversion):
    mean = relative_window.mean(axis=0) * SESSIONS_PER_YEAR
    covariance = np.atleast_2d(np.cov(relative_window, rowvar=False)) * SESSIONS_PER_YEAR
    n_assets = len(mean)
    ones = np.ones(n_assets)

    def objective(weights):
        variance = weights @ covariance @ weights
        return 0.5 * risk_aversion * variance - mean @ weights

    result = minimize(
        objective,
        ones / n_assets,
        jac=lambda weights: risk_aversion * (covariance @ weights) - mean,
        method="SLSQP",
        bounds=[(0, 1)] * n_assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1, "jac": lambda weights: ones},
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not result.success or not np.isfinite(result.x).all() or abs(result.x.sum() - 1) > 1e-8:
        raise RuntimeError(f"Sector optimization failed: {result.message}")
    weights = result.x.clip(0, 1)
    weights[weights < TRADE_TOLERANCE] = 0
    return weights / weights.sum()


def rebalance_targets(asset_returns, peer_returns, start, lag, spec):
    dates = asset_returns.index
    month_openings = np.flatnonzero(dates.to_period("M")[1:] != dates.to_period("M")[:-1]) + 1
    # Fully invested weights let each asset be expressed in benchmark-relative returns.
    relative = asset_returns.to_numpy() - spec.competition_weight * peer_returns.to_numpy()[:, None]
    targets = {}
    for opening in month_openings:
        fill = opening + lag - 1
        if fill >= len(dates) or dates[fill] < start:
            continue
        first_return = opening - spec.lookback_days
        if first_return < 1:
            raise ValueError(f"Insufficient signal history for {dates[opening - 1].date()}")
        # The last return ends at the prior month's final close. The fill is a later close.
        targets[dates[fill]] = optimal_weights(relative[first_return:opening], spec.risk_aversion)
    if not targets:
        raise ValueError("The period contains no eligible monthly rebalance")
    return targets


def simulate(asset_returns, risk_free_returns, targets, trade_rate):
    n_assets = asset_returns.shape[1]
    weights = np.zeros(n_assets)
    invested = False
    rows = []
    for date, daily_return, cash_return in zip(
        asset_returns.index, asset_returns.to_numpy(), risk_free_returns.to_numpy(), strict=True
    ):
        if invested:
            grown_weights = weights * (1 + daily_return)
            gross_factor = grown_weights.sum()
            weights = grown_weights / gross_factor
        else:
            gross_factor = 1 + cash_return

        cost_fraction = turnover = 0.0
        trades = 0
        target = targets.get(date)
        if target is not None:
            # Both purchases and sales pay bps on executed value. Scale the target after paying fees.
            def cost_balance(fraction):
                changes = (1 - fraction) * target - weights
                return fraction - trade_rate * np.abs(changes).sum()

            cost_fraction = brentq(cost_balance, 0, 1)
            changes = np.abs((1 - cost_fraction) * target - weights)
            turnover = changes.sum()
            trades = np.count_nonzero(changes > TRADE_TOLERANCE)
            weights = target
            invested = True
        rows.append((gross_factor * (1 - cost_fraction) - 1, turnover, trades))
    return pd.DataFrame(rows, index=asset_returns.index, columns=["return", "turnover", "n_trades"])


def sharpe(returns):
    volatility = returns.std(ddof=1)
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility) if volatility > 0 else None


def metrics(daily_results, risk_free_returns):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each metrics period needs at least two return observations")
    growth = np.r_[1.0, np.cumprod(1 + returns)]
    excess = returns - risk_free_returns.loc[daily_results.index].to_numpy()
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free_returns, split):
    before_split = daily_results.index < split
    return {
        "is": metrics(daily_results.loc[before_split], risk_free_returns),
        "oos": metrics(daily_results.loc[~before_split], risk_free_returns),
        "full": metrics(daily_results, risk_free_returns),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe, peer, risk_free = config["universe"], config["benchmark"], config["risk_free"]
    if not universe or len(universe) != len(set(universe)) or not set(universe) <= SECTOR_FUNDS:
        raise ValueError("universe must contain distinct Select Sector SPDR funds")
    if (peer, risk_free) != ("SPY", "BIL"):
        raise ValueError("This strategy uses SPY as benchmark and BIL as its cash proxy")

    history_start, start, end, split = (
        pd.Timestamp(config[key]) for key in ("history_start", "start", "end", "oos_start")
    )
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    if not history_start < start < split <= end < today:
        raise ValueError("Expected history_start < start < oos_start <= end before today in New York")
    lag = config["signal_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be a positive integer")

    official = Parameters(**config["parameters"])
    sensitivities = {label: replace(official, **changes) for label, changes in config["robustness"].items()}
    if len(set(sensitivities.values()) | {official}) != len(sensitivities) + 1:
        raise ValueError("Each robustness case must use a distinct parameter combination")

    tickers = sorted([*universe, peer, risk_free])
    closes, fetched_at = load_closes(tickers, risk_free, history_start, start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return closes (dividends and splits)")
    returns = closes.pct_change(fill_method=None)
    scored = returns.loc[start:]
    in_sample = returns.loc[returns.index < split]
    is_scored = in_sample.loc[start:]

    print("ROBUSTNESS (IS only)")
    for label, spec in sensitivities.items():
        targets = rebalance_targets(in_sample[universe], in_sample[peer], start, lag, spec)
        daily_results = simulate(is_scored[universe], is_scored[risk_free], targets, spec.trade_rate)
        result = {"parameters": asdict(spec), "is": metrics(daily_results, is_scored[risk_free])}
        print(f"  {label}: {json.dumps(result, allow_nan=False)}")

    targets = rebalance_targets(returns[universe], returns[peer], start, lag, official)
    daily_results = simulate(scored[universe], scored[risk_free], targets, official.trade_rate)
    benchmark_targets = {min(targets): np.ones(1)}
    benchmark_results = simulate(scored[[peer]], scored[risk_free], benchmark_targets, official.trade_rate)
    official_summary = period_metrics(daily_results, scored[risk_free], split)
    benchmark_summary = period_metrics(benchmark_results, scored[risk_free], split)

    print(f"PARAM_COMBOS_TRIED: {1 + len(sensitivities)}")
    print(f"OFFICIAL_PARAMETERS: {json.dumps(asdict(official), allow_nan=False)}")
    for period, result in official_summary.items():
        print(f"{period.upper()}: {json.dumps(result, allow_nan=False)}")
        print(f"BENCHMARK {peer} {period.upper()}: {json.dumps(benchmark_summary[period], allow_nan=False)}")
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": scored.index[0].date().isoformat(), "end": scored.index[-1].date().isoformat()},
        **official_summary,
        "benchmark": {"ticker": peer, **benchmark_summary},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()
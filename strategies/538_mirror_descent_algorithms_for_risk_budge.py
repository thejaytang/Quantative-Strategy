"""538_mirror_descent_algorithms_for_risk_budge
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/538_mirror_descent_algorithms_for_risk_budge.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c0a8f76240a0431b019ab49c34040a95a765860a on 2026-11-27 10:04 Europe/Oslo, exit 0, output sha256 425598b99088a568c1a5c52467ee1e986128f907bf85369589e49882a4876fcd
Data source: Yahoo Finance (yfinance) daily adjusted ETF closes for SPY, QQQ, TLT and GLD. BIL is also fetched, and its realized daily return is used for excess Sharpe reporting only. The loader uses a matching price cache if present and otherwise downloads the data. The run records the source as 'yfinance download' and the original fetch timestamp as 2026-11-27T08:18:30+00:00.
Data fetched: 2026-11-27T09:04:29+00:00
Sample period: Configured price window 2007-06-01 through 2025-12-31, with pre-backtest data used for rolling-window warm-up. Logged backtest sessions span 2009-01-02 through 2025-12-31.   data_range (from output): {'start': '2009-01-02', 'end': '2025-12-31'}
IS/OOS split: IS 2009-01-02 through 2018-12-31 inclusive. OOS 2019-01-02 through 2025-12-31. The main specification is simulated continuously across the cutoff and the resulting daily series is split for reporting. The three robustness alternatives run only on IS data.
Costs and slippage: Base friction is 2 bps transaction cost plus 3 bps modeled fixed slippage, totaling 5 bps on the absolute dollar amount of each buy and sell at each rebalance, including the initial purchase. The simulator marks the existing holdings to the execution close and solves for the post-cost portfolio value using the absolute differences between the resulting target dollar holdings and the marked holdings. It debits the friction on the execution date. The IS higher-friction case uses 6 bps cost plus 9 bps slippage.
Signal lag: The target uses daily returns through the final trading session of the month, close t. The configured one-session lag fills at the next trading session's close. The old holdings receive the execution day's close-to-close return, friction is booked at execution, and the new weights receive returns starting with the following close-to-close interval.
Price adjustment: yfinance auto_adjust=True, total-return daily closes
Survivorship bias: The universe is a fixed, retrospectively selected list of SPY, QQQ, TLT and GLD. The README identifies selection and survivorship bias and states that funds that had exited and alternatives available at the time were excluded.
Parameter combinations tried: 4
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09001801391864306, ann_vol=0.07845001447984494, sharpe=1.138121201968981, sharpe_excess=1.1053160131472926, max_drawdown=-0.11006413957676786, total_return=1.3645176888616115, turnover=5.305555076437441, n_trades=480
    oos: cagr=0.11871767823210821, ann_vol=0.11150219557860876, sharpe=1.062058193988751, sharpe_excess=0.835364842407527, max_drawdown=-0.24830268087345975, total_return=1.1891230075505805, turnover=2.9548176864135147, n_trades=336
    full: cagr=0.10174053575989217, ann_vol=0.09347233457236347, sharpe=1.083508532021487, sharpe_excess=0.9563839976796435, max_drawdown=-0.24830268087346075, total_return=4.176220074447265, turnover=8.260372762850956, n_trades=816
    benchmark  full: cagr=0.12040915777659045, ann_vol=0.10492450140736359, sharpe=1.1362798381054147, sharpe_excess=1.0226001170070418, max_drawdown=-0.25513828583784337, total_return=5.883915581486703, turnover=6.878530547059312, n_trades=816
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "538_ mirror_descent_algorithms_for_risk_budge"\nuniverse: [SPY, QQQ, TLT, GLD]\nrisk_free: BIL\ndata_start: 2007-06-01\nstart: 2009-01-01\nis_end: 2018-12-31\nend: 2025-12-31\n\nlookback_sessions: 252\nexecution_lag_sessions: 1\ncost_bps: 2.0\nslippage_bps: 3.0\n\nsolver:\n  step_exponent: 0.55\n  max_iterations: 20000\n  contribution_tolerance: 0.000001\n\nrobustness:\n  lookback_sessions: [189, 315]\n  friction_multiplier: 3.0\n'
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
"""Monthly volatility risk budgeting with deterministic entropic mirror descent."""

import json
import signal
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
FETCH_HEADER = "# fetched_at="
SOURCE_HEADER = "# source="


@dataclass(frozen=True)
class Specification:
    lookback_sessions: int
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def price_cache_path(tickers, start, end):
    symbols = "-".join(tickers)
    return CACHE_DIR / f"yf_adjusted_{symbols}_{start.date()}_{end.date()}.csv"


def validate_prices(prices, tickers, start, end):
    if not set(tickers).issubset(prices.columns):
        raise ValueError("Adjusted closing prices are missing for one or more symbols.")
    prices = prices.loc[start:end, tickers]
    dates = prices.index
    if (dates.empty or not dates.is_monotonic_increasing or dates.has_duplicates
            or dates[0] > start + pd.Timedelta(days=7)
            or dates[-1] < end - pd.Timedelta(days=7)):
        raise ValueError("Price history has invalid dates or incomplete coverage.")
    values = prices.to_numpy()
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Price history contains missing or nonpositive closes.")
    return prices


def save_prices(path, prices, fetched_at, source):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                            prefix=".prices_", delete=False) as stream:
        candidate = Path(stream.name)
    try:
        with candidate.open("w", encoding="utf-8") as stream:
            stream.write(f"{FETCH_HEADER}{fetched_at}\n{SOURCE_HEADER}{source}\n")
            prices.to_csv(stream, index_label="date")
        candidate.replace(path)
    finally:
        candidate.unlink(missing_ok=True)


def load_prices(tickers, start, end):
    path = price_cache_path(tickers, start, end)
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            fetched_at = stream.readline().strip().removeprefix(FETCH_HEADER)
            source = stream.readline().strip().removeprefix(SOURCE_HEADER)
            if datetime.fromisoformat(fetched_at).tzinfo is None or not source.startswith("yfinance"):
                raise ValueError(f"Missing timestamp or yfinance provenance in {path.name}.")
            prices = pd.read_csv(stream, index_col=0, parse_dates=True,
                                 dtype=dict.fromkeys(tickers, str)).astype(float)
        return validate_prices(prices, tickers, start, end), fetched_at, source

    import yfinance as yf

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=CACHE_DIR, prefix="yf_") as session_cache:
        yf.set_tz_cache_location(session_cache)
        bars = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, ignore_tz=True, multi_level_index=True,
            progress=False, threads=False, timeout=10,
        )
    if bars is None or bars.empty or "Close" not in bars:
        raise RuntimeError("yfinance returned no adjusted closing prices.")
    prices = validate_prices(bars["Close"].sort_index(), tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source = "yfinance download"
    save_prices(path, prices, fetched_at, source)
    return prices, fetched_at, source


def risk_budget_weights(history, *, step_exponent, max_iterations, contribution_tolerance):
    covariance = np.cov(history, rowvar=False, ddof=1)
    variances = covariance.diagonal()
    if not np.isfinite(covariance).all() or np.min(variances) <= 0:
        raise ValueError("Each asset must have positive finite volatility.")

    # Scaling covariance changes the optimizer's scale but leaves its proportions intact.
    covariance = covariance / variances.mean()
    smallest_eigenvalue = np.linalg.eigvalsh(covariance)[0]
    if smallest_eigenvalue <= 0:
        raise ValueError("The covariance matrix must be positive definite.")

    n_assets = covariance.shape[0]
    budget = np.full(n_assets, 1 / n_assets)
    # At the optimum y.T @ covariance @ y is one. This radius contains that optimum.
    radius = np.sqrt(n_assets / smallest_eigenvalue)
    log_radius = np.log(radius)
    scaled_weights = 1 / np.sqrt(covariance.diagonal())
    scaled_weights /= np.sqrt(scaled_weights @ covariance @ scaled_weights)

    marginal = covariance @ scaled_weights
    for iteration in range(1, max_iterations + 1):
        # Negative entropy step with the paper's gradient taming and L1 projection.
        tame = min(1.0, scaled_weights.min())
        step = iteration ** -step_exponent
        gradient = marginal - budget / scaled_weights
        log_candidate = np.log(scaled_weights) - step * tame * gradient
        excess = max(0.0, np.logaddexp.reduce(log_candidate) - log_radius)
        scaled_weights = np.exp(log_candidate - excess)
        marginal = covariance @ scaled_weights
        contributions = scaled_weights * marginal
        error = np.max(np.abs(contributions / contributions.sum() - budget))
        if error <= contribution_tolerance:
            return scaled_weights / scaled_weights.sum()

    raise RuntimeError(f"Mirror descent exceeded its iteration limit with contribution error {error:.3g}.")


def schedule_rebalances(dates, start, lag):
    months = dates.to_period("M")
    # The session before each monthly boundary supplies the signal.
    first_sessions = np.flatnonzero(months[1:] != months[:-1]) + 1
    execution_sessions = first_sessions + lag - 1
    valid = (execution_sessions >= dates.searchsorted(start)) & (execution_sessions < len(dates))
    return zip(first_sessions[valid] - 1, execution_sessions[valid], strict=True)


def make_targets(asset_returns, schedule, lookback, solver):
    returns = asset_returns.to_numpy()
    targets = {}
    for signal_session, execution_session in schedule:
        window_start = signal_session + 1 - lookback
        date = asset_returns.index[signal_session].date()
        if window_start < 0:
            raise ValueError(f"The {date} signal needs {lookback} returns. Set an earlier data_start.")
        try:
            targets[execution_session] = risk_budget_weights(
                returns[window_start:signal_session + 1], **solver,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to compute the {date} signal: {exc}") from exc
    if not targets:
        raise ValueError("The sample has no scheduled rebalances.")
    return targets


def simulate(asset_returns, targets, start, friction):
    dates = asset_returns.index
    first_session = dates.searchsorted(start)
    returns = asset_returns.to_numpy()
    weights = np.zeros(returns.shape[1])
    daily_results = []
    for session in range(first_session, len(dates)):
        marked = weights * (1 + returns[session])
        gross_growth = 1 + weights @ returns[session]
        weights = marked / gross_growth
        post_cost_fraction = 1.0
        turnover = 0.0
        n_trades = 0
        if session in targets:
            target = targets[session]

            def rebalance_gap(fraction):
                return fraction + friction * np.abs(fraction * target - weights).sum() - 1

            post_cost_fraction = brentq(rebalance_gap, 0, 1, xtol=1e-14)
            changes = np.abs(post_cost_fraction * target - weights)
            turnover = changes.sum()
            n_trades = np.count_nonzero(changes > 1e-10)
            weights = target
        daily_results.append((gross_growth * post_cost_fraction - 1, turnover, n_trades))
    return pd.DataFrame(daily_results, index=dates[first_session:],
                        columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period needs at least two daily returns.")
    excess = returns - risk_free.loc[returns.index]
    curve = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    ann_scale = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std()
    excess_volatility = excess.std()
    return {
        "cagr": float(curve[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(volatility * ann_scale),
        "sharpe": float(returns.mean() * ann_scale / volatility) if volatility else None,
        "sharpe_excess": float(excess.mean() * ann_scale / excess_volatility) if excess_volatility else None,
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1)),
        "total_return": float(curve[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], risk_free),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], risk_free),
        "full": metrics(daily_results, risk_free),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = config["universe"]
    risk_free_ticker = config["risk_free"]
    price_symbols = [*tickers, risk_free_ticker]
    if len(tickers) < 2 or len(set(price_symbols)) != len(price_symbols):
        raise ValueError("Use at least two distinct assets and a separate cash proxy.")
    data_start, start, is_end, end = (pd.Timestamp(config[key]) for key in
                                      ("data_start", "start", "is_end", "end"))
    lag = config["execution_lag_sessions"]
    if not data_start < start < is_end < end or type(lag) is not int or lag < 1:
        raise ValueError("Check the date order and use a delay of at least one session.")
    solver = config["solver"]
    if (not 0.5 < solver["step_exponent"] <= 1 or solver["max_iterations"] < 1
            or not 0 < solver["contribution_tolerance"] < 1):
        raise ValueError("Invalid mirror descent settings.")

    base = Specification(config["lookback_sessions"], config["cost_bps"], config["slippage_bps"])
    robustness = config["robustness"]
    cost_multiplier = robustness["friction_multiplier"]
    high_friction = Specification(base.lookback_sessions, base.cost_bps * cost_multiplier,
                                  base.slippage_bps * cost_multiplier)
    alternatives = dict.fromkeys(
        [Specification(window, base.cost_bps, base.slippage_bps)
         for window in robustness["lookback_sessions"]] + [high_friction]
    )
    alternatives.pop(base, None)
    for spec in (base, *alternatives):
        if (type(spec.lookback_sessions) is not int or spec.lookback_sessions <= len(tickers)
                or not 0 <= spec.cost_bps or not 0 <= spec.slippage_bps
                or not 0 <= spec.friction < 1):
            raise ValueError(f"Invalid lookback or friction: {spec}")

    prices, fetched_at, source = load_prices(price_symbols, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"DATA_SOURCE: {source}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, total-return daily closes")
    returns = prices.pct_change(fill_method=None).iloc[1:]
    assets = returns[tickers]
    risk_free = returns[risk_free_ticker]
    print(f"RISK_FREE: {risk_free_ticker}")

    is_assets = assets.loc[:is_end]
    is_risk_free = risk_free.loc[:is_end]
    print("ROBUSTNESS (IS only)")
    combos_tried = 0
    for spec in alternatives:
        combos_tried += 1
        targets = make_targets(is_assets, schedule_rebalances(is_assets.index, start, lag),
                               spec.lookback_sessions, solver)
        daily_results = simulate(is_assets, targets, start, spec.friction)
        print(json.dumps({**asdict(spec), "is": metrics(daily_results, is_risk_free)}, allow_nan=False))

    combos_tried += 1
    schedule = schedule_rebalances(assets.index, start, lag)
    targets = make_targets(assets, schedule, base.lookback_sessions, solver)
    daily_results = simulate(assets, targets, start, base.friction)
    benchmark_name = "Monthly equal weight"
    equal = np.full(len(tickers), 1 / len(tickers))
    benchmark_daily = simulate(assets, dict.fromkeys(targets, equal), start, base.friction)
    official = period_metrics(daily_results, risk_free, is_end)
    benchmark = period_metrics(benchmark_daily, risk_free, is_end)
    print(f"PARAM_COMBOS_TRIED: {combos_tried}")
    for period, summary in official.items():
        print(f"{period.upper()}: {json.dumps(summary, allow_nan=False)}")
        print(f"BENCHMARK {period.upper()} ({benchmark_name}): "
              f"{json.dumps(benchmark[period], allow_nan=False)}")

    print("RESULTS_JSON: " + json.dumps({
        "strategy": config["strategy"],
        "data_range": {
            "start": daily_results.index[0].date().isoformat(),
            "end": daily_results.index[-1].date().isoformat(),
        },
        **official,
        "benchmark": {"name": benchmark_name, **benchmark},
    }, allow_nan=False))


def stop_at_deadline(_signal_number, _frame):
    raise TimeoutError("The backtest exceeded its 14-minute limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_at_deadline)
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)

# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 720_kendall_correlation_coefficients_for_por\nuniverse:\n  - SPY\n  - IWM\n  - EFA\n  - EEM\n  - EWJ\n  - EWZ\n  - XLB\n  - XLE\n  - XLF\n  - XLI\n  - XLK\n  - XLP\n  - XLU\n  - XLV\n  - XLY\n  - VNQ\n  - IEF\n  - TLT\n  - TIP\n  - LQD\n  - HYG\n  - GLD\n  - SLV\n  - DBC\nrisk_free: BIL\nbenchmark: equal_weight_monthly\ndata_start: '2010-01-01'\nbacktest_start: '2011-01-03'\nis_end: '2019-12-31'\nend: '2025-12-31'\nlookback: 60\nestimator: kendall_icvc\ncv_folds: 10\neigenvalue_floor: 0.000001\nlag_sessions: 1\ninitial_capital: 100000\ncommission_bps: 2.0\nslippage_bps: 3.0\nrobustness:\n  lookbacks: [45, 75]\n  estimators: [kendall_sine, pearson_icvc]\n  friction_multiple: 2.0\n"
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
"""Research backtest for the Kendall ICVC minimum-variance ETF portfolio.

Paper: https://arxiv.org/abs/2410.17366
Full method: https://arxiv.org/html/2410.17366
Related code and research (ideas only, independently implemented):
- PortfolioOptimisers.jl (MIT): https://github.com/dcelisgarza/PortfolioOptimisers.jl/blob/628bd68f2ca943c3a57763fa555353aed03ebd1d/src/05_Moments/08_RankCovariance.jl
  https://raw.githubusercontent.com/dcelisgarza/PortfolioOptimisers.jl/628bd68f2ca943c3a57763fa555353aed03ebd1d/LICENSE
- Quantitative Finance Lab tutorial (license unclear): https://ramtin-asadi.github.io/Quantitative-Finance-Lab/notebooks/17_network_portfolio.html
- PyPortfolioOpt (MIT): https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/efficient_frontier/efficient_frontier.py
  https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/risk_models.py
  https://pypi.org/project/pyportfolioopt/
  https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/LICENSE
- Paper listings: https://tomasespana.github.io/
  https://paperswithbacktest.com/strategies/kendall-correlation-coefficients-for-portfolio-optimization
- Other minimum-variance examples: https://github.com/quantopian/algorithm-component-library/blob/master/portfolio_allocation.py
  https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/portfolio-construction/supported-models
Price basis: https://ranaroussi.github.io/yfinance/advanced/price_adjustment.html
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import isotonic_regression, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
FIELDS = ("Open", "Close", "Dividends", "Capital Gains")
ESTIMATORS = {"kendall_icvc", "kendall_sine", "pearson_icvc"}


@dataclass(frozen=True)
class Model:
    lookback: int
    estimator: str
    cv_folds: int
    eigenvalue_floor: float

    def __post_init__(self):
        if self.cv_folds < 2 or self.lookback < 2 * self.cv_folds:
            raise ValueError("Each of at least two folds needs at least two observations")
        if self.estimator not in ESTIMATORS or not 0 < self.eigenvalue_floor < 1:
            raise ValueError("Invalid covariance estimator or eigenvalue floor")


def fetch_prices(tickers, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    signature = sha256(",".join(tickers).encode()).hexdigest()[:12]
    cache_path = cache_dir / f"yf_daily_actions_{start.date()}_{end.date()}_{signature}.csv.gz"
    if not cache_path.exists():
        with TemporaryDirectory(dir=cache_dir) as request_cache:
            yf.set_tz_cache_location(request_cache)
            bars = yf.download(
                tickers,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=False,
                threads=4,
                timeout=10,
                progress=False,
            )
        if bars.empty:
            raise RuntimeError("yfinance returned an empty daily panel")
        columns = pd.MultiIndex.from_product([FIELDS, tickers])
        raw = bars.reindex(columns=columns)
        # Yahoo omits Capital Gains for symbols with no events of that type.
        raw["Capital Gains"] = raw["Capital Gains"].fillna(0.0)
        validate_prices(raw, start, end)
        with NamedTemporaryFile(dir=cache_dir, suffix=".csv.gz", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            raw.to_csv(temporary_path, compression="gzip")
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    raw = pd.read_csv(cache_path, header=[0, 1], index_col=0, parse_dates=True)
    validate_prices(raw, start, end)
    fetched_at = datetime.fromtimestamp(cache_path.stat().st_mtime, timezone.utc).isoformat()
    return raw, fetched_at


def validate_prices(raw, start, end):
    if raw.empty or not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("Daily panel is empty, duplicated, or out of order")
    if not 0 <= (raw.index[0] - start).days <= 7 or not 0 <= (end - raw.index[-1]).days <= 7:
        raise ValueError("Daily panel does not cover the configured date range")
    invalid = ~np.isfinite(raw) | raw.lt(0)
    invalid.loc[:, ["Open", "Close"]] |= raw.loc[:, ["Open", "Close"]].eq(0)
    if invalid.to_numpy().any():
        date_number, column_number = np.argwhere(invalid.to_numpy())[0]
        raise ValueError(f"Bad daily data for {raw.columns[column_number]} on {raw.index[date_number]}")


def total_return_prices(raw):
    # Ex-date distributions buy additional units at that day's open.
    distributions = raw["Dividends"] + raw["Capital Gains"]
    multiplier = (1.0 + distributions / raw["Open"]).cumprod()
    return raw["Open"] * multiplier, raw["Close"] * multiplier


def kendall_correlation(values):
    left, right = np.triu_indices(values.shape[0], k=1)
    signs = np.sign(values[left] - values[right])
    length = np.linalg.norm(signs, axis=0)
    if (length == 0).any():
        raise ValueError("Kendall correlation requires nonconstant samples")
    unit_signs = signs / length
    return unit_signs.T @ unit_signs


def standardize(training, test):
    center = training.mean(axis=0)

    def remove_session_scale(returns):
        centered = returns - center
        scale = np.linalg.norm(centered, axis=1, keepdims=True)
        return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)

    scaled_training = remove_session_scale(training)
    volatility = scaled_training.std(axis=0, ddof=1)
    if (volatility == 0).any():
        raise ValueError("Cannot standardize a constant return series")
    return scaled_training / volatility, remove_session_scale(test) / volatility


def icvc(returns, model):
    def eigenvectors(values):
        if model.estimator == "kendall_icvc":
            matrix = kendall_correlation(values)
        else:
            matrix = np.corrcoef(values, rowvar=False)
        return np.linalg.eigh(matrix)[1]

    fold_variances = []
    rows = np.arange(len(returns))
    for test_rows in np.array_split(rows, model.cv_folds):
        train_rows = np.setdiff1d(rows, test_rows, assume_unique=True)
        training, test = standardize(returns[train_rows], returns[test_rows])
        # Match eigenvalues by rank; the fold eigenvectors never see their test rows.
        fold_variances.append((test @ eigenvectors(training)).var(axis=0, ddof=1))

    empirical_directions = eigenvectors(standardize(returns, returns)[0])
    projected_variances = np.mean(fold_variances, axis=0)
    eigenvalues = isotonic_regression(projected_variances).x
    eigenvalues = np.maximum(eigenvalues, model.eigenvalue_floor)
    return (empirical_directions * eigenvalues) @ empirical_directions.T


def estimate_covariance(returns, model):
    if model.estimator == "kendall_sine":
        transformed = np.sin(np.pi / 2 * kendall_correlation(returns))
        eigenvalues, directions = np.linalg.eigh(transformed)
        spectrum = np.maximum(eigenvalues, model.eigenvalue_floor)
        matrix = (directions * spectrum) @ directions.T
    else:
        matrix = icvc(returns, model)

    marginal_volatility = returns.std(axis=0, ddof=1)
    scales = marginal_volatility / np.sqrt(np.diag(matrix))
    return matrix * np.outer(scales, scales)


def min_variance_weights(covariance):
    asset_count = len(covariance)
    scaled = covariance / np.diag(covariance).mean()
    total_weight = np.ones(asset_count)
    result = minimize(
        lambda weights: weights @ scaled @ weights,
        x0=total_weight / asset_count,
        jac=lambda weights: 2.0 * (scaled @ weights),
        bounds=[(0.0, 1.0)] * asset_count,
        constraints={
            "type": "eq",
            "fun": lambda weights: weights.sum() - 1.0,
            "jac": lambda weights: total_weight,
        },
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 200},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")
    weights = np.where(result.x > 1e-10, result.x, 0.0)
    return weights / weights.sum()


def rebalance_sessions(sessions, start, lookback, lag):
    months = sessions.to_period("M")
    month_ends = np.flatnonzero(months[:-1] != months[1:])
    fills = {}
    for signal_row in month_ends:
        fill_row = signal_row + lag
        if fill_row >= len(sessions) or sessions[fill_row] < start:
            continue
        if signal_row < lookback:
            raise ValueError("Insufficient warm-up for the first monthly signal")
        fills[sessions[fill_row]] = signal_row
    return fills


def make_targets(closes, model, signal_rows):
    returns = closes.pct_change().to_numpy()
    targets = {}
    for fill, signal_row in signal_rows.items():
        window = returns[signal_row + 1 - model.lookback : signal_row + 1]
        covariance = estimate_covariance(window, model)
        targets[fill] = min_variance_weights(covariance)
    return targets


def rebalance(positions, cash, opening, weights, friction):
    current_values = positions * opening
    equity = cash + current_values.sum()
    # Solve for post-cost exposure. The objective is monotonic when friction < 1.
    spendable = equity * (1 - 1e-12)
    low, high = 0.0, spendable
    for _ in range(48):
        budget = (low + high) / 2
        traded = np.abs(budget * weights - current_values).sum()
        if budget + friction * traded > spendable:
            high = budget
        else:
            low = budget

    target_values = low * weights
    changes = np.abs(target_values - current_values)
    gross = changes.sum()
    cash = equity - target_values.sum() - friction * gross
    if cash < 0:
        raise RuntimeError("A rebalance required negative cash")
    return target_values / opening, cash, gross / equity, int((changes > 1e-9 * equity).sum())


def simulate(opens, closes, targets, start, initial_capital, friction):
    eval_opens = opens.loc[start:]
    eval_closes = closes.loc[start:]
    positions = np.zeros(opens.shape[1])
    cash = previous_value = initial_capital
    rows = []
    for date, opening, closing in zip(
        eval_opens.index, eval_opens.to_numpy(), eval_closes.to_numpy(), strict=True
    ):
        turnover = trades = 0
        if date in targets:
            positions, cash, turnover, trades = rebalance(
                positions, cash, opening, targets[date], friction
            )
        value = cash + positions @ closing
        rows.append((value / previous_value - 1.0, turnover, trades))
        previous_value = value
    return pd.DataFrame(rows, index=eval_opens.index, columns=["return", "turnover", "trades"])


def metrics(daily_results, cash_proxy):
    if len(daily_results) < 2:
        raise ValueError("Each evaluation period needs at least two sessions")
    returns = daily_results["return"].to_numpy()
    excess = returns - cash_proxy.loc[daily_results.index].to_numpy()
    growth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    root_year = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / returns.size) - 1.0),
        "ann_vol": float(volatility * root_year),
        "sharpe": float(returns.mean() / volatility * root_year) if volatility else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * root_year) if excess_volatility else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["trades"].sum()),
    }


def period_metrics(daily_results, cash_proxy, is_end):
    within_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[within_sample], cash_proxy),
        "oos": metrics(daily_results.loc[~within_sample], cash_proxy),
        "full": metrics(daily_results, cash_proxy),
    }


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    data_start, start, is_end, end = (pd.Timestamp(cfg[key]) for key in (
        "data_start", "backtest_start", "is_end", "end"
    ))
    lag = cfg["lag_sessions"]
    universe = cfg["universe"]
    tickers = universe + [cfg["risk_free"]]
    initial_capital = cfg["initial_capital"]
    commissions, slippage = cfg["commission_bps"], cfg["slippage_bps"]
    friction = (commissions + slippage) / 10_000
    robustness = cfg["robustness"]
    stress = robustness["friction_multiple"]
    model = Model(cfg["lookback"], cfg["estimator"], cfg["cv_folds"], cfg["eigenvalue_floor"])
    if (
        not data_start < start <= is_end < end
        or type(lag) is not int or lag < 1
        or len(universe) < 2 or len(set(tickers)) != len(tickers)
        or not initial_capital > 0
        or not (commissions >= 0 and slippage >= 0 and stress > 1 and stress * friction < 1)
        or cfg["benchmark"] != "equal_weight_monthly"
    ):
        raise ValueError("Invalid dates, universe, execution assumptions, or benchmark")

    variants = [
        (f"lookback_{lookback}", replace(model, lookback=lookback), friction)
        for lookback in robustness["lookbacks"]
    ]
    variants.extend(
        (estimator, replace(model, estimator=estimator), friction)
        for estimator in robustness["estimators"]
    )
    variants.append((f"friction_{stress:g}x", model, friction * stress))
    if len({(variant, rate) for _, variant, rate in variants} | {(model, friction)}) != 1 + len(variants):
        raise ValueError("Robustness settings duplicate a parameter combination")

    raw, fetched_at = fetch_prices(tickers, data_start, end)
    opens, closes = total_return_prices(raw)
    cash_proxy = closes[cfg["risk_free"]].pct_change()
    opens, closes = opens[universe], closes[universe]
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance split-adjusted OHLC; forward total return via ex-date open reinvestment")

    is_opens, is_closes = opens.loc[:is_end], closes.loc[:is_end]
    print("ROBUSTNESS (IS only)")
    trials = 0
    for label, variant, rate in variants:
        signal_rows = rebalance_sessions(is_closes.index, start, variant.lookback, lag)
        targets = make_targets(is_closes, variant, signal_rows)
        daily = simulate(is_opens, is_closes, targets, start, initial_capital, rate)
        trials += 1
        print(f"  {label}: {json.dumps(metrics(daily, cash_proxy), allow_nan=False)}")

    signal_rows = rebalance_sessions(closes.index, start, model.lookback, lag)
    targets = make_targets(closes, model, signal_rows)
    daily = simulate(opens, closes, targets, start, initial_capital, friction)
    trials += 1
    official = period_metrics(daily, cash_proxy, is_end)
    if min(official["is"]["n_trades"], official["oos"]["n_trades"]) <= 0:
        raise RuntimeError("The official run must execute trades in both evaluation periods")

    equal_weights = np.full(len(universe), 1.0 / len(universe))
    benchmark_targets = dict.fromkeys(signal_rows, equal_weights)
    benchmark_daily = simulate(opens, closes, benchmark_targets, start, initial_capital, friction)
    benchmark = period_metrics(benchmark_daily, cash_proxy, is_end)
    for period, result in official.items():
        print(f"{period.upper()} strategy: {json.dumps(result, allow_nan=False)}")
        print(f"{period.upper()} benchmark ({cfg['benchmark']}): {json.dumps(benchmark[period], allow_nan=False)}")

    results = {
        "strategy": cfg["strategy"],
        "data_range": {"start": daily.index[0].date().isoformat(), "end": daily.index[-1].date().isoformat()},
        **official,
        "benchmark": {"name": cfg["benchmark"], **benchmark["full"], **benchmark},
    }
    print(f"PARAM_COMBOS_TRIED: {trials}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

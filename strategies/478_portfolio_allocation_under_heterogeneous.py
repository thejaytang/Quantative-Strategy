# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 478_portfolio_allocation_under_heterogeneous\nuniverse: [SPY, EWJ, USO, GLD]\nrisk_free_ticker: BIL\ndata_start: '2011-01-01'\nbacktest_start: '2014-01-01'\nis_end: '2020-12-31'\nbacktest_end: '2025-12-31'\nannualization_sessions: 252\ncost_bps: 5.0\nslippage_bps: 5.0\nexecution_lag_sessions: 1\ncash_buffer_bps: 50.0\nmodel:\n  lookback_sessions: 520\n  scales: [5, 30, 55, 80, 105, 130]\n  fluctuation_orders: [1, 1.5, 2, 2.5, 3, 3.5, 4]\n  detrend_degree: 2\n  annual_required_return: 0.01\n"
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
import hashlib
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
TRADE_TOLERANCE = 1e-8


class EquityCalendar(AbstractHolidayCalendar):
    # U.S. equity sessions in 2011–2025, including shortened sessions.
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("Hurricane Sandy", year=2012, month=10, day=29),
        Holiday("Hurricane Sandy", year=2012, month=10, day=30),
        Holiday("George H. W. Bush national day of mourning", year=2018, month=12, day=5),
        Holiday("Jimmy Carter national day of mourning", year=2025, month=1, day=9),
    ]


def load_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for name in ("data_start", "backtest_start", "is_end", "backtest_end"):
        config[name] = pd.Timestamp(config[name])
    if not config["data_start"] < config["backtest_start"] <= config["is_end"] < config["backtest_end"]:
        raise ValueError("The data, backtest, and split dates are out of order")
    lag = config["execution_lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("execution_lag_sessions must be a positive integer")
    cost_bps, slippage_bps = config["cost_bps"], config["slippage_bps"]
    if not (0 <= cost_bps and 0 <= slippage_bps and cost_bps + slippage_bps < 10_000):
        raise ValueError("Costs and slippage must be nonnegative and total less than 10,000 bps")
    if not 0 < config["cash_buffer_bps"] < 10_000:
        raise ValueError("cash_buffer_bps must be between 0 and 10,000, exclusive")
    symbols = [*config["universe"], config["risk_free_ticker"]]
    if len(symbols) < 3 or len(set(symbols)) != len(symbols):
        raise ValueError("Use at least two unique assets and a distinct risk-free ticker")
    model = config["model"]
    if not model["scales"] or any(
        not model["detrend_degree"] + 1 < scale <= model["lookback_sessions"]
        for scale in model["scales"]
    ):
        raise ValueError("Each scale must fit the window and leave degrees of freedom after detrending")
    if not model["fluctuation_orders"] or not all(q > 0 for q in model["fluctuation_orders"]):
        raise ValueError("The model requires strictly positive fluctuation orders")
    return config


def check_prices(prices, start, end):
    if start < pd.Timestamp("2011-01-01") or end > pd.Timestamp("2025-12-31"):
        raise ValueError("The equity calendar covers 2011 through 2025")
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Prices must have unique, ascending session dates")
    expected = pd.bdate_range(start, end).difference(EquityCalendar().holidays(start, end))
    missing = expected.difference(prices.index)
    unexpected = prices.index.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Price dates differ from the U.S. equity calendar. "
            f"Missing: {missing.strftime('%Y-%m-%d').tolist()}; "
            f"unexpected: {unexpected.strftime('%Y-%m-%d').tolist()}"
        )
    valid = np.isfinite(prices) & (prices > 0)
    if not valid.to_numpy().all():
        invalid = prices.columns[~valid.all()].tolist()
        raise ValueError(f"Missing or invalid daily prices for {invalid}")


def load_prices(config):
    symbols = [*config["universe"], config["risk_free_ticker"]]
    start, end = config["data_start"], config["backtest_end"]
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(repr((sorted(symbols), str(start), str(end))).encode()).hexdigest()[:16]
    columns = pd.MultiIndex.from_product((("Open", "Close"), symbols))
    path = cache / f"yfinance_adjusted_open_close_{key}.json"
    if path.exists():
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        price_csv = snapshot["prices_csv"]
        if hashlib.sha256(price_csv.encode("utf-8")).hexdigest() != snapshot["sha256"]:
            raise ValueError(f"Price cache checksum mismatch: {path.name}")
        fetched_at = datetime.fromisoformat(snapshot["fetched_at"])
        if fetched_at.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError(f"Price cache timestamp must be UTC: {path.name}")
        prices = pd.read_csv(
            StringIO(price_csv), header=[0, 1], index_col=0, parse_dates=True
        ).loc[:, columns]
        check_prices(prices, start, end)
        return prices, fetched_at, snapshot["sha256"]

    with TemporaryDirectory(dir=cache) as session_cache:
        yf.set_tz_cache_location(session_cache)
        downloaded = yf.download(
            symbols,
            start=start.strftime("%Y-%m-%d"),
            end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            group_by="column",
            ignore_tz=True,
            progress=False,
            timeout=15,
        )
        fetched_at = datetime.now(timezone.utc)
    if downloaded.empty:
        raise RuntimeError("yfinance returned no daily price data")
    prices = downloaded.reindex(columns=columns).sort_index()
    check_prices(prices, start, end)
    price_csv = prices.to_csv(index_label="Date")
    snapshot = {
        "fetched_at": fetched_at.isoformat(),
        "sha256": hashlib.sha256(price_csv.encode("utf-8")).hexdigest(),
        "prices_csv": price_csv,
    }
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(snapshot, temporary)
    temporary_path.replace(path)
    return prices, fetched_at, snapshot["sha256"]


def detrend_basis(scale, degree):
    time = np.linspace(-1.0, 1.0, scale)
    design = np.polynomial.polynomial.polyvander(time, degree)
    return np.linalg.qr(design, mode="reduced")[0]


def local_covariances(profile, scale, basis):
    # Divide from both ends, retaining the time order within each segment.
    segment_count = len(profile) // scale
    span = segment_count * scale
    blocks = np.stack((profile[:span], profile[-span:])).reshape(-1, scale, profile.shape[1])
    residuals = blocks - basis @ (basis.T @ blocks)
    return residuals.transpose(0, 2, 1) @ residuals / scale


def signed_fluctuations(covariances, order):
    return (np.sign(covariances) * np.abs(covariances) ** (order / 2)).mean(axis=0)


def minimum_risk(matrix, expected, required):
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    if eigenvalues[-1] > 0:
        matrix = (eigenvectors * (eigenvalues / eigenvalues[-1])) @ eigenvectors.T
    else:
        matrix = np.zeros_like(matrix)

    size = len(expected)
    max_return = expected.max()
    if required >= max_return:
        # When the floor is beyond reach, use the highest estimated return.
        eligible = expected == max_return
        weights = np.zeros(size)
        if eligible.sum() == 1:
            weights[eligible] = 1.0
        else:
            weights[eligible] = minimum_risk(matrix[np.ix_(eligible, eligible)], expected[eligible], -np.inf)
        return weights

    unit = np.ones(size)
    initial = unit / size
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: unit}]
    if required > expected.min():
        span = max_return - expected.min()
        scaled_returns = (expected - required) / span
        constraints.append({"type": "ineq", "fun": lambda w: scaled_returns @ w, "jac": lambda w: scaled_returns})
        if scaled_returns @ initial < 0:
            initial = np.zeros(size)
            initial[expected.argmax()] = 1.0

    solution = minimize(
        lambda w: w @ matrix @ w,
        initial,
        jac=lambda w: 2 * (matrix @ w),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * size,
        constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 150},
    )
    weights = np.maximum(solution.x, 0)
    weights /= weights.sum()
    if not solution.success or not np.isfinite(weights).all() or expected @ weights < required - 1e-10:
        raise RuntimeError(f"Portfolio optimization failed: {solution.message}")
    return weights


def build_targets(log_returns, config):
    model = config["model"]
    lookback = model["lookback_sessions"]
    lag = config["execution_lag_sessions"]
    required = np.log1p(model["annual_required_return"]) / config["annualization_sessions"]
    bases = {scale: detrend_basis(scale, model["detrend_degree"]) for scale in model["scales"]}
    sessions = log_returns.index
    first_sessions = np.flatnonzero(~sessions.to_period("M").duplicated())
    strategy_targets, benchmark_targets = {}, {}
    for first_session in first_sessions:
        signal = first_session - 1
        execution = signal + lag
        if signal < lookback or execution >= len(sessions):
            continue
        date = sessions[execution]
        if date < config["backtest_start"]:
            continue

        history = log_returns.iloc[signal - lookback + 1 : signal + 1].to_numpy()
        expected = history.mean(axis=0)
        profile = np.cumsum(history - expected, axis=0)
        allocations = []
        for scale, basis in bases.items():
            covariances = local_covariances(profile, scale, basis)
            for order in model["fluctuation_orders"]:
                matrix = signed_fluctuations(covariances, order)
                allocations.append(minimum_risk(matrix, expected, required))
        strategy_targets[date] = np.mean(allocations, axis=0)
        covariance = np.cov(history, rowvar=False)
        benchmark_targets[date] = minimum_risk(covariance, expected, required)

    return strategy_targets, benchmark_targets


def planned_changes(positions, previous_close, previous_value, target, cash_buffer):
    target_units = target * (previous_value * (1.0 - cash_buffer)) / previous_close
    changes = target_units - positions
    changes[np.abs(changes) * previous_close <= TRADE_TOLERANCE * previous_value] = 0.0
    return changes


def fill_at_open(changes, opening, cash, fee_rate):
    sales = np.maximum(-changes, 0.0)
    buys = np.maximum(changes, 0.0)
    proceeds = float(sales @ opening)
    available = cash + proceeds * (1.0 - fee_rate)
    buy_value = float(buys @ opening)
    buy_cost = buy_value * (1.0 + fee_rate)
    if buy_cost > available:
        # Only reduce buys when the opening prices would exhaust the cash budget.
        scale = np.nextafter(available / buy_cost, 0.0)
        buys *= scale
        buy_value *= scale
        buy_cost *= scale
    return buys - sales, available - buy_cost, proceeds + buy_value


def simulate(opens, closes, targets, config):
    sessions = closes.index
    first = sessions.searchsorted(config["backtest_start"])
    if first == 0:
        raise ValueError("The backtest requires a prior close for sizing")
    opening_prices, closing_prices = opens.to_numpy(), closes.to_numpy()
    fee_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    cash_buffer = config["cash_buffer_bps"] / 10_000
    positions = np.zeros(closes.shape[1])
    cash = 1.0
    daily_results = []
    for i in range(first, len(sessions)):
        date = sessions[i]
        previous_close = closing_prices[i - 1]
        previous_value = cash + positions @ previous_close
        turnover = 0.0
        trades = 0
        if date in targets:
            changes = planned_changes(positions, previous_close, previous_value, targets[date], cash_buffer)
            opening = opening_prices[i]
            opening_value = cash + positions @ opening
            fills, cash, gross_amount = fill_at_open(changes, opening, cash, fee_rate)
            positions += fills
            if cash < 0.0 or (positions < 0.0).any():
                raise RuntimeError(f"Negative cash or position after the opening fills on {date.date()}")
            turnover = gross_amount / opening_value
            trades = int(np.count_nonzero(fills))
        closing_value = cash + positions @ closing_prices[i]
        daily_results.append((closing_value / previous_value - 1.0, turnover, trades))
    return pd.DataFrame(daily_results, index=sessions[first:], columns=["return", "turnover", "trades"])


def metrics(daily_results, risk_free, annualization):
    returns = daily_results["return"].to_numpy()
    excess = returns - risk_free.loc[daily_results.index].to_numpy()
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    volatility = returns.std(ddof=1) if len(returns) > 1 else None
    excess_volatility = excess.std(ddof=1) if len(excess) > 1 else None
    return {
        "cagr": float(growth[-1] ** (annualization / len(returns)) - 1) if len(returns) else None,
        "ann_vol": float(volatility * np.sqrt(annualization)) if volatility is not None else None,
        "sharpe": float(returns.mean() / volatility * np.sqrt(annualization)) if volatility else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * np.sqrt(annualization)) if excess_volatility else None,
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1).min()) if len(returns) else None,
        "total_return": float(growth[-1] - 1) if len(returns) else None,
        "turnover": float(daily_results["turnover"].mean() * annualization) if len(returns) else None,
        "n_trades": int(daily_results["trades"].sum()),
    }


def main():
    config = load_config()
    prices, fetched_at, price_digest = load_prices(config)
    opens = prices["Open"][config["universe"]]
    closes = prices["Close"][config["universe"]]
    risk_free = prices["Close"][config["risk_free_ticker"]].pct_change(fill_method=None)
    targets, benchmark_targets = build_targets(np.log(closes).diff(), config)
    daily_results = simulate(opens, closes, targets, config)
    benchmark_results = simulate(opens, closes, benchmark_targets, config)

    sessions = daily_results.index
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": sessions[0].date().isoformat(), "end": sessions[-1].date().isoformat()},
    }
    in_sample = sessions <= config["is_end"]
    annualization = config["annualization_sessions"]
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print(f"DATA_SHA256: {price_digest}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return adjusted open and close")
    # The original model configuration is the only one run by this script.
    print("PARAM_COMBOS_TRIED: 1")
    print("BENCHMARK: conventional long-only mean-variance")
    for period, selector in (("is", in_sample), ("oos", ~in_sample), ("full", slice(None))):
        results[period] = metrics(daily_results.loc[selector], risk_free, annualization)
        comparison = metrics(benchmark_results.loc[selector], risk_free, annualization)
        print(f"{period.upper()}: {json.dumps(results[period], allow_nan=False)}")
        print(f"BENCHMARK_{period.upper()}: {json.dumps(comparison, allow_nan=False)}")
    results["benchmark"] = comparison
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

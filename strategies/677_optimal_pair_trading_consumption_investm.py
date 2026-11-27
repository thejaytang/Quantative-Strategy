# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 677_optimal_pair_trading_consumption_investm\n\nuniverse:\n  pair: [XOM, CVX]\n  benchmark: SPY\n  excess_return_proxy: BIL\n\ndates:\n  warmup_start: '2009-01-02'\n  start: '2012-01-03'\n  is_end: '2023-12-29'\n  end: '2025-12-31'\n\nsignal:\n  lookback_sessions: 252\n  cointegration_p_max: 0.10\n  cointegration_adf_lags: 1\n  horizon_sessions: 63\n  utility_power: 0.50\n  terminal_utility_weight: 1.0\n  feedback_fraction: 0.01\n  gross_weight_cap: 0.60\n\nexecution:\n  lag_sessions: 1\n  initial_capital: 100000.0\n  commission_bps: 2.0\n  slippage_bps: 3.0\n  borrow_bps_annual: 50.0\n  cash_rate_annual: 0.0\n\nrobustness:\n  lookback_sessions: [189, 315]\n  horizon_sessions: [42, 84]\n  friction_multiplier: 2.0\n"
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
"""C250 research backtest using Kabanov and Kozhevnikov (2023).

Paper: https://arxiv.org/abs/2312.06842
Research reference: Hudson & Thames, ArbitrageLab, BSD-3-Clause.
https://raw.githubusercontent.com/hudson-and-thames/arbitragelab/master/arbitragelab/stochastic_control_approach/ou_model_mudchanatongsuk.py
https://github.com/hudson-and-thames/arbitragelab/blob/master/LICENSE.txt
https://pypi.org/project/arbitragelab/
Research reference: QuantConnect, Optimal Pairs Trading tutorial.
https://www.quantconnect.com/research/15294/optimal-pairs-trading/p1
The implementation is original and uses equations (4.6)–(4.12) of C250.
"""

import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.special import softmax
from statsmodels.tsa.stattools import coint


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
QUAD_NODES, QUAD_WEIGHTS = np.polynomial.legendre.leggauss(24)
BAR_COLUMNS = ["Open", "Close", "Distributions", "Stock Splits"]


def validate_bars(bars, symbol, start, end):
    if (bars.empty or bars.index[0] != start or bars.index[-1] != end
            or not bars.index.is_unique or not bars.index.is_monotonic_increasing
            or not np.isfinite(bars[BAR_COLUMNS].to_numpy()).all()
            or (bars[["Open", "Close"]] <= 0).any().any()):
        raise ValueError(f"Incomplete or invalid daily history for {symbol}: {start.date()} to {end.date()}")


def read_history(symbol, start, end):
    path = CACHE / f"{symbol}_{start.date()}_{end.date()}_ohlc_actions.csv"
    if path.exists():
        with path.open(encoding="utf-8") as cached:
            fetched_at = datetime.fromisoformat(cached.readline().removeprefix("# ").strip())
            bars = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
        validate_bars(bars, symbol, start, end)
        return bars, fetched_at

    ticker = yf.Ticker(symbol)
    for attempt in range(2):
        try:
            history = ticker.history(
                start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=False, back_adjust=False, actions=True,
                repair=False, keepna=True, timeout=15,
            )
            bars = history[["Open", "Close", "Stock Splits"]].assign(
                Distributions=history["Dividends"] + history.get("Capital Gains", 0.0)
            )[BAR_COLUMNS]
            bars.index = bars.index.tz_localize(None).rename("Date")
            validate_bars(bars, symbol, start, end)
            break
        except Exception as exc:
            if attempt == 1:
                raise RuntimeError(f"Could not load Yahoo Finance daily history for {symbol}") from exc

    fetched_at = datetime.now(timezone.utc)
    with NamedTemporaryFile(mode="w", dir=CACHE, encoding="utf-8", delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(f"# {fetched_at.isoformat()}\n")
            bars.to_csv(temporary)
            temporary.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return bars, fetched_at


def load_market(symbols, warmup_start, end):
    CACHE.mkdir(exist_ok=True)
    histories = {}
    timestamps = {}
    with TemporaryDirectory(prefix="yf-", dir=CACHE) as metadata_dir:
        yf.set_tz_cache_location(metadata_dir)
        yf.config.debug.hide_exceptions = False
        for symbol in symbols:
            histories[symbol], timestamps[symbol] = read_history(symbol, warmup_start, end)

    dates = histories[symbols[0]].index
    for symbol, bars in histories.items():
        if not bars.index.equals(dates):
            raise ValueError(f"{symbol} dates differ from {symbols[0]}; no missing bars are imputed")

    market = {column: pd.DataFrame({symbol: histories[symbol][column] for symbol in symbols})
              for column in BAR_COLUMNS[:3]}
    print(f"DATA_FETCHED_AT: {min(timestamps.values()).isoformat()}")
    for symbol, fetched_at in timestamps.items():
        splits = int(histories[symbol]["Stock Splits"].ne(0).sum())
        print(f"DATA_SOURCE {symbol}: {fetched_at.isoformat()} | split events: {splits}")
    print("PRICE_ADJUSTMENT: Yahoo split-adjusted OHLC, explicit ex-date cash distributions, "
          "forward total-return signal indices")
    return market


def forward_total_returns(closes, distributions):
    return (closes + distributions) / closes.shift(1) - 1.0


def feedback(state, mean_reversion, diffusion_variance, cash_rate, model):
    """Return alpha/x from the paper, in normalized spread units per dollar."""
    gamma = model["utility_power"]
    kappa_gamma = (mean_reversion + cash_rate) / (1.0 - gamma)
    root_gap = np.sqrt((mean_reversion**2 - gamma * cash_rate**2) / (1.0 - gamma))
    upper_root = gamma * kappa_gamma + mean_reversion + root_gap
    lower_root = gamma * kappa_gamma**2 / upper_root
    horizon = model["horizon_sessions"] / SESSIONS_PER_YEAR
    ages = horizon * np.r_[0.5 * (QUAD_NODES + 1.0), 1.0]

    # q = sigma^2 * g at each age, and its integral from age zero.
    decay = -np.expm1(-2.0 * root_gap * ages)
    q = lower_root * upper_root * decay / (2.0 * root_gap + lower_root * decay)
    q_integral = lower_root * ages - np.log1p(lower_root * decay / (2.0 * root_gap))
    log_kernel = (0.5 * q_integral + cash_rate * gamma * ages / (1.0 - gamma)
                  + 0.5 * state**2 * q / diffusion_variance)
    log_mass = log_kernel + np.r_[np.log(horizon * QUAD_WEIGHTS / 2.0),
                                 np.log(model["terminal_utility_weight"]) / (1.0 - gamma)]
    # The inhomogeneous consumption term contributes all quadrature nodes.
    derivative = state / diffusion_variance * (softmax(log_mass) @ q)
    return derivative - state * kappa_gamma / diffusion_variance


def pair_weights(window, latest, model, cash_rate):
    """Estimate the hedge and exact discrete OU transition on past closes."""
    first, second = window.T
    centered_second = second - second.mean()
    second_variation = centered_second @ centered_second
    if second_variation <= 0.0 or np.ptp(first) <= 0.0:
        return np.zeros(2)

    hedge = (centered_second @ (first - first.mean())) / second_variation
    if hedge <= 0.0:
        return np.zeros(2)
    intercept = first.mean() - hedge * second.mean()
    spread = first - hedge * second - intercept
    if np.std(spread) <= 1e-10:
        return np.zeros(2)
    _, p_value, _ = coint(first, second, trend="c", maxlag=model["cointegration_adf_lags"], autolag=None)
    if not np.isfinite(p_value) or p_value > model["cointegration_p_max"]:
        return np.zeros(2)

    lagged_spread = spread[:-1]
    next_spread = spread[1:]
    predictors = np.column_stack((np.ones(len(lagged_spread)), lagged_spread))
    drift, persistence = np.linalg.lstsq(predictors, next_spread, rcond=None)[0]
    if not 0.0 < persistence < 1.0:
        return np.zeros(2)
    innovations = next_spread - drift - persistence * lagged_spread
    mean_reversion = -SESSIONS_PER_YEAR * np.log(persistence)
    diffusion_variance = 2.0 * mean_reversion * np.mean(innovations**2) / (1.0 - persistence**2)
    if mean_reversion < cash_rate or diffusion_variance <= 1e-14:
        return np.zeros(2)

    ou_center = drift / (1.0 - persistence)
    state = latest[0] - hedge * latest[1] - intercept - ou_center
    spread_loading = feedback(state, mean_reversion, diffusion_variance, cash_rate, model)
    weights = model["feedback_fraction"] * spread_loading * latest * np.array([1.0, -hedge])
    gross = np.abs(weights).sum()
    if gross > model["gross_weight_cap"]:
        weights *= model["gross_weight_cap"] / gross
    return weights


def opening_targets(indices, model, execution):
    lookback = model["lookback_sessions"]
    cash_rate = execution["cash_rate_annual"]
    weeks = indices.index.to_period("W-SUN")
    values = indices.to_numpy()
    targets = np.full_like(values, np.nan)
    decisions = np.flatnonzero(weeks[1:] != weeks[:-1]) + 1
    for row in decisions[decisions >= lookback]:
        targets[row] = pair_weights(values[row - lookback:row], values[row], model, cash_rate)
    return pd.DataFrame(targets, index=indices.index, columns=indices.columns).shift(execution["lag_sessions"])


def simulate(market, targets, start, execution):
    closes = market["Close"][targets.columns].to_numpy()
    opens = market["Open"][targets.columns].to_numpy()
    distributions = market["Distributions"][targets.columns].to_numpy()
    dates = targets.index
    start_row = dates.get_loc(start)
    dollar_targets = targets.to_numpy()
    initial_capital = execution["initial_capital"]
    cost_rate = (execution["commission_bps"] + execution["slippage_bps"]) / 10_000.0
    borrow_rate = execution["borrow_bps_annual"] / 10_000.0

    positions = np.zeros(targets.shape[1])
    cash = previous_value = initial_capital
    daily_results = []
    for row in range(start_row, len(dates)):
        days = (dates[row] - dates[row - 1]).days
        prior_short = np.maximum(-positions * closes[row - 1], 0.0).sum()
        cash += max(cash - prior_short, 0.0) * np.expm1(execution["cash_rate_annual"] * days / 365.0)
        cash -= prior_short * borrow_rate * days / 365.0
        if cash < 0.0:
            raise RuntimeError(f"Insufficient cash for financing on {dates[row].date()}")

        # Ex-date entitlement belongs to prior-close positions. Settle after the open fills.
        distribution_cash = positions @ distributions[row]
        open_values = positions * opens[row]
        opening_value = cash + open_values.sum()
        if opening_value <= 0.0:
            raise RuntimeError(f"Capital exhausted at the open on {dates[row].date()}")
        target = dollar_targets[row]
        traded, trade_count = 0.0, 0
        if cash < np.maximum(-open_values, 0.0).sum():
            target = np.zeros_like(positions)
        if np.isfinite(target).all():
            gross_target = np.abs(target).sum()
            # Reserve an upper bound on costs from both exiting old and opening new positions.
            budget = max(0.0, opening_value - np.abs(open_values).sum() * cost_rate)
            budget /= 1.0 + gross_target * cost_rate
            new_values = target * budget
            changes = np.abs(new_values - open_values)
            traded = changes.sum()
            trade_count = int(np.count_nonzero(changes))
            cash = opening_value - new_values.sum() - traded * cost_rate
            if cash < np.maximum(-new_values, 0.0).sum() - opening_value * 1e-12:
                raise RuntimeError(f"Insufficient cash or short collateral on {dates[row].date()}")
            cash = max(cash, 0.0)
            positions = new_values / opens[row]

        cash += distribution_cash
        closing_value = cash + positions @ closes[row]
        if cash < 0.0 or closing_value <= 0.0:
            raise RuntimeError(f"Insufficient cash or capital at the close on {dates[row].date()}")
        daily_results.append((closing_value / previous_value - 1.0, traded / opening_value, trade_count, cash))
        previous_value = closing_value

    return pd.DataFrame(daily_results, index=dates[start_row:], columns=["return", "turnover", "trades", "cash"])


def metrics(daily, cash_returns):
    returns = daily["return"].to_numpy()
    excess = returns - cash_returns.loc[daily.index].to_numpy()
    if len(returns) < 2 or not np.isfinite(excess).all():
        raise ValueError("Each reporting period needs at least two sessions with complete BIL returns")
    growth = np.r_[1.0, np.cumprod(1.0 + returns)]
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    scale = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(scale * vol),
        "sharpe": float(scale * returns.mean() / vol) if vol > 0.0 else None,
        "sharpe_excess": float(scale * excess.mean() / excess_vol) if excess_vol > 0.0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(SESSIONS_PER_YEAR * daily["turnover"].mean()),
        "n_trades": int(daily["trades"].sum()),
    }


def reporting_periods(daily, cash_returns, is_end):
    return {
        "is": metrics(daily.loc[:is_end], cash_returns),
        "oos": metrics(daily.loc[daily.index > is_end], cash_returns),
        "full": metrics(daily, cash_returns),
    }


def in_sample_trials(base_model, base_execution, robustness):
    specs = [("default", base_model, base_execution)]
    for parameter in ("lookback_sessions", "horizon_sessions"):
        for value in robustness[parameter]:
            specs.append((f"{parameter}={value}", {**base_model, parameter: value}, base_execution))
    multiplier = robustness["friction_multiplier"]
    higher_cost = {key: base_execution[key] * multiplier
                   for key in ("commission_bps", "slippage_bps", "borrow_bps_annual")}
    specs.append((f"friction_multiplier={multiplier:g}", base_model, {**base_execution, **higher_cost}))
    if len({(tuple(model.items()), tuple(execution.items())) for _, model, execution in specs}) != len(specs):
        raise ValueError("Robustness specifications must be distinct")
    return specs


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    pair = universe["pair"]
    benchmark_symbol = universe["benchmark"]
    proxy_symbol = universe["excess_return_proxy"]
    symbols = [*pair, benchmark_symbol, proxy_symbol]
    if len(pair) != 2 or len(set(symbols)) != 4:
        raise ValueError("Specify two distinct pair legs, a benchmark and a separate excess-return proxy")

    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    warmup_start, start, is_end, end = (dates[key] for key in ("warmup_start", "start", "is_end", "end"))
    execution = config["execution"]
    model = config["signal"]
    trials = in_sample_trials(model, execution, config["robustness"])
    if (not warmup_start < start < is_end < end or execution["lag_sessions"] < 1
            or execution["initial_capital"] <= 0 or execution["cash_rate_annual"] < 0
            or any(execution[key] < 0 for key in ("commission_bps", "slippage_bps", "borrow_bps_annual"))
            or not 0 < model["utility_power"] < 1 or not 0 < model["gross_weight_cap"] < 1
            or model["terminal_utility_weight"] <= 0 or model["feedback_fraction"] <= 0
            or not 0 < model["cointegration_p_max"] < 1 or model["cointegration_adf_lags"] < 0
            or config["robustness"]["friction_multiplier"] <= 1.0
            or any(spec["lookback_sessions"] <= 2 * spec["cointegration_adf_lags"] + 10
                   or spec["horizon_sessions"] < 1 for _, spec, _ in trials)):
        raise ValueError("Invalid dates, costs, execution lag or model parameters")

    market = load_market(symbols, warmup_start, end)
    all_dates = market["Close"].index
    minimum_warmup = max(spec["lookback_sessions"] for _, spec, _ in trials) + execution["lag_sessions"]
    if start not in all_dates or is_end not in all_dates or all_dates.get_loc(start) < minimum_warmup:
        raise ValueError("Use actual session dates and enough observations before the evaluation start")

    market_returns = forward_total_returns(market["Close"], market["Distributions"])
    if (market_returns.iloc[1:] <= -1.0).any().any():
        raise ValueError("Nonpositive daily total-return factor")
    pair_indices = (1.0 + market_returns[pair].fillna(0.0)).cumprod()
    cash_returns = market_returns[proxy_symbol]
    is_market = {field: values.loc[:is_end] for field, values in market.items()}
    baseline_is = None
    print("ROBUSTNESS: in-sample only, fixed default retained")
    for label, trial_model, trial_execution in trials:
        targets = opening_targets(pair_indices.loc[:is_end], trial_model, trial_execution)
        daily = simulate(is_market, targets, start, trial_execution)
        if label == "default":
            baseline_is = daily
        print(f"ROBUSTNESS IS {label}: {json.dumps(metrics(daily, cash_returns.loc[:is_end]), allow_nan=False)}")

    targets = opening_targets(pair_indices, model, execution)
    official_daily = simulate(market, targets, start, execution)
    pd.testing.assert_frame_equal(baseline_is, official_daily.loc[:is_end], check_exact=True)
    official = reporting_periods(official_daily, cash_returns, is_end)
    if min(official["is"]["n_trades"], official["oos"]["n_trades"]) <= 0:
        raise RuntimeError("The default specification needs actual fills in both reporting periods")

    benchmark_targets = pd.DataFrame(np.nan, index=all_dates, columns=[benchmark_symbol])
    benchmark_targets.loc[start, benchmark_symbol] = 1.0
    benchmark_daily = simulate(market, benchmark_targets, start, execution)
    benchmark = reporting_periods(benchmark_daily, cash_returns, is_end)
    for period, values in official.items():
        print(f"{period.upper()}: {json.dumps(values, allow_nan=False)}")
        print(f"BENCHMARK {benchmark_symbol} {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
    print("LOOKAHEAD_CHECK: trailing pre-signal fit; close-to-later-open lag; daily public bars only; "
          "forward distribution adjustments; exact in-sample prefix match")
    print(f"MIN_CASH: {official_daily['cash'].min():.6f}")
    print(f"PARAM_COMBOS_TRIED: {len(trials)}")
    print("已回测 · 待审查")
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        **official,
        "benchmark": {"name": f"{benchmark_symbol} buy and hold", **benchmark["full"]},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()

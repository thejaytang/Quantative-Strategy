"""668_dynamic_mean_variance_asset_allocation_i
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/668_dynamic_mean_variance_asset_allocation_i.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 19:59 Europe/Oslo, exit 0, output sha256 88d020d4311fb9ea9c3b289e4c83977a29286c1913defaa28fae8fa08805f26b
Data source: Yahoo Finance（yfinance）日频数据。风险 ETF 为 SPY、IEF、TLT、GLD，现金配置 ETF 为 BIL，使用开盘价、收盘价和现金分派。利率状态取 ^FVX 美国五年期国债日收盘收益率，以百分点计。该次运行读取了本地行情快照。
Data fetched: 2026-11-27T18:59:28+00:00
Sample period: 行情自 2007-08-01 开始，用于回测前预热。日频回测从 2014-01-03 至 2025-12-31，包含首尾交易日。   data_range (from output): ['2014-01-03', '2025-12-31']
IS/OOS split: 样本内 2014-01-03 至 2019-12-31。样本外首个交易日为 2020-01-02，至 2025-12-31。默认规格连续模拟全段，样本外继承样本内末的持仓与现金，模型继续按滚动窗口更新。四个替代规格仅在样本内运行。
Costs and slippage: 单边成本 2 bps，滑点按固定 3 bps 建模。成交使用开盘价，合计 5 bps 乘以买入与卖出名义金额绝对值之和，从开盘资金预算扣除，并纳入费后目标仓位求解。样本内的双倍摩擦情景为成本 4 bps、滑点 6 bps。
Signal lag: 每月首个 ETF 交易日 t 收盘后，使用截至 t 收盘的 ETF 总回报生成目标权重，在下一交易日开盘模拟成交。^FVX 缺报最多用过去报价向前填充 3 个交易日，随后整体滞后 1 个交易日，故信号最晚使用 t−1 的 ^FVX 收盘观测。
Price adjustment: yfinance auto_adjust=False, split-adjusted prices plus explicit ex-date distributions
Survivorship bias: 采用事后选定的固定 ETF 池。README 记载各 ETF 在回测开始前已上市，且所选基金仍存续，样本未纳入清算或退市产品，存在事后选择和存续偏差。
Parameter combinations tried: 5
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09511865202815462, ann_vol=0.07097195742704647, sharpe=1.3159850766613093, sharpe_excess=1.2133738014787927, max_drawdown=-0.0861233951538708, total_return=0.7230476173991385, turnover=3.38028079909511, n_trades=183
    oos: cagr=0.10845417003923297, ann_vol=0.13633332181075886, sharpe=0.8238047905969967, sharpe_excess=0.6321906315354915, max_drawdown=-0.2582442192959522, total_return=0.8518109333606783, turnover=3.546301174119297, n_trades=180
    full: cagr=0.10176402492686498, ann_vol=0.10865582811284621, sharpe=0.9465843429204789, sharpe_excess=0.7930747009721092, max_drawdown=-0.25824421929595, total_return=2.190758416600814, turnover=3.4632634724582108, n_trades=363
    benchmark: cagr=0.0911826894931953, ann_vol=0.10100450159624551, sharpe=0.9146346537592831, sharpe_excess=0.7491207715066713, max_drawdown=-0.21175491297472937, total_return=1.8426075504950972, turnover=0.303792874678577, n_trades=288
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 668_dynamic_mean_variance_asset_allocation_i\nuniverse: [SPY, IEF, TLT, GLD]\ncash_etf: BIL\nrate_ticker: '^FVX'\nbenchmark: {SPY: 0.60, IEF: 0.40}\n\ndates:\n  data_start: 2007-08-01\n  start: 2014-01-03\n  is_end: 2019-12-31\n  end: 2025-12-31\n\nexecution:\n  signal_to_open_sessions: 1\n  rate_publication_lag_sessions: 1\n  max_rate_forward_fill_sessions: 3\n  cost_bps: 2.0\n  slippage_bps: 3.0\n\nmodel:\n  lookback_months: 60\n  horizon_months: 3\n  risk_aversion: 4.0\n  ckls_power: 0.5\n  rate_floor_pct: 0.01\n  risk_weight_cap: 0.65\n  mean_prior_months: 24\n  slope_prior_months: 24\n  covariance_shrinkage: 0.15\n  state_grid_size: 31\n  quadrature_nodes: 9\n\nrobustness:\n  lookback_months: [54, 66]\n  myopic_horizon_months: 1\n  friction_multiplier: 2.0\n"
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
"""C244-inspired rolling ETF allocation with an interest-rate hedge.

Paper: Lei, Pun and Tang (2024), https://arxiv.org/abs/2412.18498
Research references (ideas only):
- QuantConnect LEAN rolling estimation and optimization (Apache-2.0):
  https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Portfolio/MeanVarianceOptimizationPortfolioConstructionModel.py
- Artur Sepp's finite-horizon feedback allocation (MIT):
  https://github.com/ArturSepp/GoalBasedAllocation/blob/main/src/goal_based_allocation/riccati_solver.py
  https://pypi.org/project/goal-based-allocation/
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


@dataclass
class Market:
    opens: pd.DataFrame
    closes: pd.DataFrame
    distributions: pd.DataFrame
    yield_close: pd.Series


def prepare_market(raw, tickers, rate_ticker, start, end):
    raw = raw.sort_index()
    if not raw.index.is_unique:
        raise ValueError("Duplicate price dates")
    etf_dates = raw["Close"][tickers].notna().any(axis=1) | raw["Open"][tickers].notna().any(axis=1)
    opens = raw.loc[etf_dates, "Open"][tickers]
    closes = raw.loc[etf_dates, "Close"][tickers]
    distributions = raw.loc[etf_dates, "Distribution"][tickers]
    prices = np.column_stack([opens, closes])
    if opens.empty or opens.index[0] != start or opens.index[-1] != end:
        raise ValueError(f"Incomplete ETF price span for {start.date()} through {end.date()}")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Incomplete or invalid ETF opens or closes")
    if not np.isfinite(distributions).all().all() or (distributions < 0).any().any():
        raise ValueError("Invalid ETF distributions")
    yield_close = raw.loc[etf_dates, ("Close", rate_ticker)]
    yield_close = yield_close.where(np.isfinite(yield_close) & (yield_close > 0))
    if yield_close.isna().all():
        raise ValueError(f"Missing yield history for {rate_ticker}")
    return Market(opens, closes, distributions, yield_close)


def load_market(config):
    tickers = [*config["universe"], config["cash_etf"]]
    rate_ticker = config["rate_ticker"]
    start = pd.Timestamp(config["dates"]["data_start"])
    end = pd.Timestamp(config["dates"]["end"])
    download_end = end + pd.Timedelta(days=1)
    key = sha256(repr(("split_prices_and_distributions_v1", tickers, rate_ticker, start, end)).encode())
    CACHE.mkdir(exist_ok=True)
    snapshot = CACHE / f"yahoo_{key.hexdigest()[:16]}.csv"

    if snapshot.exists():
        with snapshot.open(encoding="utf-8") as stream:
            fetched_at = datetime.fromisoformat(stream.readline().strip()).isoformat()
            raw = pd.read_csv(stream, header=[0, 1], index_col=0, parse_dates=[0], float_precision="round_trip")
        return prepare_market(raw, tickers, rate_ticker, start, end), fetched_at

    # Keep library housekeeping inside this folder and retain only public price data.
    with TemporaryDirectory(prefix="yf_", dir=CACHE) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        for attempt in range(2):
            downloaded = yf.download(
                [*tickers, rate_ticker],
                start=start.date().isoformat(),
                end=download_end.date().isoformat(),
                auto_adjust=False,
                actions=True,
                ignore_tz=True,
                threads=True,
                progress=False,
                timeout=12,
            )
            try:
                dividends = downloaded["Dividends"].reindex(columns=tickers).fillna(0.0)
                gains = pd.DataFrame(0.0, index=downloaded.index, columns=tickers)
                if "Capital Gains" in downloaded:
                    gains = downloaded["Capital Gains"].reindex(columns=tickers).fillna(0.0)
                raw = pd.concat(
                    {"Open": downloaded["Open"][tickers],
                     "Close": downloaded["Close"][[*tickers, rate_ticker]],
                     "Distribution": dividends + gains},
                    axis=1,
                )
                market = prepare_market(raw, tickers, rate_ticker, start, end)
                break
            except (KeyError, ValueError) as exc:
                if attempt == 1:
                    raise RuntimeError("Yahoo Finance did not return complete required data") from exc

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", prefix="prices_", dir=CACHE, delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            stream.write(f"{fetched_at}\n")
            raw.to_csv(stream)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    try:
        temporary_path.replace(snapshot)
    finally:
        temporary_path.unlink(missing_ok=True)
    return market, fetched_at


def monthly_inputs(market, execution, stop):
    closes = market.closes.loc[:stop]
    sessions = closes.index
    signal_dates = sessions[~sessions.to_period("M").duplicated()]
    distributions = market.distributions.loc[:stop]
    daily_gross = (closes + distributions).div(closes.shift()).fillna(1.0)
    monthly_gross = daily_gross.cumprod().loc[signal_dates].pct_change(fill_method=None) + 1.0
    monthly_excess = monthly_gross.iloc[:, :-1].div(monthly_gross.iloc[:, -1], axis=0) - 1.0
    lag = execution["rate_publication_lag_sessions"]
    max_carry = execution["max_rate_forward_fill_sessions"]
    observed = market.yield_close.loc[:stop]
    if max_carry:
        observed = observed.ffill(limit=max_carry)
    observed = observed.shift(lag).loc[signal_dates]
    return sessions, signal_dates, monthly_excess, observed


def long_weights(mean, covariance, gamma, cap):
    """Solve the concave objective with a nonnegative BIL residual."""
    curvature = gamma * covariance
    if np.all(mean <= 0):
        return np.zeros_like(mean)
    free = np.linalg.solve(curvature, mean)
    if np.all((free >= 0) & (free <= cap)) and free.sum() <= 1:
        return free

    # An objective scale near one keeps the solver tolerance meaningful.
    scale = max(np.max(np.diag(curvature)), np.max(np.abs(mean)))
    linear, quadratic = mean / scale, curvature / scale
    solution = minimize(
        lambda weights: 0.5 * weights @ quadratic @ weights - linear @ weights,
        np.full(len(mean), min(cap, 1.0 / len(mean))),
        jac=lambda weights: quadratic @ weights - linear,
        bounds=[(0.0, cap)] * len(mean),
        constraints={"type": "ineq", "fun": lambda weights: 1.0 - weights.sum(),
                     "jac": lambda weights: -np.ones_like(weights)},
        method="SLSQP",
        options={"ftol": 1e-11, "maxiter": 100},
    )
    if not solution.success or not np.isfinite(solution.x).all() or solution.x.sum() > 1.0 + 1e-8:
        raise RuntimeError(f"Constrained optimization failed: {solution.message}")
    weights = np.clip(solution.x, 0.0, cap)
    return weights / max(1.0, weights.sum())


def feedback_weights(rates, excess, model, shocks, probabilities):
    """Fit monthly transitions and solve a finite game on additive excess returns."""
    previous_rate = rates[:-1]
    rate_change = np.diff(rates)
    current_rate = rates[-1]
    n = len(rate_change)
    floor = model["rate_floor_pct"]
    power = model["ckls_power"]

    # One-month Euler transition with square-root diffusion at the default power.
    diffusion_scale = np.maximum(previous_rate, floor) ** power
    precision = diffusion_scale ** -2
    rate_center = np.average(previous_rate, weights=precision)
    centered_rate = previous_rate - rate_center
    change_center = np.average(rate_change, weights=precision)
    slope = (precision * centered_rate) @ (rate_change - change_center)
    speed = np.clip(-slope / ((precision * centered_rate) @ centered_rate), 0.0, 1.0)
    innovations = (rate_change - change_center + speed * centered_rate) / diffusion_scale
    rate_sigma = innovations.std(ddof=1)
    rate_shocks = (innovations - innovations.mean()) / rate_sigma

    # Shrink the unconditional premium toward zero and damp the state slope.
    state_mean, state_scale = previous_rate.mean(), previous_rate.std()
    standardized_rate = (previous_rate - state_mean) / state_scale
    sample_mean = excess.mean(axis=0)
    premium = sample_mean * n / (n + model["mean_prior_months"])
    state_loading = standardized_rate @ (excess - sample_mean) / (n + model["slope_prior_months"])
    residuals = excess - sample_mean - np.outer(standardized_rate, state_loading)
    shock_loading = rate_shocks @ residuals / (n - 1)
    other_shocks = residuals - np.outer(rate_shocks, shock_loading)
    residual_covariance = np.atleast_2d(np.cov(other_shocks, rowvar=False, ddof=1))
    shrink = model["covariance_shrinkage"]
    covariance = (
        (1.0 - shrink) * residual_covariance
        + shrink * np.diag(residual_covariance.diagonal())
        + np.outer(shock_loading, shock_loading)
        + np.eye(excess.shape[1]) * 1e-10
    )

    gamma = model["risk_aversion"]
    cap = model["risk_weight_cap"]
    horizon = model["horizon_months"]
    padding = np.max(np.abs(shocks)) * rate_sigma * max(rates.max(), floor) ** power * np.sqrt(horizon)
    grid = np.linspace(max(floor, rates.min() - padding), rates.max() + padding, model["state_grid_size"])
    continuation = np.zeros_like(grid)

    for remaining in range(1, horizon + 1):
        states = grid if remaining < horizon else np.array([current_rate])
        bounded_states = states.clip(previous_rate.min(), previous_rate.max())
        means = premium + np.outer((bounded_states - state_mean) / state_scale, state_loading)
        drift = change_center - speed * (states - rate_center)
        next_states = states[:, None] + drift[:, None] + np.outer(
            rate_sigma * np.maximum(states, floor) ** power, shocks,
        )
        future = np.interp(np.maximum(next_states, floor), grid, continuation)
        # Future choices depend on the next yield, so only the shared rate shock
        # connects this month's ETF innovations to next month's continuation.
        hedge_covariance = np.outer(future @ (probabilities * shocks), shock_loading)
        scores = means - gamma * hedge_covariance
        weights = np.array([long_weights(score, covariance, gamma, cap) for score in scores])
        continuation = np.sum(weights * means, axis=1) + future @ probabilities

    risky_weights = weights[0]
    return np.append(risky_weights, max(0.0, 1.0 - risky_weights.sum()))


def build_targets(market, config, model, start, stop):
    execution = config["execution"]
    sessions, signals, excess, rates = monthly_inputs(market, execution, stop)
    fill_sessions = sessions.get_indexer(signals) + execution["signal_to_open_sessions"]
    lookback = model["lookback_months"]
    nodes, weights = np.polynomial.hermite.hermgauss(model["quadrature_nodes"])
    shocks, probabilities = nodes * np.sqrt(2.0), weights / np.sqrt(np.pi)
    targets = {}
    for i, fill in enumerate(fill_sessions):
        if fill >= len(sessions) or sessions[fill] < start:
            continue
        if i <= lookback:
            raise ValueError(f"Insufficient pre-sample history for the {signals[i].date()} signal")
        sample_rates = rates.iloc[i - lookback:i + 1].to_numpy()
        sample_excess = excess.iloc[i - lookback + 1:i + 1].to_numpy()
        if rates.index[i] >= sessions[fill] or execution["rate_publication_lag_sessions"] < 1:
            raise ValueError("A signal or yield input violates the required lag")
        if not np.isfinite(sample_rates).all() or not np.isfinite(sample_excess).all():
            raise ValueError(f"Incomplete monthly inputs for the {signals[i].date()} signal")
        if np.ptp(sample_rates[:-1]) == 0 or np.ptp(np.diff(sample_rates)) == 0:
            raise ValueError(f"Insufficient yield variation for the {signals[i].date()} signal")
        targets[sessions[fill]] = feedback_weights(sample_rates, sample_excess, model, shocks, probabilities)
    if not targets:
        raise ValueError(f"No rebalance targets between {start.date()} and {stop.date()}")
    return targets


def rebalance(positions, cash, weights, friction):
    """Find post-cost target notionals whose trade and cost fit the opening budget."""
    opening_value = positions.sum() + cash
    lower, upper = 0.0, opening_value
    for _ in range(48):
        post_cost_value = 0.5 * (lower + upper)
        moved = np.abs(weights * post_cost_value - positions).sum()
        if post_cost_value + friction * moved <= opening_value:
            lower = post_cost_value
        else:
            upper = post_cost_value
    new_positions = weights * lower
    traded = np.abs(new_positions - positions)
    new_cash = opening_value - new_positions.sum() - friction * traded.sum()
    if new_cash < -1e-12 * opening_value or (new_positions < 0.0).any():
        raise RuntimeError("Rebalance breached the self-financing long-only constraint")
    n_trades = int(np.count_nonzero(traded > 1e-10 * opening_value))
    return new_positions, max(0.0, new_cash), traded.sum() / opening_value, n_trades


def simulate(market, targets, start, stop, execution):
    sessions = market.closes.index
    first = sessions.searchsorted(start)
    last = sessions.searchsorted(stop, side="right")
    opens = market.opens.iloc[first:last].to_numpy()
    closes = market.closes.iloc[first:last].to_numpy()
    distributions = market.distributions.iloc[first:last].to_numpy()
    previous_closes = market.closes.iloc[first - 1:last - 1].to_numpy()
    friction = (execution["cost_bps"] + execution["slippage_bps"]) / 10000.0
    if not 0 <= friction < 1:
        raise ValueError("Combined cost and slippage must be less than 10,000 bps and nonnegative")
    positions = np.zeros(opens.shape[1])
    cash, previous_value = 1.0, 1.0
    daily_results = []
    for i, date in enumerate(sessions[first:last]):
        # Ex-date distributions belong to positions carried from the prior close.
        distribution_cash = positions @ (distributions[i] / previous_closes[i])
        positions *= opens[i] / previous_closes[i]
        turnover, n_trades = 0.0, 0
        if date in targets:
            positions, cash, turnover, n_trades = rebalance(positions, cash, targets[date], friction)
        positions *= closes[i] / opens[i]
        cash += distribution_cash
        value = positions.sum() + cash
        if not np.isfinite(value) or value <= 0 or cash < 0:
            raise RuntimeError(f"Invalid close valuation on {date.date()}")
        bil_return = (closes[i, -1] + distributions[i, -1]) / previous_closes[i, -1] - 1.0
        daily_results.append((value / previous_value - 1.0, bil_return, turnover, n_trades))
        previous_value = value
    return pd.DataFrame(daily_results, index=sessions[first:last],
                        columns=["return", "bil_return", "turnover", "n_trades"])


def metrics(daily_results):
    if len(daily_results) < 2:
        raise ValueError("Metrics require at least two return observations")
    returns = daily_results["return"].to_numpy()
    excess = returns - daily_results["bil_return"].to_numpy()
    curve = np.r_[1.0, np.cumprod(1.0 + returns)]
    years = len(returns) / SESSIONS_PER_YEAR
    scale = np.sqrt(SESSIONS_PER_YEAR)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(curve[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(volatility * scale),
        "sharpe": float(returns.mean() / volatility * scale) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * scale) if excess_volatility > 0 else None,
        "max_drawdown": float(np.min(curve / np.maximum.accumulate(curve) - 1.0)),
        "total_return": float(curve[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def validate_config(config):
    dates = config["dates"]
    execution = config["execution"]
    benchmark = config["benchmark"]
    etfs = [*config["universe"], config["cash_etf"]]
    if not dates["data_start"] < dates["start"] < dates["is_end"] < dates["end"]:
        raise ValueError("Use chronological dates with both IS and OOS periods")
    if min(execution["signal_to_open_sessions"], execution["rate_publication_lag_sessions"]) < 1:
        raise ValueError("Use at least one session of signal and yield publication lag")
    if min(execution["cost_bps"], execution["slippage_bps"], execution["max_rate_forward_fill_sessions"]) < 0:
        raise ValueError("Friction and the rate carry limit must be nonnegative")
    if len(etfs) != len(set(etfs)) or not set(benchmark).issubset(etfs):
        raise ValueError("ETF tickers must be distinct and contain all benchmark tickers")
    if min(benchmark.values()) < 0 or not np.isclose(sum(benchmark.values()), 1.0, atol=1e-12, rtol=0):
        raise ValueError("Benchmark weights must be nonnegative and sum to one")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    market, fetched_at = load_market(config)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=False, split-adjusted prices plus explicit ex-date distributions")
    model, execution = config["model"], config["execution"]
    start, is_end, end = (pd.Timestamp(config["dates"][name]) for name in ("start", "is_end", "end"))
    oos_start = is_end + pd.Timedelta(days=1)

    is_targets = build_targets(market, config, model, start, is_end)
    is_metrics = metrics(simulate(market, is_targets, start, is_end, execution))
    print("ROBUSTNESS (IS only)")
    print(f"  Default: {json.dumps(is_metrics, allow_nan=False)}")
    combos_tried = 1
    robustness = config["robustness"]
    variants = [(f"Lookback {months} months", {**model, "lookback_months": months})
                for months in robustness["lookback_months"]]
    variants.append(("Myopic only", {**model, "horizon_months": robustness["myopic_horizon_months"]}))
    for label, variant in variants:
        targets = build_targets(market, config, variant, start, is_end)
        result = metrics(simulate(market, targets, start, is_end, execution))
        combos_tried += 1
        print(f"  {label}: {json.dumps(result, allow_nan=False)}")

    multiplier = robustness["friction_multiplier"]
    high_friction = {**execution, "cost_bps": execution["cost_bps"] * multiplier,
                     "slippage_bps": execution["slippage_bps"] * multiplier}
    result = metrics(simulate(market, is_targets, start, is_end, high_friction))
    combos_tried += 1
    print(f"  {multiplier:g}x costs and slippage: {json.dumps(result, allow_nan=False)}")
    print(f"PARAM_COMBOS_TRIED: {combos_tried}")

    oos_targets = build_targets(market, config, model, oos_start, end)
    targets = is_targets | oos_targets
    full_results = simulate(market, targets, start, end, execution)
    benchmark_weights = np.array([config["benchmark"].get(ticker, 0.0) for ticker in market.closes])
    benchmark_targets = dict.fromkeys(targets, benchmark_weights)
    benchmark = simulate(market, benchmark_targets, start, end, execution)

    full_is_metrics = metrics(full_results.loc[:is_end])
    if full_is_metrics != is_metrics:
        raise RuntimeError("The IS segment changed when OOS sessions were included")
    oos_results = full_results.loc[oos_start:]
    result = {
        "strategy": config["strategy"],
        "data_range": [full_results.index[0].date().isoformat(), full_results.index[-1].date().isoformat()],
        "is": full_is_metrics,
        "oos": metrics(oos_results),
        "full": metrics(full_results),
        "benchmark": metrics(benchmark),
    }
    if min(result[period]["n_trades"] for period in ("is", "oos")) == 0:
        raise RuntimeError("No actual changes in ETF positions in at least one period")

    print("LOOK_AHEAD_SELF_CHECK: Monthly close signals use completed return windows and delayed yield quotes. "
          "Open sizing uses the previous close and current open. Split normalization cancels in price ratios. "
          "Distributions enter on the ex-date close. The IS segment matches with or without OOS sessions.")
    print("STATUS: 已回测；待审查")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
    print(f"BENCHMARK ({config['benchmark']}), FULL: {json.dumps(result['benchmark'], allow_nan=False)}")
    print(f"BENCHMARK IS: {json.dumps(metrics(benchmark.loc[:is_end]), allow_nan=False)}")
    print(f"BENCHMARK OOS: {json.dumps(metrics(benchmark.loc[oos_start:]), allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

"""577_objective_and_subjective_entropy_measure
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/577_objective_and_subjective_entropy_measure.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:08 Europe/Oslo, exit 0, output sha256 b8e126d8c6d6d873aef348c1d1ede2adbd3debc02472460bdc1125a29555f444
Data source: 免费 Yahoo Finance 日线，经 yfinance 读取 ETF 开盘价、收盘价与现金分配（含可取得的资本利得分配）。投资池为 SPY、EFA、TLT、GLD、BIL，SPY 为基准，BIL 为无风险利率代理。
Data fetched: 2026-11-27T18:08:34+00:00
Sample period: 正式回测为 2010-01-05 开盘至 2026-11-20 收盘，请求的预热行情起始日为 2008-01-01。   data_range (from output): {'start': '2010-01-05', 'end': '2026-11-20'}
IS/OOS split: 样本内为 2010-01-05 至 2019-12-31（含），样本外为 2020-01-02 至 2026-11-20（含），延续样本内组合并按固定规则滚动估计历史情景。六个稳健性变体仅在样本内运行。
Costs and slippage: 默认每笔买入或卖出按成交金额扣除 5 bp 交易成本和 5 bp 滑点，合计每边 10 bp。卖出收入按合计费率扣减，买入预算包含同一费率。滑点以固定金额比例计入摩擦，模拟成交以开盘价估值。SPY 基准仅在样本首日开盘建仓时支付单边摩擦。样本内高摩擦变体将成本和滑点分别提高至 10 bp。
Signal lag: 首个信号在样本首日的前一个交易日收盘计算，此后在每月首个交易日收盘计算信号。有调仓目标时，默认在下一交易日开盘模拟成交。
Price adjustment: yfinance auto_adjust=False, Yahoo split-normalized OHLC, explicit ex-date distribution-adjusted total-return factors
Survivorship bias: 五只 ETF 构成固定投资池。README 记载它们在样本开始前均已上市，并指出事后选池与基金存续偏差仍存在。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.020075729726221736, ann_vol=0.14752276478350937, sharpe=-0.06352790402281965, sharpe_excess=-0.09183973347318408, max_drawdown=-0.3492861222747343, total_return=-0.18322978206409868, turnover=9.208943084512681, n_trades=96
    oos: cagr=0.07629678668371342, ann_vol=0.1994179320361, sharpe=0.4692486693713974, sharpe_excess=0.33179290373375275, max_drawdown=-0.3371726392350435, total_return=0.6575607068468905, turnover=4.494458655886624, n_trades=38
    full: cagr=0.018138355101265713, ann_vol=0.1706108250633046, sharpe=0.19115041093594007, sharpe_excess=0.11114636419513828, max_drawdown=-0.5238046306758826, total_return=0.35384621977332764, turnover=7.286294855084773, n_trades=134
    benchmark: cagr=0.13275772350884285, ann_vol=0.17086858042671293, sharpe=0.8153675763553129, sharpe_excess=0.7351102038016347, max_drawdown=-0.3371726392350436, total_return=7.172987847438208, turnover=0.059276725158524085, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 577_objective_and_subjective_entropy_measure\nuniverse: [SPY, EFA, TLT, GLD, BIL]\nbenchmark: SPY\nrisk_free_proxy: BIL\n\ndates:\n  history_start: '2008-01-01'\n  start: '2010-01-05'\n  is_end: '2019-12-31'\n  end: '2026-11-20'\n\nmodel:\n  lookback_sessions: 120\n  regime_ticker: SPY\n  volatility_sessions: 21\n  stress_volatility: 0.20\n  regime_mix: 0.60\n  horizon_sessions: 21\n  entropy_buffer_bps: 3.0\n  use_entropy: true\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\n\nrobustness:\n  lookback_sessions: [100, 140]\n  regime_mix: [0.45, 0.75]\n  high_cost_bps: 10.0\n  high_slippage_bps: 10.0\n"
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
"""C182 research backtest. Independent implementation with no borrowed source code.

Sharma (2026): https://arxiv.org/abs/2607.09505
Riskfolio-Lib (BSD-3-Clause): https://pypi.org/project/riskfolio-lib/
https://raw.githubusercontent.com/dcajasn/Riskfolio-Lib/master/riskfolio/src/Portfolio.py
Busseti, Ryu and Boyd (GPL-3.0): https://stanford.edu/~boyd/papers/kelly.html
https://github.com/cvxgrp/kelly_code
https://raw.githubusercontent.com/cvxgrp/kelly_code/master/finite_outcome.ipynb
QuantConnect research example:
https://www.quantconnect.com/research/18312/kelly-criterion-applications-in-trading-systems/p1
Quantopian tutorial:
https://notebook.community/d00d/quantNotebooks/Notebooks/quantopian_research_public/presentations/LECTURE_Stanford_Quantopian_Tutorial_and_Markowitz_Optimization
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import logsumexp
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = (
    "yfinance auto_adjust=False, Yahoo split-normalized OHLC, "
    "explicit ex-date distribution-adjusted total-return factors"
)


@dataclass(frozen=True)
class Model:
    lookback_sessions: int
    regime_ticker: str
    volatility_sessions: int
    stress_volatility: float
    regime_mix: float
    horizon_sessions: int
    entropy_buffer_bps: float
    use_entropy: bool


@dataclass(frozen=True)
class Execution:
    lag_sessions: int
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def load_market(tickers, history_start, end):
    directory = ROOT / "data_cache"
    directory.mkdir(exist_ok=True)
    key = sha256(repr(("yahoo_ohlc_actions_v1", tickers, history_start, end)).encode())
    cache = directory / f"yahoo_{key.hexdigest()[:16]}.csv"
    if cache.exists():
        with cache.open(encoding="utf-8") as stream:
            fetched_at = stream.readline().strip()
            market = pd.read_csv(stream, header=[0, 1], index_col=0, parse_dates=[0])
        return market, fetched_at

    exclusive_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    with TemporaryDirectory(dir=directory, prefix="yahoo_session_") as session_dir:
        yf.set_tz_cache_location(session_dir)
        raw = yf.download(
            list(tickers), start=history_start, end=exclusive_end, interval="1d",
            auto_adjust=False, actions=True, keepna=True, progress=False,
            threads=4, timeout=10, multi_level_index=True,
        )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no daily data")
    market = pd.concat({
        field: raw[field].reindex(columns=tickers)
        for field in ("Open", "Close", "Dividends")
    }, axis=1).sort_index()
    if "Capital Gains" in raw.columns.get_level_values(0):
        distributions = raw["Capital Gains"].reindex(columns=tickers).fillna(0)
        market["Dividends"] = market["Dividends"] + distributions
    market_factors(market, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
        temporary_path = Path(stream.name)
        try:
            stream.write(f"{fetched_at}\n")
            market.to_csv(stream)
            stream.flush()
            temporary_path.replace(cache)
        finally:
            temporary_path.unlink(missing_ok=True)
    return market, fetched_at


def market_factors(market, end):
    if not market.index.is_monotonic_increasing or not market.index.is_unique:
        raise ValueError("Daily data must have ordered, unique session dates")
    if market.empty or market.index[-1] < pd.Timestamp(end) - pd.Timedelta(days=4):
        raise ValueError("Daily data ends before the requested sample")
    opens, closes, distributions = (market[field] for field in ("Open", "Close", "Dividends"))
    if not np.isfinite(market.to_numpy()).all() or (market.to_numpy() < 0).any():
        raise ValueError("Incomplete or negative price or distribution observations")
    previous_basis = closes.shift(1) - distributions
    if (opens <= 0).any().any() or (closes <= 0).any().any():
        raise ValueError("Open and close prices must be positive")
    if (previous_basis.iloc[1:] <= 0).any().any():
        raise ValueError("Distributions exceed the preceding close on a session")
    # An ex-date factor is determined by the prior close and the distribution effective at open.
    return opens / previous_basis, closes / opens


def weighted_scenarios(returns, volatility, signal_pos, model):
    start = signal_pos + 1 - model.lookback_sessions
    window = returns[start:signal_pos + 1]
    previous_volatility = volatility[start - 1:signal_pos]
    if start < 1 or len(window) != model.lookback_sessions or not np.isfinite(previous_volatility).all():
        raise ValueError("Insufficient history for the first signal and regime classifications")
    current_stress = volatility[signal_pos] >= model.stress_volatility
    same_regime = (previous_volatility >= model.stress_volatility) == current_stress
    probabilities = np.full(len(window), 1 / len(window))
    if same_regime.any():
        probabilities *= 1 - model.regime_mix
        probabilities[same_regime] += model.regime_mix / same_regime.sum()
    return window, probabilities


def kelly_weights(returns, probabilities, initial):
    # Scaling only the objective and its gradient improves numerical stopping tolerances.
    def negative_growth(weights):
        daily = returns @ weights
        return -SESSIONS_PER_YEAR * (probabilities @ np.log1p(daily))

    def gradient(weights):
        return -SESSIONS_PER_YEAR * (probabilities / (1 + returns @ weights)) @ returns

    result = minimize(
        negative_growth, initial, jac=gradient, method="SLSQP",
        bounds=[(0, 1)] * len(initial),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1,
                     "jac": lambda weights: np.ones_like(weights)},
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not result.success or abs(result.x.sum() - 1) > 1e-8:
        raise RuntimeError(f"Kelly optimization failed: {result.message}")
    weights = np.where(result.x > 1e-9, result.x, 0)
    return weights / weights.sum()


def subjective_projection(probabilities, directional_returns):
    """Tilt Q so the incumbent is log-optimal along the candidate direction."""
    if directional_returns.min() >= 0:
        # No full-support probabilities can give the incumbent a nonpositive slope.
        return None
    direction = directional_returns / np.max(np.abs(directional_returns))
    log_p = np.log(probabilities)

    def log_probabilities(tilt):
        tilted = log_p - tilt * direction
        return tilted - logsumexp(tilted)

    def slope(tilt):
        return np.exp(log_probabilities(tilt)) @ direction

    upper = 1.0
    while slope(upper) > 0:
        upper *= 2
        if upper > 2**50:
            raise RuntimeError("Unable to bracket the subjective probability tilt")
    log_q = log_probabilities(brentq(slope, 0, upper, xtol=1e-12))
    log_ratio = log_q - log_p
    q = np.exp(log_q)
    forward_kl = max(0.0, -probabilities @ log_ratio)
    reverse_kl = max(0.0, q @ log_ratio)
    subjective_gain = -q @ np.log1p(directional_returns)
    return forward_kl, reverse_kl, subjective_gain


def target_fraction(previous, weights, friction):
    """Solve post-fee value + fee on gross two-sided turnover = opening value."""
    def imbalance(fraction):
        return fraction + friction * np.abs(fraction * weights - previous).sum() - 1

    return brentq(imbalance, 0, 1, xtol=1e-14)


def fill_at_open(positions, cash, weights, friction):
    before = positions.sum() + cash
    previous = positions / before
    target = target_fraction(previous, weights, friction) * weights
    sales = np.maximum(previous - target, 0)
    purchases = np.maximum(target - previous, 0)
    available = cash / before + sales.sum() * (1 - friction)
    purchase_cost = purchases.sum() * (1 + friction)
    if purchase_cost > available:
        purchases *= available / purchase_cost
    remaining = max(0.0, available - purchases.sum() * (1 + friction))
    traded = sales + purchases
    return (previous - sales + purchases) * before, remaining * before, traded.sum(), int((traded > 1e-9).sum())


def select_target(returns, probabilities, positions, model, execution):
    invested = positions.sum()
    missing_diagnostics = (np.nan, np.nan, np.nan, np.nan)
    if invested == 0:
        equal_weights = np.full(len(positions), 1 / len(positions))
        return kelly_weights(returns, probabilities, equal_weights), missing_diagnostics

    incumbent = positions / invested
    candidate = kelly_weights(returns, probabilities, incumbent)
    direction = (returns @ (candidate - incumbent)) / (1 + returns @ incumbent)
    growth_gain = probabilities @ np.log1p(direction)
    if growth_gain <= 1e-12:
        return None, missing_diagnostics

    projection = subjective_projection(probabilities, direction) if model.use_entropy else None
    forward_kl, reverse_kl, subjective_gain = projection if projection is not None else (np.nan,) * 3
    evidence = min(growth_gain, forward_kl, reverse_kl) if projection is not None else growth_gain
    transition_loss = -np.log(target_fraction(incumbent, candidate, execution.friction))
    hurdle = transition_loss + model.entropy_buffer_bps / 10_000
    target = candidate if model.horizon_sessions * evidence > hurdle else None
    return target, (forward_kl, reverse_kl, growth_gain, subjective_gain)


def backtest(overnight, intraday, start, model, execution):
    start_pos = overnight.index.searchsorted(start)
    first_signal = start_pos - execution.lag_sessions
    if first_signal < 1 or start_pos >= len(overnight):
        raise ValueError("The first signal and fill require sessions within the input data")
    daily_returns = overnight * intraday - 1
    volatility = daily_returns[model.regime_ticker].rolling(model.volatility_sessions).std()
    volatility = volatility.to_numpy() * np.sqrt(SESSIONS_PER_YEAR)
    returns = daily_returns.to_numpy()
    at_open = overnight.to_numpy()
    during_session = intraday.to_numpy()
    months = overnight.index.to_period("M")
    positions = np.zeros(overnight.shape[1])
    cash = previous_total = 1.0
    pending = {}
    records = []
    columns = [
        "return", "turnover", "n_trades", "signal_pos", "forward_kl", "reverse_kl",
        "objective_gain", "subjective_gain",
    ]

    for pos in range(first_signal, len(overnight)):
        positions = positions * at_open[pos]
        turnover, trades, signal_pos = 0.0, 0, -1
        if pos in pending:
            signal_pos, target = pending.pop(pos)
            positions, cash, turnover, trades = fill_at_open(
                positions, cash, target, execution.friction,
            )
            if pos - signal_pos != execution.lag_sessions or cash < 0:
                raise RuntimeError("Fill violated session lag or available cash")

        positions = positions * during_session[pos]
        total = positions.sum() + cash
        diagnostics = (np.nan,) * 4
        if pos == first_signal or (pos >= start_pos and months[pos] != months[pos - 1]):
            scenarios, probabilities = weighted_scenarios(returns, volatility, pos, model)
            target, diagnostics = select_target(
                scenarios, probabilities, positions, model, execution,
            )
            if target is not None:
                pending[pos + execution.lag_sessions] = (pos, target)
        if pos >= start_pos:
            records.append((total / previous_total - 1, turnover, trades, signal_pos, *diagnostics))
        previous_total = total

    return pd.DataFrame(records, index=overnight.index[start_pos:], columns=columns)


def performance(daily_results, risk_free):
    daily = daily_results["return"].to_numpy()
    excess = daily - risk_free.loc[daily_results.index].to_numpy()
    path = np.r_[1.0, np.cumprod(1 + daily)]
    vol = daily.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(path[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(daily.mean() / vol * annualizer) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float((path / np.maximum.accumulate(path) - 1).min()),
        "total_return": float(path[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def benchmark_path(overnight, intraday, sample, execution):
    growth = (overnight * intraday).loc[sample].copy()
    entry_fraction = 1 / (1 + execution.friction)
    growth.iloc[0] = entry_fraction * intraday.loc[sample[0]]
    results = pd.DataFrame({"return": growth - 1, "turnover": 0.0, "n_trades": 0})
    results.iloc[0, 1:] = [entry_fraction, 1]
    return results


def robustness_variants(config, model, execution):
    for value in config["lookback_sessions"]:
        yield f"lookback_sessions={value}", replace(model, lookback_sessions=value), execution
    for value in config["regime_mix"]:
        yield f"regime_mix={value}", replace(model, regime_mix=value), execution
    yield "growth_hurdle_only", replace(model, use_entropy=False), execution
    high_cost = replace(execution, cost_bps=config["high_cost_bps"], slippage_bps=config["high_slippage_bps"])
    yield "higher_cost_and_slippage", model, high_cost


def validate_settings(dates, model, execution):
    if not dates["history_start"] < dates["start"] <= dates["is_end"] < dates["end"]:
        raise ValueError("Dates must give warm-up, in-sample, and out-of-sample periods")
    if execution.lag_sessions < 1 or min(execution.cost_bps, execution.slippage_bps) < 0:
        raise ValueError("Require positive session lag and nonnegative friction terms")
    if not 0 <= execution.friction < 1:
        raise ValueError("Combined trading friction must be less than 100%")
    if not 0 <= model.regime_mix < 1 or min(model.lookback_sessions, model.volatility_sessions) < 2:
        raise ValueError("Require positive pooled probability and valid rolling windows")
    if model.horizon_sessions < 1 or min(model.stress_volatility, model.entropy_buffer_bps) < 0:
        raise ValueError("Require positive horizon and nonnegative volatility and entropy thresholds")


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = config["dates"]
    universe = config["universe"]
    benchmark = config["benchmark"]
    proxy = config["risk_free_proxy"]
    model = Model(**config["model"])
    execution = Execution(**config["execution"])
    validate_settings(dates, model, execution)
    if len(set(universe)) != len(universe) or model.regime_ticker not in universe:
        raise ValueError("Require unique assets and a regime ticker in the universe")
    tickers = list(dict.fromkeys([*universe, benchmark, proxy]))
    market, fetched_at = load_market(tickers, dates["history_start"], dates["end"])
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    overnight, intraday = market_factors(market, dates["end"])
    asset_open = overnight.loc[:, universe]
    asset_intraday = intraday.loc[:, universe]
    is_end = dates["is_end"]
    start = dates["start"]
    first_session = overnight.index[overnight.index.searchsorted(start)]
    risk_free = overnight[proxy] * intraday[proxy] - 1
    risk_free.loc[first_session] = intraday.loc[first_session, proxy] - 1

    is_open = asset_open.loc[:is_end]
    is_intraday = asset_intraday.loc[:is_end]
    is_default = backtest(is_open, is_intraday, start, model, execution)
    if len(is_default) < 2:
        raise RuntimeError("Require at least two in-sample sessions")
    trials = 1
    print("ROBUSTNESS (IS only)")
    print(f"default IS: {json.dumps(performance(is_default, risk_free.loc[:is_end]))}")
    for label, variant_model, variant_execution in robustness_variants(config["robustness"], model, execution):
        validate_settings(dates, variant_model, variant_execution)
        variant = backtest(is_open, is_intraday, start, variant_model, variant_execution)
        metrics = performance(variant, risk_free.loc[:is_end])
        print(f"{label} IS: {json.dumps(metrics)}")
        trials += 1
    print(f"PARAM_COMBOS_TRIED: {trials}")

    full = backtest(asset_open, asset_intraday, start, model, execution)
    pd.testing.assert_frame_equal(is_default, full.loc[:is_end], check_exact=True)
    print("LOOKAHEAD_CHECK: default IS prefix reproduced exactly")
    oos = full.loc[full.index > pd.Timestamp(is_end)]
    if len(oos) < 2 or full["n_trades"].sum() <= 0:
        raise RuntimeError("Require at least two out-of-sample sessions and actual fills")
    benchmark_results = benchmark_path(overnight[benchmark], intraday[benchmark], full.index, execution)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": full.index[0].date().isoformat(), "end": full.index[-1].date().isoformat()},
        "is": performance(is_default, risk_free),
        "oos": performance(oos, risk_free),
        "full": performance(full, risk_free),
        "benchmark": performance(benchmark_results, risk_free),
    }
    diagnostics = full[["forward_kl", "reverse_kl", "objective_gain", "subjective_gain"]].dropna()
    print(f"KL_EVALUATIONS: {len(diagnostics)}")
    if not diagnostics.empty:
        print(f"MEAN_DAILY_ENTROPY_AND_GROWTH_GAPS: {json.dumps(diagnostics.mean().to_dict())}")
    for label in ("is", "oos", "full", "benchmark"):
        print(f"{label.upper()}: {json.dumps(results[label], allow_nan=False)}")
    print("STATUS: 已回测，待复核")
    print(f"RESULTS_JSON: {json.dumps(results, allow_nan=False)}")


if __name__ == "__main__":
    main()

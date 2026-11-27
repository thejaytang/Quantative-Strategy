"""431_mitigating_return_drag_and_regime_whipsa
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/431_mitigating_return_drag_and_regime_whipsa.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c3e441c5e45062dfc079c2de97e601275410b0bf on 2026-11-27 08:00 Europe/Oslo, exit 0, output sha256 f8911c3d07faa4e6116533a8e1e41c4412171dcb4606f3036efc3a4a59090742
Data source: 免费 Yahoo Finance 日线，由 yfinance 获取。行业 ETF 为 XLK、XLF、XLV、XLE、XLI、XLY、XLP、XLU、XLB、XLRE、XLC，基准为 SPY，参考序列为 ^VIX 和 ^IRX。日期按 SPY 有收盘价的交易日对齐，优先读取 data_cache/ 下对应的价格 CSV。（config.yaml:2-5，strategy.py:32-64，README.md:20）
Data fetched: 2026-11-27T07:00:06+00:00
Sample period: 配置请求区间为 2018-07-01 至 2026-10-30（含末日）。日志记录的实际输入为 2018-07-02 至 2026-10-30，含 24 个月估计窗口所需的预热数据。评估从首次模拟成交日 2020-08-03 开始，至 2026-10-30 结束。（config.yaml:7-9、20，strategy.py:33-44、247-256、327-328，logs/run_20261127_072106.log:4-5）   data_range (from output): {'start': '2020-08-03', 'end': '2026-10-30'}
IS/OOS split: IS 为 2020-08-03 至 2025-12-31（含），OOS 为 2026-01-02 至 2026-10-30（含）。程序以同一配置连续生成目标、更新模型状态并模拟持仓，再将同一条日收益序列按 2025-12-31 切分。HMM 在每个信号日以截至当月的扩展窗口重新拟合。（config.yaml:10，README.md:16、21，strategy.py:102-145、247-271、321-342）
Costs and slippage: 每笔模拟买入和卖出按绝对成交金额收取 5 bps 费用和 5 bps 固定滑点，共 10 bps，在成交收盘从净值扣除。目标金额基于扣减后的净值求解，原持仓先按成交日收盘价计价。SPY 基准的首次买入采用相同费率。（config.yaml:16-17，strategy.py:274-295、327-334，README.md:23）
Signal lag: 在每月最后一个 SPY 交易日收盘后生成信号，下一 SPY 交易日收盘模拟成交。成交日的收盘至收盘收益先计入原持仓，成交后使用新权重。^VIX 和 ^IRX 均按 SPY 交易日向后移一行，下行收益门槛取上一月末可用的滞后 ^IRX。（config.yaml:13-15，strategy.py:236-268、274-295，README.md:22）
Price adjustment: yfinance auto_adjust total-return ETF closes; ^VIX and ^IRX levels
Survivorship bias: 按论文固定选取当前存续的 11 只行业 ETF，待所有基金具备完整月度数据后开始估计。README 明示存续偏差和品种选择偏差，并指出基金成分和行业分类会随时间变化。（config.yaml:2、7-9、20，strategy.py:20、25-29、236-258，README.md:27）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.12594481422059411, ann_vol=0.19371932929656002, sharpe=0.7111685714190015, sharpe_excess=0.5574085993272799, max_drawdown=-0.21985998430820297, total_return=0.9016488450317035, turnover=4.171272534132459, n_trades=175
    oos: cagr=0.14557069908244902, ann_vol=0.17937170268918565, sharpe=0.847197807186867, sharpe_excess=0.6414507800009153, max_drawdown=-0.1316231149253546, total_return=0.11934180367948022, turnover=5.239264817143786, n_trades=32
    full: cagr=0.12853122062775846, ann_vol=0.19181688560462454, sharpe=0.728074135788041, sharpe_excess=0.5678418511875836, max_drawdown=-0.21985998430820297, total_return=1.128595048162786, turnover=4.313078696162448, n_trades=207
    benchmark: cagr=0.1579272272719514, ann_vol=0.16609297880977608, sharpe=0.968453708428858, sharpe_excess=0.7833970304131541, max_drawdown=-0.24496393992794496, total_return=1.499575464663005, turnover=0.1598970705018032, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 431_mitigating_return_drag_and_regime_whipsa\nuniverse: [XLK, XLF, XLV, XLE, XLI, XLY, XLP, XLU, XLB, XLRE, XLC]\nbenchmark: SPY\nvolatility_index: "^VIX"\nrisk_free: "^IRX"\n\ndates:\n  start: "2018-07-01"  # Includes the first month-end price anchor.\n  end: "2026-10-30"    # Inclusive.\n  is_end: "2025-12-31" # OOS begins on the next equity session.\n\nexecution:\n  lag_sessions: 1\n  vix_publication_lag_sessions: 1\n  irx_publication_lag_sessions: 1\n  costs_bps: 5.0\n  slippage_bps: 5.0\n\nmodel:\n  lookback_months: 24\n  clusters: 3\n  sticky_kappa: 0.30\n  confidence: 0.65\n  bull_return_weight: 1.2\n  within_cluster_turnover_penalty: 0.015\n  downside_eigenvalue_floor: 0.000001\n  vix_zscore_min_months: 12\n  hmm_initial_self_transition: 0.90\n  hmm_max_iterations: 60\n  hmm_tolerance: 0.01\n  hmm_relative_variance_floor: 0.000001\n'
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
"""Monthly downside NCO with a sticky Gaussian regime filter."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.optimize import brentq, minimize
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SECTORS = frozenset("XLK XLF XLV XLE XLI XLY XLP XLU XLB XLRE XLC".split())
SESSIONS_PER_YEAR = 252
PARAM_COMBOS_TRIED = 1


def load_prices(config):
    sectors = config["universe"]
    market, vix, irx = (config[key] for key in ("benchmark", "volatility_index", "risk_free"))
    if not set(sectors) <= SECTORS or len(set(sectors)) != len(sectors):
        raise ValueError("The universe must contain distinct Select Sector SPDR ETFs.")
    if (market, vix, irx) != ("SPY", "^VIX", "^IRX"):
        raise ValueError("Expected SPY, ^VIX, and ^IRX as the market and reference series.")
    tickers = sectors + [market, vix, irx]
    start = pd.Timestamp(config["dates"]["start"])
    end = pd.Timestamp(config["dates"]["end"])
    request = (str(start.date()), str((end + pd.Timedelta(days=1)).date()), tickers, True)
    key = hashlib.sha256(json.dumps(request).encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"yahoo_adjusted_close_{key}.csv"

    def validate(raw):
        prices = raw.loc[raw[market].notna(), tickers].sort_index()
        if (prices.empty or prices.index.has_duplicates or prices.index[0] < start
                or prices.index[-1] != end):
            raise ValueError("Incomplete price range. Set dates.end to a completed SPY session.")
        values = prices.to_numpy()
        if not np.isfinite(values).all() or (prices.drop(columns=irx) <= 0).any().any():
            raise ValueError("Missing or invalid data on the SPY trading calendar.")
        return prices

    if path.exists():
        with path.open() as cached:
            fetched_at = cached.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, index_col=0, parse_dates=True)
        return validate(prices), fetched_at

    yf.set_tz_cache_location(str(CACHE))
    for attempt in range(2):
        try:
            batch = yf.download(
                tickers, start=request[0], end=request[1], auto_adjust=True,
                progress=False, threads=4, timeout=12,
            )
            prices = validate(batch["Close"])
            break
        except (KeyError, ValueError) as error:
            if attempt == 1:
                raise RuntimeError(f"Yahoo Finance price download failed: {error}") from error
            sleep(2)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as temporary:
        temp_path = Path(temporary.name)
        try:
            temporary.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(temporary)
            temporary.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def gaussian_emissions(observations, means, variances):
    centered = observations[:, None, :] - means
    log_density = -0.5 * (np.log(variances) + centered**2 / variances).sum(axis=2)
    offset = log_density.max(axis=1)
    return np.exp(np.maximum(log_density - offset[:, None], -700)), offset


def forward_filter(emissions, start_probs, transition):
    filtered = np.empty_like(emissions)
    scales = np.empty(len(emissions))
    prior = start_probs
    for step, likelihood in enumerate(emissions):
        unscaled = prior * likelihood
        scales[step] = unscaled.sum()
        filtered[step] = unscaled / scales[step]
        prior = filtered[step] @ transition
    return filtered, scales


def bull_probability(observations, model):
    """Fit a diagonal Gaussian HMM by EM, then filter with the sticky transition."""
    halves = np.array_split(observations[np.argsort(observations[:, 1], kind="stable")], 2)
    means = np.stack([half.mean(axis=0) for half in halves])
    variance_floor = np.maximum(observations.var(axis=0), 1e-12) * model["hmm_relative_variance_floor"]
    variances = np.maximum(np.stack([half.var(axis=0) for half in halves]), variance_floor)
    start_probs = np.full(2, 0.5)
    persistence = model["hmm_initial_self_transition"]
    transition = np.full((2, 2), 1 - persistence)
    np.fill_diagonal(transition, persistence)
    previous_log_likelihood = -np.inf

    for _ in range(model["hmm_max_iterations"]):
        emissions, offset = gaussian_emissions(observations, means, variances)
        filtered, scales = forward_filter(emissions, start_probs, transition)
        log_likelihood = (np.log(scales) + offset).sum()
        if abs(log_likelihood - previous_log_likelihood) < model["hmm_tolerance"]:
            break
        previous_log_likelihood = log_likelihood

        backward = np.ones_like(filtered)
        for step in range(len(observations) - 2, -1, -1):
            backward[step] = transition @ (emissions[step + 1] * backward[step + 1]) / scales[step + 1]
        state_probs = filtered * backward
        state_probs /= state_probs.sum(axis=1, keepdims=True)
        next_likelihood = emissions[1:] * backward[1:] / scales[1:, None]
        expected_transitions = transition * (filtered[:-1].T @ next_likelihood)
        expected_transitions = np.maximum(expected_transitions, 1e-12)
        transition = expected_transitions / expected_transitions.sum(axis=1, keepdims=True)
        start_probs = np.maximum(state_probs[0], 1e-12)
        start_probs /= start_probs.sum()
        mass = state_probs.sum(axis=0)
        means = (state_probs.T @ observations) / mass[:, None]
        residuals = observations[:, None, :] - means
        variances = np.maximum(
            np.einsum("tk,tkd->kd", state_probs, residuals**2) / mass[:, None],
            variance_floor,
        )

    kappa = model["sticky_kappa"]
    sticky_transition = (transition + kappa * np.eye(2)) / (1 + kappa)
    emissions, _ = gaussian_emissions(observations, means, variances)
    filtered, _ = forward_filter(emissions, start_probs, sticky_transition)
    return filtered[-1, means[:, 0].argmax()]


def downside_covariance(returns, hurdle, eigenvalue_floor):
    downside = np.minimum(returns - hurdle[:, None], 0)
    covariance = downside.T @ downside / len(downside)
    ridge = max(0.0, eigenvalue_floor - np.linalg.eigvalsh(covariance)[0])
    return covariance + ridge * np.eye(covariance.shape[0])


def simplex_weights(covariance, excess_mean, bull, model, previous=None):
    """Solve the paper's long-only objective, using an epigraph for the L1 term."""
    size = len(excess_mean)
    if size == 1:
        return np.ones(1)
    risk_weight = 0.5 if bull else 1.0
    reward = model["bull_return_weight"] * excess_mean if bull else np.zeros(size)
    penalty = model["within_cluster_turnover_penalty"] if previous is not None else 0.0
    initial = previous if previous is not None else np.full(size, 1 / size)

    if penalty:
        initial = np.concatenate((initial, np.zeros(size)))
        eye = np.eye(size)
        absolute_value = np.block([[-eye, eye], [eye, eye]])
        rhs = np.concatenate((-previous, previous))
        trade_constraint = [{
            "type": "ineq", "fun": lambda x: absolute_value @ x - rhs,
            "jac": lambda x: absolute_value,
        }]
    else:
        trade_constraint = []

    budget = np.r_[np.ones(size), np.zeros(len(initial) - size)]

    def objective(x):
        weights = x[:size]
        return risk_weight * (weights @ covariance @ weights) - reward @ weights + penalty * x[size:].sum()

    def gradient(x):
        weight_gradient = 2 * risk_weight * (covariance @ x[:size]) - reward
        return np.r_[weight_gradient, np.full(len(x) - size, penalty)]

    solution = minimize(
        objective, initial, jac=gradient, method="SLSQP", bounds=[(0.0, 1.0)] * len(initial),
        constraints=[{"type": "eq", "fun": lambda x: budget @ x - 1, "jac": lambda x: budget}]
        + trade_constraint,
        options={"ftol": 1e-12, "maxiter": 200},
    )
    if not solution.success:
        raise RuntimeError(f"Portfolio optimization failed: {solution.message}")
    weights = np.maximum(solution.x[:size], 0.0)
    return weights / weights.sum()


def nested_weights(returns, hurdle, previous, bull, model):
    floor = model["downside_eigenvalue_floor"]
    covariance = downside_covariance(returns, hurdle, floor)
    scales = np.sqrt(covariance.diagonal())
    correlation = np.clip(covariance / np.outer(scales, scales), -1, 1)
    distance = np.sqrt(0.5 * (1 - correlation))
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="ward")
    groups = cut_tree(tree, n_clusters=[model["clusters"]]).ravel()
    excess_mean = (returns - hurdle[:, None]).mean(axis=0)
    local = np.zeros((returns.shape[1], model["clusters"]))
    for cluster in range(local.shape[1]):
        members = np.flatnonzero(groups == cluster)
        reference = previous[members]
        mass = reference.sum()
        reference = reference / mass if mass > 1e-10 else np.full(len(members), 1 / len(members))
        local[members, cluster] = simplex_weights(
            covariance[np.ix_(members, members)], excess_mean[members], bull, model, reference,
        )

    cluster_returns = returns @ local
    cluster_weights = simplex_weights(
        downside_covariance(cluster_returns, hurdle, floor), excess_mean @ local, bull, model,
    )
    return local @ cluster_weights


def build_targets(prices, config):
    model = config["model"]
    execution = config["execution"]
    sectors = config["universe"]
    if not 2 <= model["clusters"] <= len(sectors):
        raise ValueError("clusters must be between two and the number of sectors.")
    for key in ("lag_sessions", "vix_publication_lag_sessions", "irx_publication_lag_sessions"):
        if not isinstance(execution[key], int) or execution[key] < 1:
            raise ValueError(f"{key} must be at least one equity session.")

    sessions = prices.index
    month_ends = prices.groupby(sessions.to_period("M")).tail(1).index
    monthly_returns = prices.loc[month_ends, sectors + [config["benchmark"]]].pct_change(fill_method=None)
    observed_vix = prices[config["volatility_index"]].shift(execution["vix_publication_lag_sessions"])
    vix = observed_vix.loc[month_ends]
    history = vix.expanding(min_periods=model["vix_zscore_min_months"])
    vix_z = ((vix - history.mean()) / history.std().replace(0.0, np.nan)).fillna(0.0)
    irx = prices[config["risk_free"]].shift(execution["irx_publication_lag_sessions"])
    monthly_hurdle = irx.loc[month_ends].shift(1) / 1200
    equity_returns = monthly_returns[sectors].to_numpy()
    features = np.column_stack((monthly_returns[config["benchmark"]], vix_z))
    lookback = model["lookback_months"]
    confidence = model["confidence"]
    bull = True
    previous = np.full(len(sectors), 1 / len(sectors))
    targets = {}

    for month in range(lookback, len(month_ends)):
        signal_date = month_ends[month]
        fill_index = sessions.get_loc(signal_date) + execution["lag_sessions"]
        if fill_index >= len(sessions):
            break
        window = slice(month - lookback + 1, month + 1)
        hurdle = monthly_hurdle.iloc[window].to_numpy()
        if not np.isfinite(hurdle).all():
            raise ValueError("Insufficient ^IRX publication history for the first estimation window.")
        probability = bull_probability(features[1:month + 1], model)
        if probability >= confidence:
            bull = True
        elif probability <= 1 - confidence:
            bull = False
        previous = nested_weights(equity_returns[window], hurdle, previous, bull, model)
        targets[sessions[fill_index]] = previous
    if not targets:
        raise ValueError("No filled rebalance. Extend the sample beyond the model warm-up.")
    return targets, irx


def simulate(returns, targets, fee_rate):
    """Mark old positions at the fill close, then pay costs and set new weights."""
    weights = np.zeros(returns.shape[1])
    rows = []
    for date, asset_returns in zip(returns.index, returns.to_numpy()):
        gross_value = 1 + weights @ asset_returns
        weights = weights * (1 + asset_returns) / gross_value
        target = targets.get(date)
        turnover, trades = 0.0, 0
        if target is not None:
            # Trades fund their own costs, so target dollars use post-cost NAV.
            def remaining_nav(fraction):
                return fraction + fee_rate * np.abs(fraction * target - weights).sum() - 1

            fraction = brentq(remaining_nav, 0.0, 1.0)
            trades_by_asset = np.abs(fraction * target - weights)
            turnover = trades_by_asset.sum()
            trades = np.count_nonzero(trades_by_asset > 1e-8)
            gross_value *= fraction
            weights = target
        rows.append((gross_value - 1, turnover, trades))
    return pd.DataFrame(rows, index=returns.index, columns=["net_return", "turnover", "n_trades"])


def period_metrics(period, risk_free, prior_sessions):
    if len(period) < 2 or risk_free.reindex(period.index).isna().any():
        raise ValueError("Each reported period needs two or more sessions with risk-free data.")
    returns = period["net_return"].to_numpy()
    capital = np.r_[1.0, np.cumprod(1 + returns)]
    years = (period.index[-1] - prior_sessions.loc[period.index[0]]).days / 365.25

    def sharpe(values):
        vol = values.std(ddof=1)
        return float(np.sqrt(SESSIONS_PER_YEAR) * values.mean() / vol) if vol > 0 else None

    return {
        "cagr": float(capital[-1] ** (1 / years) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - risk_free.loc[period.index].to_numpy()),
        "max_drawdown": float((capital / np.maximum.accumulate(capital) - 1).min()),
        "total_return": float(capital[-1] - 1),
        "turnover": float(period["turnover"].sum() / years),
        "n_trades": int(period["n_trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    prices, fetched_at = load_prices(config)
    targets, irx = build_targets(prices, config)
    benchmark = config["benchmark"]
    sectors = config["universe"]
    first_fill = min(targets)
    daily_returns = prices[sectors + [benchmark]].pct_change(fill_method=None).loc[first_fill:]
    execution = config["execution"]
    fee_rate = (execution["costs_bps"] + execution["slippage_bps"]) / 10_000
    if min(execution["costs_bps"], execution["slippage_bps"]) < 0 or not 0 <= fee_rate < 1:
        raise ValueError("Execution costs and slippage must be nonnegative and sum to less than 100%.")
    strategy = simulate(daily_returns[sectors], targets, fee_rate)
    passive = simulate(daily_returns[[benchmark]], {first_fill: np.ones(1)}, fee_rate)
    prior_sessions = prices.index.to_series().shift(1)
    day_count = (prices.index.to_series() - prior_sessions).dt.days
    risk_free = irx * day_count / 36_500
    is_end = pd.Timestamp(config["dates"]["is_end"])

    def summarize(portfolio):
        periods = {"is": portfolio.loc[:is_end], "oos": portfolio.loc[portfolio.index > is_end], "full": portfolio}
        return {name: period_metrics(period, risk_free, prior_sessions) for name, period in periods.items()}

    benchmark_metrics = summarize(passive)
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(first_fill.date()), "end": str(strategy.index[-1].date())},
        **summarize(strategy),
        "benchmark": {"name": benchmark, **benchmark_metrics["full"], **benchmark_metrics},
    }
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return ETF closes; ^VIX and ^IRX levels")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")
    print(f"DATA_INPUT_RANGE: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"EVALUATION_RANGE: {first_fill.date()} to {strategy.index[-1].date()} (IS ends {is_end.date()})")
    for name in ("is", "oos", "full", "benchmark"):
        print(f"{name.upper()}: {json.dumps(result[name], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
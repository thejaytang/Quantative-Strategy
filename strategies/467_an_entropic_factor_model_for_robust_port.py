"""467_an_entropic_factor_model_for_robust_port
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/467_an_entropic_factor_model_for_robust_port.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e0683ac343a175cb2458afc860f8d1a1066c8e3a on 2026-11-27 08:55 Europe/Oslo, exit 0, output sha256 304732ff0d7ada520e65ce164cfd353a52a72ccc8586a915b3c8e29eed987db0
Data source: yfinance 提供的免费 Yahoo Finance 日线。股票池为 XOM、CVX、MSFT、AAPL、IBM、INTC、CSCO、AMZN、JPM、BAC、GS、AXP、JNJ、PFE、MRK、PG、KO、WMT、MCD、BA。SPY 为市场因子和买入持有基准，BIL 仅作为超额夏普的现金代理。见。
Data fetched: 2026-11-27T07:55:50+00:00
Sample period: 配置日期为 2018-01-01 至 2023-12-31。日志记录的实际价格样本为 2018-01-02 至 2023-12-29，首笔模拟成交和评估起点为 2019-01-04。见。 data_range (from output): {'start': '2018-01-02', 'end': '2023-12-29', 'evaluation_start': '2019-01-04', 'is_end': '2021-04-20'}
IS/OOS split: 按日期切分同一次连续回测。样本内为 2019-01-04 至 2021-04-20（含），样本外为 2021-04-21 至 2023-12-29。持仓跨分界延续，样本外继续以每次信号当时的最近 252 个交易日收益滚动估计。见。
Costs and slippage: 模拟计入 10 bps 交易成本与 5 bps 固定滑点，合计 15 bps，按建仓和再平衡时买入及卖出名义金额的绝对值之和扣除。成交金额依据扣除摩擦后的目标净值计算。SPY 买入持有基准按同期调整价收益计量，未扣此项摩擦。见。
Signal lag: 使用截至 t 日收盘（含）的收益计算目标权重，默认在下一交易日 t+1 收盘模拟成交。t 至 t+1 的收盘到收盘收益归成交前持仓，新持仓从 t+1 收盘后开始承担收益。见。
Price adjustment: yfinance auto_adjust total-return close
Survivorship bias: 按配置使用论文实验的固定 20 只股票，未按历史时点重建股票池。README 披露幸存者偏差及事后筛选偏差，整个历史样本早于论文发布。见。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.18555862301711135, ann_vol=0.2534800291597501, sharpe=0.7989961982958048, sharpe_excess=0.7586283015220004, max_drawdown=-0.34796182412188925, total_return=0.4765912122814211, turnover=1.9517349421957848, n_trades=560
    oos: cagr=0.0962366015843863, ann_vol=0.16262618715647573, sharpe=0.6463930627178242, sharpe_excess=0.5057271508316827, max_drawdown=-0.2146561490531641, total_return=0.28091383094937705, turnover=1.3038788788022697, n_trades=640
    full: cagr=0.13640289656724902, ann_vol=0.20925255483562352, sharpe=0.7162144427894364, sharpe_excess=0.6346343103456454, max_drawdown=-0.34796182412188925, total_return=0.8913861064695805, turnover=1.6015006531478573, n_trades=1200
    benchmark: cagr=0.16183424000791158, ann_vol=0.2097530103693755, sharpe=0.8207033538077634, sharpe_excess=0.7392701910282528, max_drawdown=-0.3371727036119295, total_return=1.1119655115432958, turnover=0.0, n_trades=0
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "467_ an_entropic_factor_model_for_robust_port"\nuniverse: [XOM, CVX, MSFT, AAPL, IBM, INTC, CSCO, AMZN, JPM, BAC, GS, AXP, JNJ, PFE, MRK, PG, KO, WMT, MCD, BA]\nbenchmark: SPY\nrisk_free: BIL\ndates:\n  start: 2018-01-01\n  end: 2023-12-31\n  is_end: 2021-04-20\nmodel:\n  lookback_sessions: 252\n  rebalance_every_sessions: 21\n  target_beta: 1.0\n  residual_padding: 1.05\n  residual_vol_floor: 0.5\nexecution:\n  signal_lag_sessions: 1\n  cost_bps: 10.0\n  slippage_bps: 5.0\n'
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
"""Rolling two-stage entropic replication of an equity benchmark."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from scipy.special import expit, xlogy


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252


def validate_config(config):
    assets = config["universe"]
    symbols = [*assets, config["benchmark"], config["risk_free"]]
    if len(assets) < 2 or len(set(symbols)) != len(symbols):
        raise ValueError("Use distinct assets, a distinct benchmark and BIL for cash comparison.")
    if config["risk_free"] != "BIL":
        raise ValueError("This implementation uses BIL for excess Sharpe.")
    start, split, end = (pd.Timestamp(config["dates"][key]) for key in ("start", "is_end", "end"))
    if not start < split < end:
        raise ValueError("Dates must be ordered start < is_end < end.")
    model = config["model"]
    execution = config["execution"]
    for value, minimum in ((model["lookback_sessions"], 3),
                           (model["rebalance_every_sessions"], 1),
                           (execution["signal_lag_sessions"], 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"Session counts must be integers of at least {minimum}.")
    fees = np.array([execution["cost_bps"], execution["slippage_bps"]], dtype=float)
    if not np.isfinite(fees).all() or (fees < 0).any() or fees.sum() >= 10_000:
        raise ValueError("Costs and slippage must be nonnegative and sum to less than 10,000 bps.")
    if not np.isfinite([model["target_beta"], model["residual_padding"], model["residual_vol_floor"]]).all():
        raise ValueError("Model constants must be finite.")
    if model["residual_padding"] <= 1 or model["residual_vol_floor"] <= 0:
        raise ValueError("Residual padding must exceed 1 and the volatility floor must be positive.")


def load_close(symbols, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = sha256(json.dumps(["adjusted-close-v1", symbols, start, end]).encode()).hexdigest()[:16]
    cache_file = cache_dir / f"yahoo_daily_{cache_key}.csv"
    if cache_file.exists():
        with cache_file.open() as source:
            fetched_at = source.readline().strip().removeprefix("# ")
            datetime.fromisoformat(fetched_at)
            close = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
    else:
        end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
        with TemporaryDirectory(prefix="yahoo_session_", dir=cache_dir) as session_dir:
            yf.set_tz_cache_location(session_dir)
            raw = yf.download(symbols, start=start, end=end_exclusive, auto_adjust=True,
                              progress=False, threads=4, timeout=15)
        if raw.empty or "Close" not in raw:
            raise RuntimeError("Yahoo Finance returned no adjusted closing prices.")
        close = raw["Close"].reindex(columns=symbols).sort_index()
        close.index = pd.DatetimeIndex(close.index).tz_localize(None)
        close.index.name = "Date"
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    close = close.reindex(columns=symbols).loc[start:end]
    bad = ~np.isfinite(close) | (close <= 0)
    if close.empty or bad.any().any():
        missing = ", ".join(bad.columns[bad.any()])
        raise RuntimeError(f"Yahoo price history is empty or has missing or invalid prices: {missing}")
    if close.index.has_duplicates or not close.index.is_monotonic_increasing:
        raise RuntimeError("Closing prices must have unique dates in chronological order.")
    if max(close.index[0] - pd.Timestamp(start), pd.Timestamp(end) - close.index[-1]).days > 7:
        raise RuntimeError("Yahoo price history does not cover the requested date range.")
    if not cache_file.exists():
        with NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                temp.write(f"# {fetched_at}\n")
                close.to_csv(temp)
                temp.flush()
                temp_path.replace(cache_file)
            finally:
                temp_path.unlink(missing_ok=True)
    return close, fetched_at


def entropy(centered):
    probability = 0.5 * (1 + centered)
    complement = 0.5 * (1 - centered)
    return np.sum(xlogy(probability, probability) + xlogy(complement, complement))


def fit_beta(factor, asset, model):
    """Minimize equation (2) over alpha, beta and implied residuals."""
    factor_change = np.diff(factor)
    nonzero = factor_change != 0
    if not nonzero.any():
        raise ValueError("The benchmark return is constant in the estimation window.")
    slopes = np.diff(asset)[nonzero] / factor_change[nonzero]
    beta_low, beta_high = slopes.min(), slopes.max()
    alpha_low = np.min(asset[1:] - beta_high * factor[1:])
    alpha_high = np.max(asset[1:] - beta_low * factor[1:])
    lower = np.array([alpha_low, beta_low])
    upper = np.array([alpha_high, beta_high])
    if (lower > upper).any():
        raise ValueError("The paper's empirical coefficient bounds are inverted.")
    center = (lower + upper) / 2
    halfwidth = (upper - lower) / 2
    design = np.column_stack((np.ones_like(factor), factor))
    midpoint_residual = asset - design @ center
    envelope = max(model["residual_padding"] * np.max(np.abs(midpoint_residual)),
                   model["residual_vol_floor"] * asset.std(ddof=1))
    if envelope == 0:
        return float(center[1])

    # Eliminate residuals using equation (22). In centered box coordinates,
    # the entropy gradient and Hessian are atanh(z) and 1 / (1 - z**2).
    midpoint = np.r_[np.zeros(2), midpoint_residual / envelope]
    transform = np.vstack((np.eye(2), -design * halfwidth / envelope))
    coefficients = np.zeros(2)
    for _ in range(80):
        centered = midpoint + transform @ coefficients
        gradient = transform.T @ np.arctanh(centered)
        hessian = transform.T @ (transform / (1 - centered**2)[:, None])
        step = np.linalg.solve(hessian, gradient)
        decrement = gradient @ step
        if decrement <= 1e-16:
            return float(center[1] + halfwidth[1] * coefficients[1])
        objective = entropy(centered)
        for power in range(50):
            scale = 0.5**power
            trial = coefficients - scale * step
            trial_centered = midpoint + transform @ trial
            if (np.abs(trial_centered) < 1).all() and (
                entropy(trial_centered) <= objective - 1e-4 * scale * decrement + 1e-13
            ):
                coefficients = trial
                break
        else:
            raise RuntimeError("First-stage entropy line search failed.")
    raise RuntimeError("First-stage entropy fit did not converge.")


def replication_weights(betas, target_beta):
    """Solve the long-only entropy problem with budget and beta constraints."""
    low, high = float(betas.min()), float(betas.max())
    reachable = low <= target_beta <= high
    if target_beta <= low or target_beta >= high:
        # At an endpoint, only assets at that beta can receive weight.
        eligible = betas == (low if target_beta <= low else high)
        return eligible / eligible.sum(), reachable

    exposure = (betas - low) / (high - low)
    target = (target_beta - low) / (high - low)

    def weights(beta_multiplier):
        tilt = beta_multiplier * exposure
        # Equation (38) with [0, 1] bounds. Solve for the budget multiplier first.
        budget_multiplier = brentq(lambda offset: expit(tilt + offset).sum() - 1,
                                  -tilt.max() - 40, -tilt.min() + 40, xtol=1e-13)
        return expit(tilt + budget_multiplier)

    def error(beta_multiplier):
        return exposure @ weights(beta_multiplier) - target

    bracket = 1.0
    while error(-bracket) > 0 or error(bracket) < 0:
        bracket *= 2
        if bracket > 1e12:
            raise RuntimeError("Second-stage exposure multiplier could not be bracketed.")
    allocation = weights(brentq(error, -bracket, bracket, xtol=1e-12))
    if not np.isclose(allocation.sum(), 1, atol=1e-9, rtol=0) or not np.isclose(
        betas @ allocation, target_beta, atol=1e-8, rtol=0
    ):
        raise RuntimeError("Second-stage allocation failed its budget or exposure constraint.")
    return allocation / allocation.sum(), reachable


def schedule_targets(returns, config):
    model = config["model"]
    lag = config["execution"]["signal_lag_sessions"]
    window = model["lookback_sessions"]
    market = returns[config["benchmark"]].to_numpy()
    assets = returns[config["universe"]].to_numpy()
    targets = {}
    unreachable = 0
    for signal in range(window - 1, len(returns) - lag, model["rebalance_every_sessions"]):
        history = slice(signal - window + 1, signal + 1)
        betas = np.array([fit_beta(market[history], asset, model) for asset in assets[history].T])
        weights, reachable = replication_weights(betas, model["target_beta"])
        targets[signal + lag] = weights
        unreachable += int(not reachable)
    if not targets:
        raise ValueError("The date range must cover the lookback and one delayed fill.")
    return targets, unreachable


def simulate(returns, targets, friction):
    first_fill = min(targets)
    weights = np.zeros(returns.shape[1])
    cash = 1.0
    rows = []
    for session, asset_returns in enumerate(returns.to_numpy()[first_fill:], start=first_fill):
        # Today's close-to-close return belongs to yesterday's positions.
        marked = weights * (1 + asset_returns)
        gross = marked.sum() + cash
        weights = marked / gross
        turnover = 0.0
        trades = 0
        post_cost = 1.0
        if session in targets:
            target = targets[session]
            # Pay friction on actual purchases and sales, sized from post-cost NAV.
            def cash_balance(scale):
                return scale + friction * np.abs(scale * target - weights).sum() - 1

            post_cost = brentq(cash_balance, 0.0, 1.0, xtol=1e-14)
            orders = np.abs(post_cost * target - weights)
            turnover = orders.sum()
            trades = int(np.count_nonzero(orders > 1e-12))
            weights = target
            cash = 0.0
        rows.append((gross * post_cost - 1, turnover, trades))
    return pd.DataFrame(rows, index=returns.index[first_fill:], columns=["return", "turnover", "trades"])


def metrics(book, cash_proxy):
    returns = book["return"].to_numpy()
    excess = (book["return"] - cash_proxy).to_numpy()
    growth = np.cumprod(np.r_[1.0, 1 + returns])
    years = len(book) / SESSIONS_PER_YEAR
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(annualizer * returns.mean() / vol) if vol > 0 else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(book["turnover"].sum() / years),
        "n_trades": int(book["trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    validate_config(config)
    dates = config["dates"]
    symbols = [*config["universe"], config["benchmark"], config["risk_free"]]
    close, fetched_at = load_close(symbols, str(dates["start"]), str(dates["end"]))
    returns = close.pct_change(fill_method=None).iloc[1:]
    split = pd.Timestamp(dates["is_end"])
    execution = config["execution"]
    friction = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close")
    print("PARAM_COMBOS_TRIED: 1")
    targets, unreachable = schedule_targets(returns, config)
    book = simulate(returns[config["universe"]], targets, friction)
    benchmark = pd.DataFrame({"return": returns.loc[book.index, config["benchmark"]],
                              "turnover": 0.0, "trades": 0})
    print(f"UNREACHABLE_BETA_REBALANCES: {unreachable}/{len(targets)}")

    result = {"strategy": config["strategy"], "data_range": {
        "start": close.index[0].date().isoformat(), "end": close.index[-1].date().isoformat(),
        "evaluation_start": book.index[0].date().isoformat(), "is_end": split.date().isoformat(),
    }}
    periods = {"is": book.index[book.index <= split], "oos": book.index[book.index > split],
               "full": book.index}
    if any(len(index) < 2 for index in periods.values()):
        raise ValueError("At least two evaluated sessions are needed on both sides of the split.")
    for period, index in periods.items():
        cash_proxy = returns.loc[index, config["risk_free"]]
        result[period] = metrics(book.loc[index], cash_proxy)
        comparison = metrics(benchmark.loc[index], cash_proxy)
        if period == "full":
            result["benchmark"] = comparison
        tracking_error = (book.loc[index, "return"] - benchmark.loc[index, "return"]).std()
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
        print(f"BENCHMARK {config['benchmark']} {period.upper()}: {json.dumps(comparison, allow_nan=False)}")
        print(f"TRACKING_ERROR {period.upper()}: {tracking_error * np.sqrt(SESSIONS_PER_YEAR):.8f}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
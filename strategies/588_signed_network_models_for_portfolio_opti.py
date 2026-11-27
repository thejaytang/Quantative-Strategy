"""588_signed_network_models_for_portfolio_opti
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/588_signed_network_models_for_portfolio_opti.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:09 Europe/Oslo, exit 0, output sha256 0759f6c559a590db0b8a4edc1cc09394d2c5980e67306eeb3710deaddac4aca6
Data source: Yahoo Finance 免费 ETF 日线及公司行动，由 yfinance 下载，或读取同一数据源的本地缓存。资产池为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY，SPY 为买入持有基准，BIL 总收益用于计算超额夏普比率。
Data fetched: 2026-11-27T18:09:12+00:00
Sample period: 实际回测区间为 2009-01-02 至 2025-12-31（含首尾）。行情请求从 2007-07-01 开始，回测前行情用于信号窗口预热。   data_range (from output): {'start': '2009-01-02', 'end': '2025-12-31'}
IS/OOS split: 按 2017-01-01 做历史时间切分。IS 为 2009–2016 年，OOS 为 2017–2025 年。正式策略跨分界日延续同一组合路径，年度窗口可向前引用分界日前数据。六组预设稳健性情景只使用 IS 行情。
Costs and slippage: 基准设定每边成本 5 bps、滑点 5 bps，合计按每笔买卖成交额计 10 bps。滑点按固定基点计入摩擦，卖出净收入乘 1−摩擦，买入总支出乘 1+摩擦，并按扣费后可用现金同比缩减买入。样本内双倍摩擦情景为每边合计 20 bps。
Signal lag: 在每年最后一个交易日收盘后，以截至该收盘（含当日）的收益窗口计算目标权重。配置滞后 1 个交易日，按次年首个交易日开盘成交。
Price adjustment: yfinance split-adjusted Open/Close; ex-date total-return from prior close less distributions
Survivorship bias: 固定使用文档所述的 1998 年首批九只 Select Sector SPDR，名单早于回测起点。行业 ETF 的成分和分类曾调整，后续成立的行业 ETF 未纳入，样本依赖 Yahoo 的历史覆盖。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.13444481121961038, ann_vol=0.15546037456600706, sharpe=0.8894827465956923, sharpe_excess=0.8882801451284964, max_drawdown=-0.19966111486529514, total_return=1.7405150776202225, turnover=1.561541167031756, n_trades=29
    oos: cagr=0.10125861435649264, ann_vol=0.17141510110787184, sharpe=0.6486570042858399, sharpe_excess=0.5181620726273266, max_drawdown=-0.34614144507924327, total_return=1.3768758755913595, turnover=1.467366831002098, n_trades=52
    full: cagr=0.11676663502967255, ann_vol=0.16407701391446516, sharpe=0.7554311485470768, sharpe_excess=0.6829151665794266, max_drawdown=-0.34614144507924316, total_return=5.513864174689868, turnover=1.5117230313677976, n_trades=81
    benchmark: cagr=0.14715159209297157, ann_vol=0.17899107555847, sharpe=0.856855238570891, sharpe_excess=0.7903102734646046, max_drawdown=-0.33717263923504426, total_return=9.272143627732245, turnover=0.058874708079572444, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 588_signed_network_models_for_portfolio_opti\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nbenchmark: SPY\nrisk_free: BIL\n\nhistory_start: 2007-07-01\nstart: 2009-01-01\noos_start: 2017-01-01\nend: 2025-12-31\n\nlookback_sessions: 252\nsubset_size: 5\nsignal_lag_sessions: 1\ncost_bps: 5.0\nslippage_bps: 5.0\n\nrobustness:\n  is_lookback_sessions: [210, 294]\n  is_subset_sizes: [4, 6]\n  is_friction_multiplier: 2.0\n'
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
"""Research backtest of Adhikari (2025), https://arxiv.org/abs/2510.05377.

Related implementations consulted for experimental design, with original code here:
- Riskfolio-Lib (BSD-3-Clause): https://github.com/dcajasn/Riskfolio-Lib
  https://github.com/dcajasn/Riskfolio-Lib/blob/master/examples/Tutorial%2045%20-%20Classic%20Mean%20Risk%20Optimization%20with%20Network%20and%20Dendrogram%20Constraints.ipynb
  https://pypi.org/project/riskfolio-lib/
- Gregnanin et al. (no license located): https://github.com/GeNiN01/Signature-Based-Portfolio-Allocation
- Search comparisons: https://www.quantconnect.com/docs/v2/writing-algorithms/algorithm-framework/portfolio-construction/supported-models
  https://notebook.community/quantopian/research_public/notebooks/lectures/Factor_Based_Risk_Management/notebook
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance split-adjusted Open/Close; ex-date total-return from prior close less distributions"


@dataclass(frozen=True)
class Parameters:
    lookback: int
    subset_size: int
    friction: float


def load_prices(tickers, history_start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = f"yahoo-v1:{history_start.date()}:{end.date()}:{','.join(tickers)}"
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    cache_path = cache_dir / f"prices_{digest}.csv"
    stamp_prefix = "DATA_FETCHED_AT: "
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as source:
            stamp = source.readline().strip()
            if not stamp.startswith(stamp_prefix):
                raise ValueError(f"Missing fetch timestamp in {cache_path}")
            fetched_at = stamp.removeprefix(stamp_prefix)
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        validate_prices(prices, tickers, history_start, end)
        return prices, fetched_at

    # Keep yfinance's temporary service caches inside this strategy directory.
    with TemporaryDirectory(dir=cache_dir) as session_dir:
        yf.set_tz_cache_location(session_dir)
        raw = yf.download(
            tickers, start=history_start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=False, actions=True, timeout=12, threads=4, progress=False,
        )
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices")
    distributions = raw["Dividends"]
    if "Capital Gains" in raw.columns.get_level_values(0):
        # This action column may be absent for tickers with no such events.
        distributions = distributions + raw["Capital Gains"].reindex(columns=tickers).fillna(0.0)
    prices = pd.concat({"Open": raw["Open"], "Close": raw["Close"], "Distribution": distributions}, axis=1)
    validate_prices(prices, tickers, history_start, end)

    # Publish the prices and fetch timestamp together, even with concurrent runs.
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir, delete=False) as staged:
        try:
            staged.write(stamp_prefix + fetched_at + "\n")
            prices.to_csv(staged)
            staged.close()
            Path(staged.name).replace(cache_path)
        finally:
            Path(staged.name).unlink(missing_ok=True)
    return prices, fetched_at


def validate_prices(prices, tickers, history_start, end):
    dates = prices.index
    if (prices.empty or not dates.is_monotonic_increasing or not dates.is_unique
            or dates[0] < history_start or dates[-1] > end
            or (dates[0] - history_start).days > 7 or (end - dates[-1]).days > 7):
        raise ValueError("Yahoo Finance coverage does not match the requested date range")
    for field in ("Open", "Close", "Distribution"):
        values = prices[field].reindex(columns=tickers)
        invalid = ~np.isfinite(values) | (values < 0 if field == "Distribution" else values <= 0)
        if invalid.to_numpy().any():
            first = invalid.stack().loc[lambda flags: flags].index[0]
            raise ValueError(f"Missing or invalid {field} for {first[1]} on {first[0].date()}")


def gross_price_moves(prices):
    # The ex-date distribution is public before the opening print. Yahoo's
    # split basis is common to the prior close, distribution and current open.
    ex_date_base = prices["Close"].shift(1) - prices["Distribution"]
    if (ex_date_base.iloc[1:] <= 0).to_numpy().any():
        raise ValueError("A distribution is at least as large as the previous close")
    overnight = prices["Open"].div(ex_date_base).iloc[1:]
    intraday = prices["Close"].div(prices["Open"]).iloc[1:]
    return overnight, intraday


def year_end_signals(dates, start, lag):
    # A year-end close is followed by the first session of the next year.
    year_ends = np.flatnonzero(dates.year[:-1] != dates.year[1:])
    return [(int(i), int(i + lag)) for i in year_ends
            if i + lag < len(dates) and dates[i + lag] >= start]


def selection_scores(window):
    means = window.mean(axis=0)
    above = (window > means).astype(np.int64)
    below = (window < means).astype(np.int64)
    negative_edges = above.T @ below + below.T @ above
    frequencies = negative_edges.sum(axis=1) / (window.shape[0] * (window.shape[1] - 1))
    return frequencies * means


def minimum_variance(window):
    covariance = np.cov(window, rowvar=False)
    scale = max(np.diag(covariance).mean(), np.finfo(float).tiny)
    covariance /= scale  # Condition the objective without changing its minimizer.
    n_assets = covariance.shape[0]
    result = minimize(
        lambda w: float(w @ covariance @ w),
        np.full(n_assets, 1.0 / n_assets),
        jac=lambda w: 2.0 * (covariance @ w),
        method="SLSQP", bounds=[(0.0, 1.0)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-12, "maxiter": 200},
    )
    weights = result.x
    if (not result.success or not np.isfinite(weights).all()
            or weights.min() < -1e-8 or abs(weights.sum() - 1.0) > 1e-8):
        raise RuntimeError(f"Minimum-variance solve failed: {result.message}")
    weights = np.maximum(weights, 0.0)
    return weights / weights.sum()


def build_targets(returns, schedule, parameters):
    targets = {}
    for signal_row, fill_row in schedule:
        window_start = signal_row + 1 - parameters.lookback
        if window_start < 0:
            raise ValueError(f"Insufficient warm-up at {returns.index[signal_row].date()}")
        window = returns.iloc[window_start:signal_row + 1].to_numpy()
        if parameters.subset_size == window.shape[1]:
            subset = np.arange(window.shape[1])
        else:
            subset = np.argsort(-selection_scores(window), kind="stable")[:parameters.subset_size]
        weights = np.zeros(returns.shape[1])
        weights[subset] = minimum_variance(window[:, subset])
        targets[returns.index[fill_row]] = weights
    return targets


def rebalance(open_values, cash, weights, friction):
    value_at_open = float(open_values.sum()) + cash
    differences = value_at_open * weights - open_values
    sells = np.maximum(-differences, 0.0)
    buys = np.maximum(differences, 0.0)
    spendable = cash + float(sells.sum()) * (1.0 - friction)
    proposed = float(buys.sum())
    if proposed > 0.0:
        buys *= min(1.0, spendable / (proposed * (1.0 + friction)))
    remaining = spendable - float(buys.sum()) * (1.0 + friction)
    if remaining < -1e-12 * value_at_open:
        raise RuntimeError("Rebalance would leave negative cash")
    traded = sells + buys
    return (
        open_values - sells + buys,
        max(remaining, 0.0),
        float(traded.sum()) / value_at_open,
        int(np.count_nonzero(traded > 1e-10 * value_at_open)),
    )


def simulate(overnight, intraday, targets, start, friction):
    overnight = overnight.loc[start:]
    intraday = intraday.loc[start:]
    positions = np.zeros(overnight.shape[1])
    cash, previous_value = 1.0, 1.0
    rows = []
    for date, before_open, after_open in zip(overnight.index, overnight.to_numpy(), intraday.to_numpy()):
        positions *= before_open
        turnover, n_trades = 0.0, 0
        if date in targets:
            positions, cash, turnover, n_trades = rebalance(positions, cash, targets[date], friction)
        positions *= after_open
        value = float(positions.sum()) + cash
        rows.append((value / previous_value - 1.0, turnover, n_trades))
        previous_value = value
    return pd.DataFrame(rows, index=overnight.index, columns=["return", "turnover", "n_trades"])


def summarize(daily_results, risk_free):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Each period needs at least two price sessions")
    excess = returns - risk_free.loc[returns.index]
    volatility, excess_volatility = returns.std(), excess.std()
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    years = len(returns) / SESSIONS_PER_YEAR
    growth = np.concatenate(([1.0], np.cumprod(1.0 + returns.to_numpy())))
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(annualizer * volatility),
        "sharpe": float(annualizer * returns.mean() / volatility) if volatility > 0 else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_volatility) if excess_volatility > 0 else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def periods(daily_results, risk_free, oos_start):
    return {
        "is": summarize(daily_results.loc[daily_results.index < oos_start], risk_free),
        "oos": summarize(daily_results.loc[daily_results.index >= oos_start], risk_free),
        "full": summarize(daily_results, risk_free),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    history_start, start, oos_start, end = (
        pd.Timestamp(config[key]) for key in ("history_start", "start", "oos_start", "end")
    )
    universe, benchmark, risk_free_ticker = config["universe"], config["benchmark"], config["risk_free"]
    tickers = [*universe, benchmark, risk_free_ticker]
    lag = config["signal_lag_sessions"]
    cost_bps, slippage_bps = (float(config[key]) for key in ("cost_bps", "slippage_bps"))
    base = Parameters(config["lookback_sessions"], config["subset_size"], (cost_bps + slippage_bps) / 10_000)
    robustness = config["robustness"]
    lookbacks = robustness["is_lookback_sessions"]
    sizes = robustness["is_subset_sizes"]
    multiplier = float(robustness["is_friction_multiplier"])

    if not history_start < start < oos_start < end:
        raise ValueError("Expected history_start < start < oos_start < end")
    if len(tickers) != len(set(tickers)) or risk_free_ticker != "BIL":
        raise ValueError("Use distinct tickers and BIL for the excess-return reference")
    if type(lag) is not int or lag < 1:
        raise ValueError("The signal lag must be at least one session")
    if any(type(n) is not int or n < 2 for n in [base.lookback, *lookbacks]):
        raise ValueError("Lookbacks must be integer counts of at least two sessions")
    if any(type(n) is not int or not 2 <= n < len(universe) for n in [base.subset_size, *sizes]):
        raise ValueError("Subset sizes must be at least two and smaller than the universe")
    if not (cost_bps >= 0 and slippage_bps >= 0 and multiplier > 1 and 0 < base.friction * multiplier < 1):
        raise ValueError("Expected nonnegative costs and slippage with a valid higher-friction case")

    variants = [(f"LOOKBACK_{n}", replace(base, lookback=n)) for n in lookbacks]
    variants += [(f"SUBSET_{n}", replace(base, subset_size=n)) for n in sizes]
    variants += [
        (f"FRICTION_{multiplier:g}X", replace(base, friction=base.friction * multiplier)),
        ("ALL_ASSETS_MIN_VARIANCE", replace(base, subset_size=len(universe))),
    ]
    if min(len(lookbacks), len(sizes)) < 2 or len({base, *(p for _, p in variants)}) != 1 + len(variants):
        raise ValueError("Specify two or more distinct variants for each main parameter")

    prices, fetched_at = load_prices(tickers, history_start, end)
    overnight, intraday = gross_price_moves(prices)
    daily_returns = overnight * intraday - 1.0
    asset_returns = daily_returns[universe]
    risk_free = daily_returns[risk_free_ticker]
    asset_overnight, asset_intraday = overnight[universe], intraday[universe]
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"EXECUTION: year-end close signal, open fill after {lag} session(s), "
          f"cost {cost_bps:g} bps and slippage {slippage_bps:g} bps per side")

    print("ROBUSTNESS (IS only, preset cases)")
    is_mask = daily_returns.index < oos_start
    is_returns = asset_returns.loc[is_mask]
    is_schedule = year_end_signals(is_returns.index, start, lag)
    for label, parameters in variants:
        targets = build_targets(is_returns, is_schedule, parameters)
        daily = simulate(asset_overnight.loc[is_mask], asset_intraday.loc[is_mask],
                         targets, start, parameters.friction)
        print(f"IS_{label}: {json.dumps(summarize(daily, risk_free), allow_nan=False)}")

    schedule = year_end_signals(asset_returns.index, start, lag)
    targets = build_targets(asset_returns, schedule, base)
    daily = simulate(asset_overnight, asset_intraday, targets, start, base.friction)
    strategy_metrics = periods(daily, risk_free, oos_start)
    if min(strategy_metrics[period]["n_trades"] for period in ("is", "oos")) <= 0:
        raise RuntimeError("The official run needs trades in both periods")
    first_open = daily.index[0]
    benchmark_daily = simulate(overnight[[benchmark]], intraday[[benchmark]],
                               {first_open: np.ones(1)}, start, base.friction)
    benchmark_metrics = periods(benchmark_daily, risk_free, oos_start)

    print(f"PARAM_COMBOS_TRIED: {1 + len(variants)}")
    for period, stats in strategy_metrics.items():
        print(f"{period.upper()}: {json.dumps(stats, allow_nan=False)}")
    for period, stats in benchmark_metrics.items():
        print(f"BENCHMARK_{period.upper()} ({benchmark}): {json.dumps(stats, allow_nan=False)}")

    result = {
        "strategy": config["strategy"],
        "data_range": {"start": first_open.date().isoformat(), "end": daily.index[-1].date().isoformat()},
        **strategy_metrics,
        "benchmark": benchmark_metrics["full"],
    }
    print("已回测 · 待审查")
    print(f"RESULTS_JSON: {json.dumps(result, allow_nan=False)}")


if __name__ == "__main__":
    signal.alarm(12 * 60)
    try:
        main()
    finally:
        signal.alarm(0)

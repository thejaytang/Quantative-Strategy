"""487_conditional_value_at_risk_under_reward_p
WARNING: known defects (see DEEP_REVIEW_FINDINGS.md); numbers invalid.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/487_conditional_value_at_risk_under_reward_p.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit fb92b5054b1596d6a39ea87072bbd5a288327fb7 on 2026-11-27 09:41 Europe/Oslo, exit 0, output sha256 567d01571adca6319bd9089187cc4d3d79e08c59fd7ddcc917a7e994ff9c0efc
Data source: Yahoo Finance 免费日线，经 yfinance 下载或从本地价格快照读取。股票池为 AAPL、NVDA、MSFT、JNJ、ABBV、LLY、BRK-B、JPM、V、GOOG、META、NFLX、WMT、COST、KO。SPY 为基准，BIL 为现金回报代理，仅用于绩效计算。
Data fetched: 2026-11-27T08:41:40+00:00
Sample period: 价格区间为 2013-01-02 至 2025-02-28，评估区间为 2017-03-01 至 2025-02-28。评估开始前的价格用于滚动估计预热。 data_range (from output): 2017-03-01 to 2025-02-28
IS/OOS split: 按交易日顺序切分。样本内为 2017-03-01 至 2020-02-28，样本外为 2020-03-02 至 2025-02-28。配置切点为 2020-02-29，切点当日或之前归入样本内，之后归入样本外。五个参数对照只跑样本内，样本外使用默认设定并延续样本内组合，信号继续按滚动窗口更新。
Costs and slippage: 默认每笔买入或卖出按绝对成交额计 5 bps 费用和 5 bps 固定滑点，合计 10 bps。滑点建模为成交额比例扣减。模拟器先按旧仓位计算成交日收盘到收盘收益，再按扣费后的目标持仓与成交前漂移持仓之差求成交额，当日扣减费用和滑点。策略与 SPY 基准均计入初始建仓成本。样本内高成本对照使用 10 bps 费用和 10 bps 滑点。
Signal lag: 默认在 t 日收盘后使用截至该收盘的最近 756 个交易日日收益生成目标权重，在 t+1 交易日收盘模拟成交。t 到 t+1 的收盘到收盘收益由旧权重获得，新权重从成交后开始承担价格变动。
Price adjustment: yfinance auto_adjust total-return (adjusted close)
Survivorship bias: 使用论文事后选定的固定 15 只股票，README 载明存续偏差和样本选择偏差。整个回测区间早于 README 所列的论文年份 2026。样本外为历史时间切分后的模拟。
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.12590521716811784, ann_vol=0.1206173054262056, sharpe=1.0439757339024995, sharpe_excess=0.918055420473005, max_drawdown=-0.1290973590750284, total_return=0.4265963952623788, turnover=10.454075060066508, n_trades=7718
    oos: cagr=0.14361374264734428, ann_vol=0.16772848638773838, sharpe=0.8843263914851306, sharpe_excess=0.7406791256163456, max_drawdown=-0.260712219570037, total_return=0.953003519358427, turnover=18.367426635409704, n_trades=11528
    full: cagr=0.13693622988591797, ann_vol=0.1517433713145983, sharpe=0.922077932161652, sharpe_excess=0.7852879064655176, max_drawdown=-0.28861915539923955, total_return=1.7861477806514627, turnover=28.82150169547621, n_trades=19246
    benchmark full: cagr=0.1389741613729395, ann_vol=0.18426834021191482, sharpe=0.7989105894069222, sharpe_excess=0.6861672428934319, max_drawdown=-0.33717262056512987, total_return=1.8262718254222756, turnover=0.9990009990009991, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 487_conditional_value_at_risk_under_reward_p\nuniverse: [AAPL, NVDA, MSFT, JNJ, ABBV, LLY, BRK-B, JPM, V, GOOG, META, NFLX, WMT, COST, KO]\nbenchmark: SPY\nrisk_free_proxy: BIL\ndates:\n  prices_start: '2013-01-02'\n  is_start: '2017-03-01'\n  is_end: '2020-02-29'\n  oos_end: '2025-02-28'\nmodel:\n  lookback: 756\n  alpha: 0.90\n  kappa: 1.0\n  loss_threshold: 0.03\n  execution_lag: 1\n  cost_bps: 5.0\n  slippage_bps: 5.0\nrobustness:\n  kappa_lower: {kappa: 0.5}\n  kappa_higher: {kappa: 1.5}\n  threshold_lower: {loss_threshold: 0.025}\n  threshold_higher: {loss_threshold: 0.035}\n  higher_cost: {cost_bps: 10.0, slippage_bps: 10.0}\n"
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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Specification:
    lookback: int
    alpha: float
    kappa: float
    loss_threshold: float
    execution_lag: int
    cost_bps: float
    slippage_bps: float

    def __post_init__(self):
        if not isinstance(self.lookback, int) or self.lookback < 2:
            raise ValueError("lookback must be an integer of at least two sessions")
        if not isinstance(self.execution_lag, int) or self.execution_lag < 1:
            raise ValueError("execution_lag must be an integer of at least one session")
        if not (0 < self.alpha < 1 and np.isfinite(self.kappa) and self.kappa >= 0):
            raise ValueError("alpha must lie in (0, 1) and kappa must be finite and nonnegative")
        if not np.isfinite(self.loss_threshold):
            raise ValueError("loss_threshold must be finite")
        if not (self.cost_bps >= 0 and self.slippage_bps >= 0 and self.cost_rate < 1):
            raise ValueError("cost and slippage must be nonnegative and total less than 10000 bps")

    @property
    def cost_rate(self):
        return (self.cost_bps + self.slippage_bps) / 10_000


def check_prices(prices, start, end):
    if not isinstance(prices.index, pd.DatetimeIndex) or not prices.index.is_unique:
        raise ValueError("Expected unique session dates in the price history")
    if not prices.index.is_monotonic_increasing:
        raise ValueError("Expected session dates in increasing order")
    if prices.empty or prices.index[0] > start + pd.Timedelta(days=7):
        raise ValueError(f"Adjusted prices are missing near the requested start {start.date()}")
    if prices.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError(f"Adjusted prices are missing near the requested end {end.date()}")
    if prices.index[0] < start or prices.index[-1] > end:
        raise ValueError("Price history extends beyond the requested dates")
    missing = (~np.isfinite(prices) | prices.le(0)).any()
    if missing.any():
        raise ValueError(f"Missing or invalid adjusted close data for {', '.join(missing[missing].index)}")


def load_prices(symbols, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    identity = f"yfinance-auto-adjust-close-daily-v2|{','.join(symbols)}|{start.date()}|{end.date()}"
    cache_path = CACHE_DIR / f"prices_{sha256(identity.encode()).hexdigest()[:16]}.csv"
    if cache_path.exists():
        snapshot = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"])
        prices = snapshot.reindex(columns=symbols)
        check_prices(prices, start, end)
        return prices, snapshot["fetched_at"].iat[0]

    prices = pd.DataFrame(index=pd.DatetimeIndex([], name="Date"), columns=symbols, dtype=float)
    pending = symbols
    with TemporaryDirectory(dir=CACHE_DIR, prefix="yf_") as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        for attempt in range(2):
            data = yf.download(
                pending,
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                multi_level_index=True,
                threads=False,
                timeout=12,
                progress=False,
            )
            if data is not None and not data.empty:
                closes = data["Close"].astype(float).sort_index().loc[start:end]
                prices = closes.combine_first(prices).reindex(columns=symbols)
            try:
                check_prices(prices, start, end)
                break
            except ValueError as error:
                if attempt == 1:
                    raise RuntimeError(f"yfinance returned an incomplete price history: {error}") from error
                invalid = ~np.isfinite(prices) | prices.le(0)
                pending = [symbol for symbol in symbols if invalid[symbol].any()] or symbols

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with TemporaryDirectory(dir=CACHE_DIR, prefix="snapshot_") as scratch:
        staging = Path(scratch) / cache_path.name
        prices.assign(fetched_at=fetched_at).to_csv(staging, index_label="Date")
        staging.replace(cache_path)
    return prices, fetched_at


def balanced_cvar(mean_loss, volatility, alpha, kappa, threshold):
    """Theorem 3.1(b), scaled to the balanced loss of equation (4.3).

    Return the objective and its derivatives with respect to mean and volatility.
    """
    upper_slope = 1 + kappa
    tail = 1 - alpha
    normalizer = 2 * np.sqrt(alpha * tail)
    lower_cut = (1 / upper_slope + 2 * alpha - 1) / normalizer
    upper_cut = (upper_slope + 2 * alpha - 1) / normalizer
    gap = threshold - mean_loss
    tail_scale = np.sqrt(alpha / tail)
    standard_cvar = mean_loss + tail_scale * volatility
    if gap <= lower_cut * volatility:
        return upper_slope * standard_cvar - kappa * threshold, upper_slope, upper_slope * tail_scale
    if gap >= upper_cut * volatility:
        return standard_cvar, 1.0, tail_scale

    distance = np.hypot(volatility, gap)
    scale = np.sqrt(kappa**2 + 4 * alpha * upper_slope)
    slope_sum = 1 + upper_slope
    return (
        threshold + (scale * distance - slope_sum * gap) / (2 * tail),
        (slope_sum - scale * gap / distance) / (2 * tail),
        scale * volatility / (distance * 2 * tail),
    )


def target_weights(asset_returns, execution_days, spec, previous=None):
    size = asset_returns.shape[1]
    if spec.lookback <= size:
        raise ValueError("lookback must exceed the number of assets for full-rank sample covariance")
    signal_rows = asset_returns.index.get_indexer(execution_days) - spec.execution_lag
    if len(signal_rows) == 0 or signal_rows[0] < spec.lookback - 1:
        raise ValueError("Price history must cover the lookback and lag before the first execution")

    # Percent units keep the typical objective close to one for numerical optimization.
    losses = -100 * asset_returns.to_numpy()
    threshold = 100 * spec.loss_threshold
    ones = np.ones(size)
    budget = {"type": "eq", "fun": lambda weights: weights.sum() - 1, "jac": lambda weights: ones}
    bounds = [(0.0, 1.0)] * size
    weights = ones / size if previous is None else previous
    targets = []
    for signal_row in signal_rows:
        window = losses[signal_row - spec.lookback + 1 : signal_row + 1]
        mean = window.mean(axis=0)
        covariance = np.atleast_2d(np.cov(window, rowvar=False, ddof=1))
        signal_day = asset_returns.index[signal_row].date()
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(f"Loss covariance must be positive definite at {signal_day}") from error

        def objective(allocation):
            spread = covariance @ allocation
            volatility = np.sqrt(allocation @ spread)
            value, mean_slope, vol_slope = balanced_cvar(
                mean @ allocation, volatility, spec.alpha, spec.kappa, threshold
            )
            return value, mean_slope * mean + vol_slope * spread / volatility

        fit = minimize(
            objective, weights, jac=True, method="SLSQP", bounds=bounds,
            constraints=budget, options={"ftol": 1e-11, "maxiter": 100},
        )
        stationarity_gap = fit.x @ fit.jac - fit.jac.min()
        if not (fit.success and abs(fit.x.sum() - 1) < 1e-7 and stationarity_gap < 1e-4):
            raise RuntimeError(f"CVaR allocation failed at {signal_day}: {fit.message}, gap={stationarity_gap:g}")
        weights = fit.x.clip(0, 1)
        weights /= weights.sum()
        targets.append(weights)
    return np.asarray(targets)


def simulate(daily_returns, targets, cost_rate):
    weights = np.zeros(daily_returns.shape[1])
    records = []
    for asset_moves, target in zip(daily_returns.to_numpy(), targets, strict=True):
        # The old weights earn today's close-to-close return before any closing fill.
        marked = weights * (1 + asset_moves)
        gross_growth = 1 + weights @ asset_moves
        drifted = marked / gross_growth
        if target is None:
            weights = drifted
            records.append((gross_growth - 1, 0.0, 0))
            continue

        # The post-cost value and the traded notionals share the same pre-fill denominator.
        def cost_equation(remaining):
            return remaining + cost_rate * np.abs(remaining * target - drifted).sum() - 1

        remaining = brentq(cost_equation, 0.0, 1.0)
        traded = np.abs(remaining * target - drifted)
        records.append((gross_growth * remaining - 1, traded.sum(), np.count_nonzero(traded > TRADE_TOLERANCE)))
        weights = target

    return pd.DataFrame(records, index=daily_returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_proxy):
    returns = daily_results["return"].to_numpy()
    excess = returns - cash_proxy.loc[daily_results.index].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each evaluation period needs at least two sessions")
    stdev = returns.std(ddof=1)
    excess_stdev = excess.std(ddof=1)
    growth_curve = np.r_[1.0, np.cumprod(1 + returns)]
    annual_scale = SESSIONS_PER_YEAR / len(returns)
    return {
        "cagr": float(np.expm1(np.log1p(returns).sum() * annual_scale)),
        "ann_vol": float(stdev * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": float(returns.mean() / stdev * np.sqrt(SESSIONS_PER_YEAR)) if stdev else None,
        "sharpe_excess": float(excess.mean() / excess_stdev * np.sqrt(SESSIONS_PER_YEAR)) if excess_stdev else None,
        "max_drawdown": float((growth_curve / np.maximum.accumulate(growth_curve) - 1).min()),
        "total_return": float(growth_curve[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_report(daily_results, cash_proxy, is_end):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], cash_proxy),
        "oos": metrics(daily_results.loc[~in_sample], cash_proxy),
        "full": metrics(daily_results, cash_proxy),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    start, is_start, is_end, end = (dates[key] for key in ("prices_start", "is_start", "is_end", "oos_end"))
    if not start < is_start < is_end < end or end.date() >= datetime.now(timezone.utc).date():
        raise ValueError("Expected increasing dates ending before the current UTC date")
    assets, benchmark, cash_symbol = config["universe"], config["benchmark"], config["risk_free_proxy"]
    symbols = [*assets, benchmark, cash_symbol]
    if not assets or len(set(symbols)) != len(symbols) or cash_symbol != "BIL":
        raise ValueError("Expected distinct stock and benchmark symbols and BIL as the cash proxy")

    base = Specification(**config["model"])
    cases = {label: replace(base, **overrides) for label, overrides in config["robustness"].items()}
    if len({base, *cases.values()}) != len(cases) + 1:
        raise ValueError("Robustness cases must specify distinct alternatives")
    prices, fetched_at = load_prices(symbols, start, end)
    returns = prices.pct_change(fill_method=None).iloc[1:]
    asset_returns = returns[assets]
    cash_proxy = returns[cash_symbol]
    is_returns = asset_returns.loc[:is_end]
    is_moves = is_returns.loc[is_start:]
    oos_moves = asset_returns.loc[asset_returns.index > is_end]
    official_is_targets = target_weights(is_returns, is_moves.index, base)

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return (adjusted close)")
    print(f"DATES: prices from {prices.index[0].date()} | IS {is_moves.index[0].date()} to {is_moves.index[-1].date()}")
    print(f"OOS {oos_moves.index[0].date()} to {oos_moves.index[-1].date()} | excess Sharpe proxy {cash_symbol}")
    print("ROBUSTNESS: IS only")
    for label, spec in {"official": base, **cases}.items():
        if (spec.lookback, spec.alpha, spec.kappa, spec.loss_threshold, spec.execution_lag) == (
            base.lookback, base.alpha, base.kappa, base.loss_threshold, base.execution_lag
        ):
            targets = official_is_targets
        else:
            targets = target_weights(is_returns, is_moves.index, spec)
        case_report = metrics(simulate(is_moves, targets, spec.cost_rate), cash_proxy)
        print(f"{label} {json.dumps(config['robustness'].get(label, config['model']))} {json.dumps(case_report)}")
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")

    official_oos_targets = target_weights(asset_returns, oos_moves.index, base, previous=official_is_targets[-1])
    evaluation_moves = asset_returns.loc[is_start:]
    official_targets = np.concatenate((official_is_targets, official_oos_targets))
    daily_results = simulate(evaluation_moves, official_targets, base.cost_rate)
    benchmark_targets = [np.ones(1)] + [None] * (len(evaluation_moves) - 1)
    benchmark_daily = simulate(returns.loc[evaluation_moves.index, [benchmark]], benchmark_targets, base.cost_rate)
    official_report = period_report(daily_results, cash_proxy, is_end)
    benchmark_report = period_report(benchmark_daily, cash_proxy, is_end)
    for period in official_report:
        print(f"STRATEGY {period.upper()} {json.dumps(official_report[period])}")
        print(f"BENCHMARK {benchmark} {period.upper()} {json.dumps(benchmark_report[period])}")

    result = {
        "strategy": config["strategy"],
        "data_range": f"{evaluation_moves.index[0].date()} to {evaluation_moves.index[-1].date()}",
        **official_report,
        "benchmark": {"name": benchmark, **benchmark_report},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def time_limit_reached(_signal, _frame):
    raise SystemExit("Backtest stopped at the 14-minute runtime limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, time_limit_reached)
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)
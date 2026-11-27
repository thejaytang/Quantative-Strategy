"""459_entropic_value_at_risk_parity_for_temper
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/459_entropic_value_at_risk_parity_for_temper.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9e9ff3660187503321abd71fccfdc940b2ed36ad on 2026-11-27 08:24 Europe/Oslo, exit 0, output sha256 c86998277089771e9069ca33a5e405515ffaa547928ce677acb611b5eb69d3b7
Data source: Yahoo Finance 公开 ETF 日线，通过 yfinance 下载或读取本地价格缓存。信号资产池为 VTI、EFA、VWO、VNQ、AGG、TIP、GLD。BIL 仅作为计算超额夏普的收益基准。（config.yaml:2-3，strategy.py:41-68，strategy.py:184-196，strategy.py:232-246，README.md:22）
Data fetched: 2026-11-27T07:24:54.350408+00:00
Sample period: 请求行情区间为 2005-04-01 至 2026-03-31（含结束日），首笔成交前的历史用于估计。日志记录的回测区间为 2007-07-02 至 2026-03-31。（config.yaml:4-8，strategy.py:58-62，strategy.py:141-150，strategy.py:232-238，logs/run_20261127_080257.log:4）   data_range (from output): 2007-07-02 to 2026-03-31
IS/OOS split: 切分日为 2016-12-31，样本内取截至该日的交易日，样本外取之后的交易日。实际 IS 为 2007-07-02 至 2016-12-30，OOS 为 2017-01-03 至 2026-03-31。先连续模拟持仓，再按日期切分结果。各次权重估计使用截至当月信号日的滚动收益。（config.yaml:6，strategy.py:136-150，strategy.py:204-208，strategy.py:244-247，README.md:24）
Costs and slippage: 每笔买入或卖出金额计收 5 bps 交易成本及 5 bps 固定滑点，合计 10 bps。两项合并为比例费用，在成交日从净值扣除，按扣费后目标仓位与漂移后原仓位之差的绝对值计算成交额。首次建仓与月度等权基准适用同一规则。滑点以固定费用建模，成交价差与市场冲击未单独建模。（config.yaml:11-12，strategy.py:158-180，strategy.py:241-245，README.md:28，README.md:34）
Signal lag: 月末最后一个交易日 t 收盘后，使用截至该收盘的过去 12 个日历月日收益估计目标权重。目标在下一交易日 t+1 收盘成交。旧仓位承担截至成交收盘的收益，新仓位从成交收盘至下一收盘的区间起产生收益。（config.yaml:8-10，strategy.py:136-150，strategy.py:158-180，README.md:28）
Price adjustment: yfinance auto_adjust=True, split- and distribution-adjusted total-return close
Survivorship bias: 固定使用论文事后选定的七只现存 ETF，存在生存偏差及资产选择偏差。资产池由配置指定并受代码名单限制，未按历史日期重建资产池。（config.yaml:2，strategy.py:19，strategy.py:214-218，README.md:34）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.0448702529693652, ann_vol=0.07201000743019824, sharpe=0.6455375321247092, sharpe_excess=0.5860985607333957, max_drawdown=-0.2287465962060986, total_return=0.5173734090828195, turnover=0.8321773632781065, n_trades=798
    oos: cagr=0.061024653436337006, ann_vol=0.06981109370825683, sharpe=0.8835331098188087, sharpe_excess=0.5584000358435247, max_drawdown=-0.19116264073454736, total_return=0.7264100782492051, turnover=0.6517876850335517, n_trades=777
    full: cagr=0.0527949031704531, ann_vol=0.07092971914073194, sharpe=0.7608718779979594, sharpe_excess=0.5726826375686281, max_drawdown=-0.2287465962060986, total_return=1.6196087459079482, turnover=0.7433401314438686, n_trades=1575
    benchmark  full: cagr=0.06560825012170768, ann_vol=0.13174568258634706, sharpe=0.5483067432165585, sharpe_excess=0.4469682559213757, max_drawdown=-0.4010292728010473, total_return=2.2853254893384696, turnover=0.3685751569651743, n_trades=1575
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 459_ entropic_value_at_risk_parity_for_temper\nuniverse: [VTI, EFA, VWO, VNQ, AGG, TIP, GLD]\ncash_proxy: BIL\ndata_start: 2005-04-01\nbacktest_start: 2007-07-01\nis_end: 2016-12-31\ndata_end: 2026-03-31\nlookback_months: 12\ntail_probability: 0.05\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\n'
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
"""Research backtest for empirical EVaR-deviation equal risk contribution."""

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq, minimize
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
PAPER_ETFS = {"VTI", "EFA", "VWO", "VNQ", "AGG", "TIP", "GLD"}
SESSIONS_PER_YEAR = 252
FETCH_HEADER = "# DATA_FETCHED_AT: "


def validate_prices(prices, universe, cash_proxy, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Adjusted closes must have a nonempty, sorted, unique date index.")
    if not start <= prices.index[0] < start + pd.Timedelta(days=7):
        raise ValueError("Price history does not reach the requested start.")
    if not end - pd.Timedelta(days=7) < prices.index[-1] <= end:
        raise ValueError("Price history does not reach the requested end.")
    cash_start = prices[cash_proxy].first_valid_index()
    if cash_start is None:
        raise ValueError(f"No adjusted closes for {cash_proxy}.")
    risky = prices[universe].to_numpy()
    cash = prices.loc[cash_start:, cash_proxy].to_numpy()
    if not (np.isfinite(risky).all() and np.isfinite(cash).all()
            and (risky > 0).all() and (cash > 0).all()):
        raise ValueError("A required adjusted close is invalid or missing.")


def load_prices(universe, cash_proxy, start, end):
    tickers = sorted([*universe, cash_proxy])
    key = sha256(f"{tickers}|{start.date()}|{end.date()}".encode()).hexdigest()[:16]
    path = CACHE_DIR / f"yfinance_adjusted_close_{key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as cached:
            header = cached.readline().strip()
            if not header.startswith(FETCH_HEADER):
                raise ValueError(f"Missing fetch timestamp in {path.name}.")
            fetched_at = datetime.fromisoformat(header.removeprefix(FETCH_HEADER))
            prices = pd.read_csv(cached, index_col="Date", parse_dates=["Date"])
        validate_prices(prices, universe, cash_proxy, start, end)
        return prices, fetched_at

    CACHE_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE_DIR, prefix="yfinance_") as library_cache:
        yf.set_tz_cache_location(library_cache)
        downloaded = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, progress=False, timeout=15,
        )
    fetched_at = datetime.now(timezone.utc)
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices.")
    prices = downloaded["Close"].reindex(columns=tickers)
    prices.index = prices.index.tz_localize(None)
    validate_prices(prices, universe, cash_proxy, start, end)

    with NamedTemporaryFile(mode="w", dir=CACHE_DIR, encoding="utf-8", delete_on_close=False) as cached:
        cached.write(FETCH_HEADER + fetched_at.isoformat() + "\n")
        prices.to_csv(cached, index_label="Date")
        cached.flush()
        Path(cached.name).replace(path)
    return prices, fetched_at


def evar_and_gradient(weights, loss_scenarios, entropy_budget):
    losses = loss_scenarios @ weights
    worst = losses.max()
    relative_losses = losses - worst
    at_worst = relative_losses == 0
    log_count = np.log(len(losses))

    # When the entropy constraint admits all mass on the worst losses, EVaR is their maximum.
    if entropy_budget >= log_count - np.log(np.count_nonzero(at_worst)):
        return worst, loss_scenarios[at_worst].mean(axis=0)

    spread = -relative_losses.min()
    scaled_losses = relative_losses / spread

    def tilted_distribution(tilt):
        # The largest scaled loss is zero, so these exponential weights cannot overflow.
        weights = np.exp(tilt * scaled_losses)
        normalizer = weights.sum()
        return weights / normalizer, np.log(normalizer)

    def entropy_gap(tilt):
        probabilities, log_normalizer = tilted_distribution(tilt)
        return tilt * (probabilities @ scaled_losses) - log_normalizer + log_count - entropy_budget

    upper = 1.0
    while entropy_gap(upper) < 0:
        upper *= 2
    tilt = brentq(entropy_gap, 0, upper)
    probabilities, _ = tilted_distribution(tilt)
    return probabilities @ losses, probabilities @ loss_scenarios


def equal_risk_weights(returns, tail_probability):
    # Center each asset over the trailing window and use percentage units for optimization.
    losses = -100 * (returns - returns.mean(axis=0))
    entropy_budget = -np.log(tail_probability)
    budget = np.full(losses.shape[1], 1 / losses.shape[1])
    risk, _ = evar_and_gradient(budget, losses, entropy_budget)
    if risk <= 0:
        raise ValueError("Cannot allocate a positive risk budget to a riskless portfolio.")

    def objective(weights):
        risk, gradient = evar_and_gradient(weights, losses, entropy_budget)
        return risk - budget @ np.log(weights), gradient - budget / weights

    # Positive homogeneous risk plus the log barrier gives x_i dEVaR/dx_i = budget_i.
    solution = minimize(
        objective, budget / risk, jac=True, method="L-BFGS-B",
        bounds=[(1e-10, None)] * len(budget),
        options={"ftol": 1e-15, "gtol": 1e-9, "maxiter": 1000},
    )
    weights = solution.x / solution.x.sum()
    risk, gradient = evar_and_gradient(weights, losses, entropy_budget)
    if risk <= 0 or not np.allclose(weights * gradient / risk, budget, atol=1e-5, rtol=0):
        raise RuntimeError(f"EVaR risk budgets did not converge: {solution.message}")
    return weights


def rebalance_schedule(returns, start, lookback, tail_probability, lag):
    dates = returns.index
    first_price = dates[0]
    month_ends = dates.to_series().groupby(dates.to_period("M")).last()
    targets = {}
    for signal in month_ends:
        fill_index = dates.get_loc(signal) + lag
        if fill_index >= len(dates) or dates[fill_index] < start:
            continue
        cutoff = signal - pd.DateOffset(months=lookback)
        if first_price > cutoff:
            continue
        window = returns.iloc[dates.searchsorted(cutoff, side="right"):dates.get_loc(signal) + 1]
        try:
            targets[dates[fill_index]] = equal_risk_weights(window.to_numpy(), tail_probability)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(f"Rebalance signal on {signal.date()} failed: {error}") from error
    if not targets:
        raise ValueError("No rebalance has both a full estimation window and a fill session.")
    return targets


def simulate(returns, targets, fee_rate):
    weights = np.zeros(returns.shape[1])
    rows = []
    for date, asset_returns in zip(returns.index, returns.to_numpy()):
        # Old holdings earn the entire close-to-close move before the close fill.
        drifted = weights * (1 + asset_returns)
        gross_factor = 1 + weights @ asset_returns
        weights = drifted / gross_factor
        turnover, n_trades = 0.0, 0
        if date in targets:
            target = targets[date]
            # Target amounts use NAV after paying fees on both purchases and sales.
            def fee_balance(nav_fraction):
                traded = np.abs(nav_fraction * target - weights).sum()
                return nav_fraction + fee_rate * traded - 1

            nav_fraction = brentq(fee_balance, 0, 1)
            trades = np.abs(nav_fraction * target - weights)
            turnover = trades.sum()
            n_trades = np.count_nonzero(trades > 1e-10)
            gross_factor *= nav_fraction
            weights = target
        rows.append((gross_factor - 1, turnover, n_trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(path, cash_returns):
    returns = path["return"]
    nav = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    vol = returns.std(ddof=1)
    excess = returns - cash_returns.loc[returns.index]
    excess_vol = excess.std(ddof=1)
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(nav[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * annualizer),
        "sharpe": float(returns.mean() / vol * annualizer) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annualizer) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "total_return": float(nav[-1] - 1),
        "turnover": float(path["turnover"].sum() / years),
        "n_trades": int(path["n_trades"].sum()),
    }


def period_metrics(path, cash_returns, split):
    periods = {"is": path.loc[:split], "oos": path.loc[path.index > split], "full": path}
    if any(len(period) < 2 for period in periods.values()):
        raise ValueError("Both IS and OOS need at least two evaluated sessions.")
    return {name: metrics(period, cash_returns) for name, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    universe, cash_proxy = config["universe"], config["cash_proxy"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or not set(universe) <= PAPER_ETFS:
        raise ValueError("Choose at least two distinct ETFs from the paper's XASSET universe.")
    if cash_proxy != "BIL":
        raise ValueError("The excess-return reference is BIL.")
    data_start, start, split, end = (
        pd.Timestamp(config[field]) for field in ("data_start", "backtest_start", "is_end", "data_end")
    )
    if not data_start < start <= split < end:
        raise ValueError("Dates must be ordered as data start, backtest start, IS end, data end.")
    lag, lookback = config["signal_lag_sessions"], config["lookback_months"]
    tail = config["tail_probability"]
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    if type(lag) is not int or type(lookback) is not int or min(lag, lookback) < 1:
        raise ValueError("Execution lag and lookback must be positive integers.")
    if not (0 < tail < 1 and cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Invalid tail probability, transaction cost or slippage.")

    prices, fetched_at = load_prices(universe, cash_proxy, data_start, end)
    returns = prices.pct_change(fill_method=None)
    # Each solve has at most seven weights, so BLAS thread startup dominates its runtime.
    with threadpool_limits(limits=1, user_api="blas"):
        targets = rebalance_schedule(returns[universe], start, lookback, tail, lag)
    evaluated = returns.loc[min(targets):]
    cash_returns = evaluated[cash_proxy]
    if cash_returns.isna().any():
        raise ValueError("BIL needs an adjusted close on and before the first evaluated session.")
    equal = np.full(len(universe), 1 / len(universe))
    benchmark_targets = dict.fromkeys(targets, equal)
    fee_rate = (cost + slippage) / 10_000
    path = simulate(evaluated[universe], targets, fee_rate)
    benchmark = simulate(evaluated[universe], benchmark_targets, fee_rate)
    strategy_results = period_metrics(path, cash_returns, split)
    benchmark_results = period_metrics(benchmark, cash_returns, split)
    result = {
        "strategy": config["strategy"],
        "data_range": f"{path.index[0].date()} to {path.index[-1].date()}",
        **strategy_results,
        "benchmark": {"name": "Monthly equal weight", **benchmark_results},
    }
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted total-return close")
    print(f"PERFORMANCE_RANGE: {result['data_range']} | excess return vs {cash_proxy}")
    for name in ("is", "oos", "full"):
        print(f"{name.upper()}: {json.dumps(strategy_results[name], allow_nan=False)}")
        print(f"BENCHMARK {name.upper()}: {json.dumps(benchmark_results[name], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
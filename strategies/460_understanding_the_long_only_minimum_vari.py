"""460_understanding_the_long_only_minimum_vari
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/460_understanding_the_long_only_minimum_vari.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9e9ff3660187503321abd71fccfdc940b2ed36ad on 2026-11-27 08:24 Europe/Oslo, exit 0, output sha256 c7bda3672a0d517e61f4f6f919c75b59ccb1c5f23ff1fe4d06815eb3ab5db895
Data source: Yahoo Finance 经 yfinance 提供日频调整收盘价，覆盖行业 ETF XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY，以及市场因子 SPY 和现金代理 BIL。程序优先读取按标的和请求日期区间标识的本地数据缓存。（config.yaml:2-4，strategy.py:35-62，strategy.py:199-203）
Data fetched: 2026-11-27T07:24:56.613001+00:00
Sample period: 请求数据区间为 2008-01-01 至 2025-12-31（含末日），2008 年用于预热。配置的回测起点为 2009-01-01，日志记录实际回测交易日范围为 2009-01-02 至 2025-12-31。（config.yaml:5-8，strategy.py:54-57，strategy.py:199-202，README.md:18，logs/run_20261127_080341.log:4）   data_range (from output): 2009-01-02 to 2025-12-31
IS/OOS split: 样本内为回测起点至 2016-12-31（含切分日，按实际交易日），样本外为 2017-01-01 至 2025-12-31（按实际交易日）。程序对同一条连续持仓路径分段统计，切分处持仓延续。（config.yaml:6-8，strategy.py:171-175，strategy.py:202-209，README.md:18）
Costs and slippage: 每笔模拟买入或卖出均按成交名义金额计 5 bps 交易成本和 5 bps 固定滑点，共 10 bps。二者合并为费用率，在执行收盘按各 ETF 买卖名义金额绝对值之和扣减净值，目标名义金额按扣费后的净值计算。策略与等权基准采用相同费率。（config.yaml:11-12，strategy.py:124-145，strategy.py:195-207，README.md:25）
Signal lag: 每月首个交易日 t 收盘后，用截至该收盘的 126 个交易日的日收益计算目标权重。配置规定在下一交易日收盘执行。执行日的收盘到收盘收益按旧持仓计入，新权重从执行收盘后的收益区间起生效。（config.yaml:9-10，strategy.py:106-118，strategy.py:124-147，README.md:23）
Price adjustment: yfinance auto_adjust total-return close
Survivorship bias: 固定资产池由九只现仍存续的早期行业 ETF 构成。README 明确记录存在幸存者偏差，ETF 历史价格反映成分与指数规则随时间变化的结果。（config.yaml:2，strategy.py:181-186，strategy.py:199-205，README.md:29）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.1258909062414162, ann_vol=0.12510728567012686, sharpe=1.0106043580381645, sharpe_excess=1.0083464381767082, max_drawdown=-0.2009167059677186, total_return=1.5796559640025198, turnover=1.9766219865021115, n_trades=303
    oos: cagr=0.08704913050921093, ann_vol=0.14653761191487824, sharpe=0.6431540006092814, sharpe_excess=0.49048424566342114, max_drawdown=-0.29310915929115555, total_return=1.1153365888782543, turnover=1.9520432575983946, n_trades=529
    full: cagr=0.10517382226498317, ann_vol=0.1368506140871331, sharpe=0.7994607608086711, sharpe_excess=0.7123329512574906, max_drawdown=-0.29310915929115544, total_return=4.456840647372551, turnover=1.9636198619043082, n_trades=832
    benchmark  full: cagr=0.1321046238331547, ann_vol=0.17361384481675773, sharpe=0.8019176375867615, sharpe_excess=0.7332739922082688, max_drawdown=-0.3672402540222687, total_return=7.210335544445012, turnover=0.3439792988184191, n_trades=1836
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 460_ understanding_the_long_only_minimum_vari\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nmarket_factor: SPY\ncash_proxy: BIL\ndata_start: 2008-01-01\nbacktest_start: 2009-01-01\nis_end: 2016-12-31\ndata_end: 2025-12-31\nlookback_sessions: 126\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\n'
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
"""One-factor long-only minimum variance research backtest."""

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


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SECTOR_ETFS = {"XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLC", "XLRE"}
SESSIONS_PER_YEAR = 252
FETCH_HEADER = "# DATA_FETCHED_AT: "


def check_prices(prices, tickers, start, end):
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Daily closes must have a nonempty, increasing, unique date index.")
    week = pd.Timedelta(days=7)
    if not (start <= prices.index[0] < start + week and end - week < prices.index[-1] <= end):
        raise ValueError("Daily prices do not cover the requested date range.")
    for ticker in tickers:
        values = prices[ticker].to_numpy()
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError(f"Missing or invalid adjusted close in {ticker}.")


def load_prices(tickers, start, end):
    tickers = sorted(tickers)
    key = sha256(f"{tickers}|{start.date()}|{end.date()}".encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"yfinance_adjusted_close_{key}.csv"
    if cache_file.exists():
        with cache_file.open(encoding="utf-8") as handle:
            stamp = handle.readline().strip()
            if not stamp.startswith(FETCH_HEADER):
                raise ValueError(f"Missing fetch timestamp in {cache_file.name}.")
            fetched_at = datetime.fromisoformat(stamp.removeprefix(FETCH_HEADER))
            prices = pd.read_csv(
                handle, index_col="Date", parse_dates=["Date"], converters=dict.fromkeys(tickers, float),
            )
        check_prices(prices, tickers, start, end)
        return prices, fetched_at

    CACHE_DIR.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE_DIR, prefix="yfinance_") as library_cache:
        yf.set_tz_cache_location(library_cache)
        raw = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, progress=False, timeout=15,
        )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no daily prices.")
    fetched_at = datetime.now(timezone.utc)
    prices = raw["Close"].reindex(columns=tickers)
    prices.index = prices.index.tz_localize(None)
    check_prices(prices, tickers, start, end)

    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete_on_close=False) as handle:
        handle.write(FETCH_HEADER + fetched_at.isoformat() + "\n")
        prices.to_csv(handle, index_label="Date")
        handle.flush()
        Path(handle.name).replace(cache_file)
    return prices, fetched_at


def minimum_variance_weights(asset_excess, market_excess):
    assets = asset_excess - asset_excess.mean(axis=0)
    market = market_excess - market_excess.mean()
    market_variance = (market @ market) / len(market)
    if market_variance <= 0:
        raise ValueError("The market factor requires positive variance.")
    beta = (market @ assets) / (len(market) * market_variance)
    residual = assets - np.outer(market, beta)
    residual_variance = np.mean(residual ** 2, axis=0)
    variance_scale = max(market_variance, np.mean(assets ** 2, axis=0).max())
    precision = 1 / np.maximum(residual_variance, np.finfo(float).eps * variance_scale)

    # Theorem 2 orients the betas so their precision-weighted sum is positive.
    if beta @ precision < 0:
        beta = -beta
    exposure_sum = exposure_squared_sum = 0.0
    active = []
    for i in np.argsort(beta):
        inclusion = 1 + market_variance * (exposure_squared_sum - beta[i] * exposure_sum)
        if inclusion <= 0:
            break
        active.append(i)
        exposure_sum += beta[i] * precision[i]
        exposure_squared_sum += beta[i] ** 2 * precision[i]

    # Woodbury formula for the inverse covariance solution on the active set.
    h = market_variance * exposure_sum / (1 + market_variance * exposure_squared_sum)
    weights = np.zeros_like(beta)
    weights[active] = precision[active] * np.maximum(1 - beta[active] * h, 0)
    return weights / weights.sum()


def rebalance_targets(returns, universe, market, cash, start, lookback, lag):
    dates = returns.index
    months = dates.to_period("M")
    excess = returns[[*universe, market]].sub(returns[cash], axis=0)
    targets = {}
    for signal in np.flatnonzero(months[1:] != months[:-1]) + 1:
        execution = signal + lag
        if signal < lookback or execution >= len(dates) or dates[execution] < start:
            continue
        window = excess.iloc[signal - lookback + 1:signal + 1]
        targets[dates[execution]] = minimum_variance_weights(
            window[universe].to_numpy(), window[market].to_numpy(),
        )
    if not targets:
        raise ValueError("No execution dates with a complete trailing return window.")
    return targets


def simulate(returns, cash_returns, targets, fee_rate):
    weights = np.zeros(returns.shape[1])
    rows = []
    for date, asset_returns, cash_return in zip(returns.index, returns.to_numpy(), cash_returns):
        marked = weights * (1 + asset_returns)
        gross_nav = marked.sum() + (1 - weights.sum()) * (1 + cash_return)
        weights = marked / gross_nav
        turnover, n_trades = 0.0, 0
        if date in targets:
            target = targets[date]

            def self_financing(nav_fraction):
                traded = np.abs(nav_fraction * target - weights).sum()
                return nav_fraction + fee_rate * traded - 1

            # Size orders against post-fee NAV and mark new weights after this close.
            nav_fraction = brentq(self_financing, 0, 1)
            traded = np.abs(nav_fraction * target - weights)
            turnover = traded.sum()
            n_trades = np.count_nonzero(traded > 1e-10)
            gross_nav *= nav_fraction
            weights = target

        rows.append((gross_nav - 1, turnover, n_trades))
    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(path, cash_returns):
    returns = path["return"]
    nav = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess = returns - cash_returns.loc[returns.index]
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(nav[-1] ** (1 / years) - 1),
        "ann_vol": float(vol * scale),
        "sharpe": float(returns.mean() / vol * scale) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * scale) if excess_vol > 0 else None,
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "total_return": float(nav[-1] - 1),
        "turnover": float(path["turnover"].sum() / years),
        "n_trades": int(path["n_trades"].sum()),
    }


def period_metrics(path, cash_returns, split):
    periods = {"is": path.loc[:split], "oos": path.loc[path.index > split], "full": path}
    if any(len(period) < 2 for period in periods.values()):
        raise ValueError("IS and OOS each require at least two sessions.")
    return {name: metrics(period, cash_returns) for name, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    market, cash = config["market_factor"], config["cash_proxy"]
    if len(set(universe)) != len(universe) or len(universe) < 2 or not set(universe) <= SECTOR_ETFS:
        raise ValueError("Choose at least two distinct sector ETFs from the supported list.")
    if (market, cash) != ("SPY", "BIL"):
        raise ValueError("This specification uses SPY for market returns and BIL for cash returns.")
    data_start, start, split, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "backtest_start", "is_end", "data_end")
    )
    if not data_start < start <= split < end:
        raise ValueError("Dates must order the warm-up, IS and OOS periods.")
    lookback, lag = config["lookback_sessions"], config["signal_lag_sessions"]
    if type(lookback) is not int or lookback < 3 or type(lag) is not int or lag < 1:
        raise ValueError("Lookback must be at least three sessions and execution lag at least one.")
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Cost and slippage must be nonnegative and sum to less than 10,000 bps.")

    prices, fetched_at = load_prices([*universe, market, cash], data_start, end)
    returns = prices.pct_change(fill_method=None)
    targets = rebalance_targets(returns, universe, market, cash, start, lookback, lag)
    evaluated = returns.loc[start:]
    cash_returns = evaluated[cash]
    fee_rate = (cost + slippage) / 10_000
    path = simulate(evaluated[universe], cash_returns, targets, fee_rate)
    equal_weight = np.full(len(universe), 1 / len(universe))
    benchmark = simulate(evaluated[universe], cash_returns, dict.fromkeys(targets, equal_weight), fee_rate)
    strategy_results = period_metrics(path, cash_returns, split)
    benchmark_results = period_metrics(benchmark, cash_returns, split)
    result = {
        "strategy": config["strategy"],
        "data_range": f"{path.index[0].date()} to {path.index[-1].date()}",
        **strategy_results,
        "benchmark": {"name": "Monthly equal-weight sector ETFs", **benchmark_results},
    }
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close")
    print(f"PERFORMANCE_RANGE: {result['data_range']} | excess returns vs {cash}")
    for name in ("is", "oos", "full"):
        print(f"{name.upper()}: {json.dumps(strategy_results[name], allow_nan=False)}")
        print(f"BENCHMARK {name.upper()}: {json.dumps(benchmark_results[name], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
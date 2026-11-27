"""468_cleaning_correlation_matrices_with_exter
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/468_cleaning_correlation_matrices_with_exter.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e0683ac343a175cb2458afc860f8d1a1066c8e3a on 2026-11-27 08:55 Europe/Oslo, exit 0, output sha256 2ff295a57ac4d890af0ee0efb50493e5180a836c1b45f79f6ce524e2e000244d
Data source: yfinance 免费日线收盘价。股票代理 SPY、IWM、EFA、EEM，债券代理 SHY、IEI、IEF、TLT，外汇代理 FXA、FXB、FXE、FXY。BIL 的日回报用于计算超额夏普比率（config.yaml:2-7，README.md:11，strategy.py:21-38、171-185、232-242）。
Data fetched: 2026-11-27T07:55:54+00:00
Sample period: 请求的价格区间为 2008-01-01 至 2025-12-31（含端点）。正式回测记录覆盖 2011-01-03 至 2025-12-31，此前数据用于预热，配置要求至少 150 周观测（config.yaml:9-18，strategy.py:29-34、103-132、237-242，README.md:13、18，logs/run_20261127_080714.log:10）。   data_range (from output): ['2011-01-03', '2025-12-31']
IS/OOS split: 样本内为 2011-01-03 至 2019-12-31（含），样本外为 2020 年首个交易日至 2025-12-31。两段来自同一连续模拟，切分时延续持仓和滚动估计状态（config.yaml:9-13，README.md:13、23，strategy.py:193-198、237-242）。
Costs and slippage: 每次模拟调仓对买卖金额计收 2 bps 费用及 3 bps 固定滑点，合计 5 bps。先按旧持仓记账并计算调仓前净值，再以费后净值定义目标仓位，按各标的绝对成交额之和扣减净值。现金余额不计息，未计入借券费（config.yaml:20-23，strategy.py:139-166，README.md:17、19）。
Signal lag: 信号使用截至该周最后一个交易日收盘的日线数据。默认在该周周五之后的首个交易日收盘模拟调仓。执行日截至收盘的收益仍计入旧持仓，新仓位从该日收盘之后开始承担收益（config.yaml:20-21，README.md:19，strategy.py:103-132、139-168、207-212）。
Price adjustment: yfinance auto_adjust=True, split- and distribution-adjusted close
Survivorship bias: 固定代理池为事后选定，存在存续偏差。已清盘、退市或未纳入的产品未进入样本。代码使用配置中的固定名单，不按历史时点重建标的池（config.yaml:2-7，strategy.py:217-232，README.md:23）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.026120866095915574, ann_vol=0.0183802981786506, sharpe=1.4121482289883187, sharpe_excess=1.153153726132971, max_drawdown=-0.02588067115983539, total_return=0.26069176843205866, turnover=0.6526319163347633, n_trades=5640
    oos: cagr=0.01606944807465438, ann_vol=0.02903832554161295, sharpe=0.5635238125786001, sharpe_excess=-0.3367566824930223, max_drawdown=-0.06957666072913138, total_return=0.10009568729379259, turnover=0.6977312203138128, n_trades=3756
    full: cagr=0.022090555555318137, ann_vol=0.023233625903739702, sharpe=0.9521102265054782, sharpe_excess=0.3809601980103657, max_drawdown=-0.06957666072913227, total_return=0.38688157745889185, turnover=0.6706620728566102, n_trades=9396
    benchmark  full: cagr=0.011962034207489536, ann_vol=0.013764118057422179, sharpe=0.8708190127142189, sharpe_excess=-0.08986072461495441, max_drawdown=-0.053961768519458886, total_return=0.19481137119179004, turnover=1.924337930548674, n_trades=9396
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '468_ cleaning_correlation_matrices_with_exter'\nuniverse:\n  equity: [SPY, IWM, EFA, EEM]\n  bonds: [SHY, IEI, IEF, TLT]\n  currency: [FXA, FXB, FXE, FXY]\nexpected_premium_groups: [equity, bonds]\nrisk_free_ticker: BIL\n\ndates:\n  data_start: '2008-01-01'\n  report_start: '2011-01-03'\n  is_end: '2019-12-31'\n  report_end: '2025-12-31'\n\nmodel:\n  volatility_decay_days: 100\n  correlation_decay_days: 750\n  warmup_weeks: 150\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\n"
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
"""Research backtest of Valeyre's MaxVar correlation estimator on ETF proxies."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
TRADE_TOLERANCE = 1e-10


def load_prices(tickers, start, end):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    request = json.dumps([tickers, start.date().isoformat(), end.date().isoformat(), True])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = cache / f"adjusted_closes_{key}.csv"
    if not path.is_file():
        yf.set_tz_cache_location(str(cache))
        data = yf.download(
            tickers,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=15,
        )
        prices = data["Close"].reindex(columns=tickers).sort_index()
        check_prices(prices, tickers, start, end)
        with NamedTemporaryFile(mode="w", dir=cache, suffix=".tmp", delete=False) as temp:
            temp_path = Path(temp.name)
            try:
                prices.to_csv(temp, index_label="date")
                temp.close()
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)

    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    check_prices(prices, tickers, start, end)
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    return prices, fetched_at.isoformat(timespec="seconds")


def check_prices(prices, tickers, start, end):
    if prices.empty or list(prices.columns) != tickers:
        raise ValueError("Price panel is empty or has unexpected ticker columns")
    index = prices.index
    if (
        not index.is_monotonic_increasing
        or index.has_duplicates
        or not start <= index[0] <= start + pd.Timedelta(days=7)
        or not end - pd.Timedelta(days=7) <= index[-1] <= end
    ):
        raise ValueError("Price panel has unexpected dates or incomplete date coverage")
    bad_prices = ~np.isfinite(prices) | (prices <= 0)
    if bad_prices.any().any():
        missing = bad_prices.sum()
        raise ValueError(f"Missing or invalid closes by ticker: {missing[missing > 0].to_dict()}")


def as_correlation(matrix):
    scale = np.sqrt(np.diag(matrix))
    if np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("Cannot normalize a matrix with invalid diagonal entries")
    return matrix / np.outer(scale, scale)


def maxvar(correlation, groups):
    """Paper equations (1)–(7), using the orthonormal basis for disjoint groups."""
    n_assets = len(correlation)
    basis = np.zeros((n_assets, len(groups)))
    for column, indices in enumerate(groups):
        _, eigenvectors = np.linalg.eigh(correlation[np.ix_(indices, indices)])
        basis[indices, column] = eigenvectors[:, -1]

    # Reconstruct the restricted spectrum directly in the group basis.
    projected = basis.T @ correlation @ basis
    delta = (n_assets - np.trace(projected)) / (n_assets - len(groups))
    if delta <= 0:
        raise ValueError("MaxVar residual variance is nonpositive")
    completed = basis @ projected @ basis.T + delta * (np.eye(n_assets) - basis @ basis.T)
    return as_correlation(completed)


def portfolio_weights(correlation, volatility, premium):
    if np.any(volatility <= 0):
        raise ValueError("Portfolio weights require positive volatility for every asset")
    direction = np.linalg.solve(correlation, premium) / volatility
    return direction / np.abs(direction).sum()


def build_targets(asset_returns, groups, premium, model, lag, report_start):
    index = asset_returns.index
    variance = asset_returns.pow(2).ewm(alpha=1 / model["volatility_decay_days"], adjust=False).mean()
    volatility = np.sqrt(variance)
    weeks = index.to_period("W-FRI")
    standardized = asset_returns.div(volatility.where(volatility > 0)).fillna(0.0)
    weekly_returns = standardized.groupby(weeks).sum().iloc[1:]
    weekly_volatility = volatility.groupby(weeks).last()
    alpha = 1 - (1 - 1 / model["correlation_decay_days"]) ** 5

    cleaned_targets = {}
    raw_targets = {}
    moment = None
    for count, (week, weekly_return) in enumerate(weekly_returns.iterrows(), start=1):
        close_position = index.searchsorted(week.end_time.normalize(), side="right") + lag - 1
        if close_position >= len(index):
            break
        observation = np.outer(weekly_return, weekly_return)
        if moment is None:
            moment = observation
        else:
            moment = (1 - alpha) * moment + alpha * observation
        execution_date = index[close_position]
        if count < model["warmup_weeks"] or execution_date < report_start:
            continue

        empirical = as_correlation(moment)
        signal_volatility = weekly_volatility.loc[week].to_numpy()
        cleaned_targets[execution_date] = portfolio_weights(maxvar(empirical, groups), signal_volatility, premium)
        raw_targets[execution_date] = portfolio_weights(empirical, signal_volatility, premium)

    if not cleaned_targets:
        raise ValueError("No dated targets after warmup; check history, lag and report dates")
    return cleaned_targets, raw_targets


def simulate(asset_returns, targets, cost_bps, slippage_bps):
    """Mark existing positions through the close, then fill any scheduled target."""
    fee = (cost_bps + slippage_bps) / 10_000
    weights = np.zeros(asset_returns.shape[1])
    rows = []
    for date, returns in zip(asset_returns.index, asset_returns.to_numpy()):
        gross_growth = 1 + weights @ returns
        if gross_growth <= 0:
            raise ValueError(f"Portfolio NAV reached zero on {date.date()}")
        weights = weights * (1 + returns) / gross_growth
        turnover = 0.0
        n_trades = 0
        net_fraction = 1.0
        if date in targets:
            target = targets[date]
            if fee:
                # Target weights are fractions of post-fee NAV. Solve for that NAV
                # before measuring the dollars traded and deducting the fee.
                def cash_balance(fraction):
                    return fraction + fee * np.abs(fraction * target - weights).sum() - 1

                net_fraction = brentq(cash_balance, 0, 1)
            traded = np.abs(net_fraction * target - weights)
            turnover = traded.sum()
            n_trades = np.count_nonzero(traded > TRADE_TOLERANCE)
            weights = target

        rows.append((gross_growth * net_fraction - 1, turnover, n_trades))

    return pd.DataFrame(rows, index=asset_returns.index, columns=["return", "turnover", "n_trades"])


def metrics(history, risk_free):
    returns = history["return"]
    if len(returns) < 2:
        raise ValueError("Each metrics period needs at least two return observations")
    nav = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    ann_factor = np.sqrt(SESSIONS_PER_YEAR)
    daily_vol = returns.std()
    excess = returns - risk_free.loc[history.index]
    excess_vol = excess.std()
    return {
        "cagr": float(nav[-1] ** (1 / years) - 1),
        "ann_vol": float(daily_vol * ann_factor),
        "sharpe": float(returns.mean() / daily_vol * ann_factor) if daily_vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * ann_factor) if excess_vol > 0 else None,
        "max_drawdown": float((nav / np.maximum.accumulate(nav) - 1).min()),
        "total_return": float(nav[-1] - 1),
        "turnover": float(history["turnover"].sum() / years),
        "n_trades": int(history["n_trades"].sum()),
    }


def summarize(history, risk_free, is_end):
    return {
        "is": metrics(history.loc[history.index <= is_end], risk_free),
        "oos": metrics(history.loc[history.index > is_end], risk_free),
        "full": metrics(history, risk_free),
    }


def main():
    with (ROOT / "config.yaml").open() as config_file:
        config = yaml.safe_load(config_file)
    dates = {name: pd.Timestamp(value) for name, value in config["dates"].items()}
    if not dates["data_start"] < dates["report_start"] < dates["is_end"] < dates["report_end"]:
        raise ValueError("Expected data_start < report_start < is_end < report_end")
    execution = config["execution"]
    lag = execution["lag_sessions"]
    cost = execution["cost_bps"]
    slippage = execution["slippage_bps"]
    if type(lag) is not int or lag < 1 or not (cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Use an integer lag of at least one session and valid friction rates")
    model = config["model"]
    if min(model["volatility_decay_days"], model["correlation_decay_days"]) <= 1 or model["warmup_weeks"] < 2:
        raise ValueError("Use decay lengths above one day and at least two warmup weeks")

    universe = config["universe"]
    tickers = [ticker for group in universe.values() for ticker in group]
    risk_free_ticker = config["risk_free_ticker"]
    if (
        any(not group for group in universe.values())
        or not 0 < len(universe) < len(tickers)
        or len(set(tickers + [risk_free_ticker])) != len(tickers) + 1
    ):
        raise ValueError("Use nonempty disjoint groups and a separate risk-free ticker")
    groups = [[tickers.index(ticker) for ticker in group] for group in universe.values()]
    premium_groups = set(config["expected_premium_groups"])
    if not premium_groups or not premium_groups.issubset(universe):
        raise ValueError("Expected premium groups must name at least one configured group")
    premium = np.array([float(name in premium_groups) for name, group in universe.items() for _ in group])

    prices, fetched_at = load_prices(tickers + [risk_free_ticker], dates["data_start"], dates["report_end"])
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted close")
    print("PARAM_COMBOS_TRIED: 1")

    returns = prices.pct_change(fill_method=None).iloc[1:]
    targets, benchmark_targets = build_targets(returns[tickers], groups, premium, model, lag, dates["report_start"])
    asset_returns = returns.loc[dates["report_start"]:, tickers]
    strategy = summarize(simulate(asset_returns, targets, cost, slippage), returns[risk_free_ticker], dates["is_end"])
    benchmark = summarize(
        simulate(asset_returns, benchmark_targets, cost, slippage), returns[risk_free_ticker], dates["is_end"]
    )
    benchmark_name = "Uncleaned correlation, same portfolio rule"
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")

    results = {
        "strategy": config["strategy"],
        "data_range": [asset_returns.index[0].date().isoformat(), asset_returns.index[-1].date().isoformat()],
        **strategy,
        "benchmark": {"name": benchmark_name, **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()
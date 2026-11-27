"""468_cleaning_correlation_matrices_with_exter
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/468_cleaning_correlation_matrices_with_exter.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c71364ad99535bbf59be998b004cff431ee83048 on 2026-11-27 11:54 Europe/Oslo, exit 0, output sha256 cc824156fe61b790342e2f5aecc74e664f0f5ecc56f365641f696944fa9a5ced
Data source: yfinance 免费日线收盘价。股票代理 SPY、IWM、EFA、EEM，债券代理 SHY、IEI、IEF、TLT，外汇代理 FXA、FXB、FXE、FXY。BIL 的日回报用于计算超额夏普比率（config.yaml:2-7，README.md:11，strategy.py:21-38、171-185、232-242）。
Data fetched: 2026-11-27T10:54:25+00:00
Sample period: 请求的价格区间为 2008-01-01 至 2025-12-31（含端点）。正式回测记录覆盖 2011-01-03 至 2025-12-31，此前数据用于预热，配置要求至少 150 周观测（config.yaml:9-18，strategy.py:29-34、103-132、237-242，README.md:13、18，logs/run_20261127_080714.log:10）。   data_range (from output): ['2011-01-03', '2025-12-31']
IS/OOS split: 样本内为 2011-01-03 至 2019-12-31（含），样本外为 2020 年首个交易日至 2025-12-31。两段来自同一连续模拟，切分时延续持仓和滚动估计状态（config.yaml:9-13，README.md:13、23，strategy.py:193-198、237-242）。
Costs and slippage: 每次模拟调仓对买卖金额计收 2 bps 费用及 3 bps 固定滑点，合计 5 bps。先按旧持仓记账并计算调仓前净值，再以费后净值定义目标仓位，按各标的绝对成交额之和扣减净值。现金余额不计息，未计入借券费（config.yaml:20-23，strategy.py:139-166，README.md:17、19）。
Signal lag: 信号使用截至该周最后一个交易日收盘的日线数据。默认在该周周五之后的首个交易日收盘模拟调仓。执行日截至收盘的收益仍计入旧持仓，新仓位从该日收盘之后开始承担收益（config.yaml:20-21，README.md:19，strategy.py:103-132、139-168、207-212）。
Price adjustment: yfinance auto_adjust=True, split- and distribution-adjusted close
Survivorship bias: 固定代理池为事后选定，存在存续偏差。已清盘、退市或未纳入的产品未进入样本。代码使用配置中的固定名单，不按历史时点重建标的池（config.yaml:2-7，strategy.py:217-232，README.md:23）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.023983124741544515, ann_vol=0.016415927183119572, sharpe=1.4519950437485794, sharpe_excess=1.1613195240701875, max_drawdown=-0.023973742293136202, total_return=0.2372908974358945, turnover=0.6113980650677456, n_trades=5640
    oos: cagr=0.015695698670614266, ann_vol=0.027340822454115586, sharpe=0.5833021509235745, sharpe_excess=-0.372758571713634, max_drawdown=-0.0658289899259995, total_return=0.09767638418672564, turnover=0.6650979226959388, n_trades=3756
    full: cagr=0.02066182974776254, ann_vol=0.02145945371661404, sharpe=0.9637880055335821, sharpe_excess=0.34546616597561913, max_drawdown=-0.06582898992600106, total_return=0.35814499848457615, turnover=0.6328666189657615, n_trades=9396
    benchmark  full: cagr=0.011217058569584548, ann_vol=0.011415666306326547, sharpe=0.98286093072338, sharpe_excess=-0.1739612366665184, max_drawdown=-0.04680027185613478, total_return=0.18171301540847806, turnover=1.665104063528596, n_trades=9396
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: '468_cleaning_correlation_matrices_with_exter'\nuniverse:\n  equity: [SPY, IWM, EFA, EEM]\n  bonds: [SHY, IEI, IEF, TLT]\n  currency: [FXA, FXB, FXE, FXY]\nexpected_premium_groups: [equity, bonds]\nrisk_free_ticker: BIL\n\nlegacy_snapshot:\n  sha256: '05456ebbecf590957a36b037306f803f0d4fe677500d52ee3b3915541bb0093d'\n  logged_at: '2026-11-27T07:05:18+00:00'\n\ndates:\n  data_start: '2008-01-01'\n  report_start: '2011-01-03'\n  is_end: '2019-12-31'\n  report_end: '2025-12-31'\n\nmodel:\n  volatility_decay_days: 100\n  correlation_decay_days: 750\n  warmup_weeks: 150\n\nexecution:\n  lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\n  cash_buffer: 0.05\n"
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
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)


ROOT = Path(__file__).resolve().parent
SESSIONS_PER_YEAR = 252
MAX_CORRELATION_CONDITION = 1e8
FETCH_TIME_PREFIX = b"# DATA_FETCHED_AT: "
NYSE_SPECIAL_CLOSURES = pd.DatetimeIndex(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


def load_prices(tickers, start, end, legacy_snapshot):
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    request = json.dumps([tickers, start.date().isoformat(), end.date().isoformat(), True])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    path = cache / f"adjusted_closes_{key}.csv"
    if not path.is_file():
        yf.set_tz_cache_location(str(cache / "yfinance"))
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
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        check_prices(prices, tickers, start, end)
        temp_path = None
        try:
            with NamedTemporaryFile(mode="w", dir=cache, suffix=".tmp", delete=False) as temp:
                temp_path = Path(temp.name)
                temp.write(FETCH_TIME_PREFIX.decode() + fetched_at + "\n")
                prices.to_csv(temp, index_label="date")
            temp_path.replace(path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    with path.open("rb") as snapshot:
        header = snapshot.readline()
        if header.startswith(FETCH_TIME_PREFIX):
            fetched_at = datetime.fromisoformat(header[len(FETCH_TIME_PREFIX):].decode().strip())
            if fetched_at.utcoffset() is None:
                raise ValueError("Cached fetch time must specify a timezone")
            fetched_at = fetched_at.astimezone(timezone.utc).isoformat(timespec="seconds")
        else:
            digest = hashlib.sha256(header + snapshot.read()).hexdigest()
            if digest != legacy_snapshot["sha256"]:
                raise ValueError(f"Cached prices have no fetch time and do not match the original snapshot: {path}")
            fetched_at = legacy_snapshot["logged_at"]
            snapshot.seek(0)
        prices = pd.read_csv(snapshot, index_col=0, parse_dates=True)
    check_prices(prices, tickers, start, end)
    return prices, fetched_at


def check_prices(prices, tickers, start, end):
    if prices.empty or list(prices.columns) != tickers:
        raise ValueError("Price panel is empty or has unexpected ticker columns")
    if start.year < 2008 or end.year > 2025:
        raise ValueError("NYSE historical session coverage for this backtest is 2008–2025")
    holidays = NyseHolidays().holidays(start, end).union(NYSE_SPECIAL_CLOSURES)
    expected_sessions = pd.bdate_range(start, end).difference(holidays)
    if not prices.index.equals(expected_sessions):
        missing = expected_sessions.difference(prices.index).strftime("%Y-%m-%d").tolist()
        extra = prices.index.difference(expected_sessions).tolist()
        raise ValueError(
            f"Price dates differ from NYSE sessions (missing: {missing}, extra: {extra}, or unordered/duplicate dates)"
        )
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


def well_conditioned(correlation):
    eigenvalues = np.linalg.eigvalsh(correlation)
    return eigenvalues[0] > 0 and eigenvalues[-1] <= MAX_CORRELATION_CONDITION * eigenvalues[0]


def build_targets(asset_returns, groups, premium, model, lag, report_start):
    index = asset_returns.index
    variance = asset_returns.pow(2).ewm(alpha=1 / model["volatility_decay_days"], adjust=False).mean()
    volatility = np.sqrt(variance)
    weeks = index.to_period("W-FRI")
    standardized = asset_returns.div(volatility.where(volatility > 0)).fillna(0.0)
    weekly_returns = standardized.groupby(weeks).sum().iloc[1:]
    weekly_volatility = volatility.groupby(weeks).last()
    weekly_closes = index.to_series().groupby(weeks).last()
    alpha = 1 - (1 - 1 / model["correlation_decay_days"]) ** 5

    cleaned_targets = {}
    raw_targets = {}
    skipped_rebalances = 0
    moment = np.zeros((asset_returns.shape[1], asset_returns.shape[1]))
    for count, (week, weekly_return) in enumerate(weekly_returns.iterrows(), start=1):
        close_position = index.searchsorted(weekly_closes.loc[week], side="right") + lag - 1
        if close_position >= len(index):
            break
        observation = np.outer(weekly_return, weekly_return)
        moment = (1 - alpha) * moment + alpha * observation
        execution_date = index[close_position]
        if count < model["warmup_weeks"] or execution_date < report_start:
            continue

        signal_volatility = weekly_volatility.loc[week].to_numpy()
        if np.any(np.diag(moment) <= 0) or np.any(signal_volatility <= 0):
            skipped_rebalances += 1
            continue
        empirical = as_correlation(moment)
        if not well_conditioned(empirical):
            skipped_rebalances += 1
            continue
        cleaned = maxvar(empirical, groups)
        if not well_conditioned(cleaned):
            skipped_rebalances += 1
            continue
        cleaned_targets[execution_date] = portfolio_weights(cleaned, signal_volatility, premium)
        raw_targets[execution_date] = portfolio_weights(empirical, signal_volatility, premium)

    return cleaned_targets, raw_targets, skipped_rebalances


def simulate(asset_prices, targets, report_start, execution):
    """Fix adjusted-price units at the prior close, then settle them at the next close."""
    fee = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    gross_fraction = 1 - execution["cash_buffer"]
    first_session = asset_prices.index.searchsorted(report_start)
    if not 0 < first_session < len(asset_prices):
        raise ValueError("Need a prior close and at least one report session")
    dates = asset_prices.index[first_session:]
    closes = asset_prices.to_numpy()
    positions = np.zeros(asset_prices.shape[1])
    cash = previous_value = 1.0
    rows = []
    for session, date in enumerate(dates, start=first_session):
        turnover = 0.0
        n_trades = 0
        target = targets.get(date)
        close = closes[session]
        if target is not None:
            sizing_close = closes[session - 1]
            target_positions = target * (gross_fraction * previous_value) / sizing_close
            share_changes = target_positions - positions
            fill_values = share_changes * close
            traded = np.abs(fill_values).sum()
            before_fill = cash + positions @ close
            if before_fill <= 0:
                raise ValueError(f"Portfolio value is nonpositive before the fill on {date.date()}")
            cash -= fill_values.sum() + fee * traded
            if cash < 0:
                raise ValueError(f"Predetermined fill would leave negative cash on {date.date()}")
            positions = target_positions
            turnover = traded / before_fill
            n_trades = np.count_nonzero(share_changes)

        value = cash + positions @ close
        if value <= 0:
            raise ValueError(f"Portfolio value is nonpositive on {date.date()}")
        if np.abs(positions * close).sum() > value:
            raise ValueError(f"Marked gross exposure exceeds portfolio value on {date.date()}")
        rows.append((value / previous_value - 1, turnover, n_trades))
        previous_value = value

    return pd.DataFrame(rows, index=dates, columns=["return", "turnover", "n_trades"])


def metrics(history, risk_free):
    returns = history["return"]
    value_path = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    ann_factor = np.sqrt(SESSIONS_PER_YEAR)
    daily_vol = returns.std()
    excess = returns - risk_free.loc[history.index]
    excess_vol = excess.std()
    return {
        "cagr": float(value_path[-1] ** (1 / years) - 1) if years else None,
        "ann_vol": float(daily_vol * ann_factor) if pd.notna(daily_vol) else None,
        "sharpe": float(returns.mean() / daily_vol * ann_factor) if daily_vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * ann_factor) if excess_vol > 0 else None,
        "max_drawdown": float((value_path / np.maximum.accumulate(value_path) - 1).min()) if years else None,
        "total_return": float(value_path[-1] - 1) if years else None,
        "turnover": float(history["turnover"].sum() / years) if years else None,
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
    if not 0 < execution["cash_buffer"] < 1:
        raise ValueError("The sizing cash buffer must be between zero and one")
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

    prices, fetched_at = load_prices(
        tickers + [risk_free_ticker], dates["data_start"], dates["report_end"], config["legacy_snapshot"]
    )
    returns = prices.pct_change(fill_method=None).iloc[1:]
    targets, benchmark_targets, skipped_rebalances = build_targets(
        returns[tickers], groups, premium, model, lag, dates["report_start"]
    )
    asset_prices = prices[tickers]
    daily_results = simulate(asset_prices, targets, dates["report_start"], execution)
    benchmark_daily_results = simulate(asset_prices, benchmark_targets, dates["report_start"], execution)
    strategy = summarize(daily_results, returns[risk_free_ticker], dates["is_end"])
    benchmark = summarize(benchmark_daily_results, returns[risk_free_ticker], dates["is_end"])
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted close")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"SKIPPED_REBALANCES: {skipped_rebalances}")
    benchmark_name = "Uncleaned correlation, same portfolio rule"
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")

    results = {
        "strategy": config["strategy"],
        "data_range": [daily_results.index[0].date().isoformat(), daily_results.index[-1].date().isoformat()],
        **strategy,
        "benchmark": {"name": benchmark_name, **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

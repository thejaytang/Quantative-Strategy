"""535_shrinkage_estimators_for_mean_and_covari
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/535_shrinkage_estimators_for_mean_and_covari.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit fc4032f1dc84592dc32f39915df656911d60b83b on 2026-11-27 18:52 Europe/Oslo, exit 0, output sha256 9bf757db46abe7924c982dac35131b30b4f35de0ebc9a139046b5f5c769f05eb
Data source: Yahoo Finance 公开美元日线，经 yfinance 下载日复权收盘价。行业 ETF 为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY，另取 BIL 日总回报用于事后计算超额收益。运行日志记录所用快照下载于 2026-11-27T08:14:08+00:00。
Data fetched: 2026-11-27T17:52:41+00:00
Sample period: 日线价格输入为 2012-09-20 至 2026-10-30，2012-09-20 起的数据用于预热。模拟区间为 2014-01-02 至 2026-10-30，首日为首次成交日。 data_range (from output): {'start': '2014-01-02', 'end': '2026-10-30'}
IS/OOS split: 按 README 所述论文原始样本截止日划分。IS 为 2014-01-02 至 2024-06-06，OOS 为 2024-06-07 至 2026-10-30，份额与现金跨界延续。四组替代参数仅在 IS 运行，默认参数用于全期滚动估计。此 OOS 为回溯划分，起点早于 README 所载论文首次公开日期 2026-01-28。
Costs and slippage: 默认每边 2 bps 费用加 3 bps 固定滑点，合计 5 bps。以成交日复权收盘价乘锁定份额，按买卖绝对成交金额之和扣减现金，含首次建仓。IS 高摩擦情景为 4 bps 费用加 6 bps 滑点，合计 10 bps。滑点作为固定现金扣减建模，模拟成交价仍取收盘价。
Signal lag: 在月末最后一个 NYSE 交易日 t 收盘后，用截至 t 收盘的日收益率生成权重。默认按 t 收盘锁定买卖份额，在下一交易日 t+1 收盘模拟成交。配置更长滞后时，份额仍在成交前一交易日收盘后锁定。首笔信号和定量在 2013-12-31 收盘后，成交在 2014-01-02 收盘。
Price adjustment: yfinance auto_adjust=True, dividend- and split-adjusted closes, fractional total-return units for fills
Survivorship bias: 全样本固定使用回顾时选定的九只存续行业 ETF，存在基金选择及幸存者偏差。资产池省略后设行业基金，未模拟基金退市或清算，也无逐期成分表。
Parameter combinations tried: 5
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09415316450024336, ann_vol=0.13640167980843376, sharpe=0.7281966061222721, sharpe_excess=0.6358917333748142, max_drawdown=-0.3142061521769186, total_return=1.5530760869375313, turnover=1.617086247630385, n_trades=667
    oos: cagr=0.09391717576231451, ann_vol=0.10311149297397697, sharpe=0.9222505068072562, sharpe_excess=0.5211799983247761, max_drawdown=-0.1043279790308489, total_return=0.23916602813768417, turnover=1.1969672913910319, n_trades=161
    full: cagr=0.09410913670840992, ann_vol=0.1308213268339418, sharpe=0.7532230751823523, sharpe_excess=0.6159499325751323, max_drawdown=-0.3142061521769186, total_return=2.163685154183676, turnover=1.538712646249508, n_trades=828
    benchmark full: cagr=0.11667661224419357, ann_vol=0.15300453536944184, sharpe=0.7982020867802545, sharpe_excess=0.6808290243454371, max_drawdown=-0.3505866722187352, total_return=3.1090104312552462, turnover=0.3603171547710131, n_trades=1386
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 535_shrinkage_estimators_for_mean_and_covari\nuniverse: [XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY]\nrisk_free: BIL\nbenchmark: equal_weight  # Primary benchmark in RESULTS_JSON.\n\ndata_start: 2012-09-20\nbacktest_start: 2014-01-02  # First simulated session, after the 2013-12-31 signal.\nis_end: 2024-06-06  # Inclusive, the cited paper's data cutoff.\nbacktest_end: 2026-10-30\n# Full-day NYSE closures for Hurricane Sandy and two national days of mourning.\nexceptional_market_closures: [2012-10-29, 2012-10-30, 2018-12-05, 2025-01-09]\n\nlookback_days: 260\nrebalance_frequency: monthly\n# Count NYSE sessions from the signal close to the fill close.\nexecution_lag_days: 1\n# At the close before each fill, reserve 5% of that session's value and lock quantities.\ncash_buffer_fraction: 0.05\ncost_bps: 2.0\nslippage_bps: 3.0\n\nrobustness:\n  lookback_days: [234, 252, 286]\n  high_cost_bps: 4.0\n  high_slippage_bps: 6.0\n"
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
"""Run a long-only COV2 minimum-variance backtest on US sector ETFs."""

import hashlib
import json
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path

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
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


class NYSEHolidayCalendar(AbstractHolidayCalendar):
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


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    for key in ("data_start", "backtest_start", "is_end", "backtest_end"):
        config[key] = pd.Timestamp(config[key])
    config["exceptional_market_closures"] = pd.DatetimeIndex(config["exceptional_market_closures"])
    start, first, split, end = (
        config[key] for key in ("data_start", "backtest_start", "is_end", "backtest_end")
    )
    if not start < first < split < end:
        raise ValueError("Require data_start < backtest_start < is_end < backtest_end.")
    universe = config["universe"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or config["risk_free"] in universe:
        raise ValueError("Use at least two distinct sector ETFs and a separate cash proxy.")
    if config["rebalance_frequency"] != "monthly" or config["benchmark"] != "equal_weight":
        raise ValueError("Expected monthly rebalancing and the equal_weight benchmark.")
    lag = config["execution_lag_days"]
    if type(lag) is not int or lag < 1:
        raise ValueError("execution_lag_days must be a positive integer.")
    if not 0 < config["cash_buffer_fraction"] < 1:
        raise ValueError("cash_buffer_fraction must be between zero and one.")
    lookbacks = [config["lookback_days"], *config["robustness"]["lookback_days"]]
    if any(type(days) is not int or days < 2 for days in lookbacks):
        raise ValueError("All lookbacks must be integers of at least two sessions.")
    if len(set(lookbacks)) != len(lookbacks):
        raise ValueError("Use distinct lookbacks for the default and robustness cases.")
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    stressed_cost = config["robustness"]["high_cost_bps"]
    stressed_slippage = config["robustness"]["high_slippage_bps"]
    if not (0 <= cost <= stressed_cost and 0 <= slippage <= stressed_slippage
            and cost + slippage < stressed_cost + stressed_slippage < 10_000):
        raise ValueError("Use nonnegative costs and a higher friction case below 10,000 bps.")
    return config


def check_prices(prices, tickers, start, end, exceptional_closures):
    if list(prices.columns) != tickers or prices.empty:
        raise ValueError("Yahoo price panel is empty or has unexpected columns.")
    if prices.index.hasnans or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Yahoo price dates must be unique and increasing.")
    holidays = NYSEHolidayCalendar().holidays(start, end)
    expected = pd.bdate_range(start, end).difference(holidays.union(exceptional_closures))
    if not prices.index.equals(expected):
        missing = expected.difference(prices.index).strftime("%Y-%m-%d")[:5].tolist()
        unexpected = prices.index.difference(expected).strftime("%Y-%m-%d")[:5].tolist()
        raise ValueError(f"Yahoo price dates disagree with NYSE sessions. Missing: {missing}. Unexpected: {unexpected}.")
    values = prices.to_numpy(dtype=float)
    bad_rows, bad_cols = np.where(~np.isfinite(values) | (values <= 0))
    if bad_rows.size:
        examples = [f"{tickers[col]} {prices.index[row].date()}"
                    for row, col in zip(bad_rows[:5], bad_cols[:5])]
        raise ValueError(f"Missing or invalid adjusted close: {', '.join(examples)}")


def load_prices(config):
    tickers = [*config["universe"], config["risk_free"]]
    start, end = config["data_start"], config["backtest_end"]
    exceptional_closures = config["exceptional_market_closures"]
    cache_key = json.dumps(["yfinance daily auto_adjust close", tickers, str(start.date()), str(end.date())])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    CACHE.mkdir(exist_ok=True)
    cache_file = CACHE / f"adjusted_closes_{digest}.csv"
    if not cache_file.is_file():
        download_prices(cache_file, tickers, start, end, exceptional_closures)

    # Both the download and cache-hit paths simulate the same serialized snapshot.
    with cache_file.open(encoding="utf-8") as source:
        header = source.readline()
        if not header.startswith("# fetched_at="):
            raise ValueError(f"Missing Yahoo download timestamp in {cache_file.name}.")
        fetched_at = header.removeprefix("# fetched_at=").strip()
        if datetime.fromisoformat(fetched_at).utcoffset() is None:
            raise ValueError(f"Yahoo download timestamp needs a timezone in {cache_file.name}.")
        prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"])
    check_prices(prices, tickers, start, end, exceptional_closures)
    return prices, fetched_at


def download_prices(cache_file, tickers, start, end, exceptional_closures):
    with tempfile.TemporaryDirectory(dir=cache_file.parent) as working_cache:
        yf.set_tz_cache_location(working_cache)
        for attempt in range(2):
            raw = yf.download(
                tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d", auto_adjust=True, ignore_tz=True, keepna=True,
                group_by="column", multi_level_index=True, progress=False, timeout=15,
            )
            try:
                if raw is None or raw.empty or "Close" not in raw.columns.get_level_values(0):
                    raise ValueError("Yahoo returned no adjusted closes.")
                prices = raw["Close"].reindex(columns=tickers).sort_index()
                check_prices(prices, tickers, start, end, exceptional_closures)
            except ValueError as error:
                if attempt == 1:
                    raise RuntimeError(f"Public Yahoo download failed: {error}") from error
            else:
                break

        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        temporary_csv = Path(working_cache) / "prices.csv"
        with temporary_csv.open("w", encoding="utf-8") as destination:
            destination.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(destination, index_label="Date", date_format="%Y-%m-%d")
        temporary_csv.replace(cache_file)


def cov2(returns):
    """C149 §3.2, equations (6)–(8), with a two-parameter target."""
    sessions, assets = returns.shape
    centered = returns - returns.mean(axis=0)
    sample = centered.T @ centered / sessions
    off_diagonal = ~np.eye(assets, dtype=bool)
    sample_off_sum = sample[off_diagonal].sum()
    target = np.full_like(sample, sample_off_sum / (assets * (assets - 1)))
    np.fill_diagonal(target, np.trace(sample) / assets)

    squared = centered ** 2
    pi_terms = squared.T @ squared / sessions - sample ** 2
    off_products = centered.sum(axis=1) ** 2 - squared.sum(axis=1)
    rho_off = (np.mean(off_products ** 2) - sample_off_sum ** 2) / (assets * (assets - 1))
    rho = np.trace(pi_terms) + rho_off
    gamma = np.sum((sample - target) ** 2)
    if gamma <= np.finfo(float).eps * np.sum(sample ** 2):
        return sample
    intensity = np.clip((pi_terms.sum() - rho) / (sessions * gamma), 0.0, 1.0)
    return (1.0 - intensity) * sample + intensity * target


def minimum_variance_weights(covariance):
    assets = covariance.shape[0]
    equal = np.full(assets, 1.0 / assets)
    typical_variance = np.trace(covariance) / assets
    if typical_variance == 0:
        return equal
    scaled = covariance / typical_variance
    result = minimize(
        lambda weights: 0.5 * weights @ scaled @ weights,
        equal,
        jac=lambda weights: scaled @ weights,
        bounds=[(0.0, 1.0)] * assets,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0,
                     "jac": lambda weights: np.ones_like(weights)},
        method="SLSQP", options={"ftol": 1e-13, "maxiter": 200},
    )
    weights = result.x
    gradient = scaled @ weights
    if (not result.success or abs(weights.sum() - 1.0) > 1e-9
            or weights.min() < -1e-9 or gradient @ weights - gradient.min() > 2e-6):
        raise RuntimeError(f"Minimum variance solve failed: {result.message}")
    weights = weights.clip(min=0.0)
    return weights / weights.sum()


def rebalance_schedule(sessions, start, lag):
    # The price loader checks these dates against the NYSE calendar before use.
    months = sessions.to_period("M")
    month_end_rows = np.flatnonzero(months[:-1] != months[1:])
    schedule = {}
    for signal_row in month_end_rows:
        fill_row = signal_row + lag
        if fill_row < len(sessions) and sessions[fill_row] >= start:
            schedule[fill_row] = signal_row
    return schedule


def build_targets(asset_returns, schedule, lookback, *, shrink=True):
    values = asset_returns.to_numpy()
    targets = {}
    for fill_row, signal_row in schedule.items():
        first_row = signal_row + 1 - lookback
        if first_row < 1:
            raise ValueError(f"Insufficient warmup for lookback_days={lookback}.")
        window = values[first_row:signal_row + 1]
        covariance = cov2(window) if shrink else np.cov(window, rowvar=False, ddof=0)
        try:
            weights = minimum_variance_weights(covariance)
        except RuntimeError as error:
            raise RuntimeError(f"Signal {asset_returns.index[signal_row].date()}: {error}") from error
        targets[fill_row] = weights
    return targets


def size_rebalance(positions, cash, target, sizing_prices, cash_buffer):
    value = cash + positions @ sizing_prices
    planned_positions = value * (1.0 - cash_buffer) * target / sizing_prices
    # Skip numerical dust using the same prices that determine the order quantities.
    trade_values = np.abs(planned_positions - positions) * sizing_prices
    planned_positions = np.where(trade_values > 1e-9 * value, planned_positions, positions)
    return planned_positions, planned_positions - positions


def simulate(asset_prices, targets, friction_bps, cash_buffer, start):
    first_row = asset_prices.index.searchsorted(start)
    calendar = asset_prices.index[first_row:]
    closes = asset_prices.to_numpy()
    positions = np.zeros(closes.shape[1])
    cash = previous_value = 1.0
    daily_results = np.zeros((len(calendar), 3))
    friction_rate = friction_bps / 10_000.0

    pending = None
    first_target = targets.get(first_row)
    if first_target is not None:
        if first_row == 0:
            raise ValueError("A planned fill requires a preceding session close for sizing.")
        # The initial allocation is sized at the last close before the simulation starts.
        pending = size_rebalance(positions, cash, first_target, closes[first_row - 1], cash_buffer)

    for output_row, row in enumerate(range(first_row, len(closes))):
        close = closes[row]
        if pending is not None:
            planned_positions, quantities = pending
            fill_values = quantities * close
            gross_traded = np.abs(fill_values).sum()
            trading_cost = gross_traded * friction_rate
            cash_after_fill = cash - fill_values.sum() - trading_cost
            if not np.isfinite(cash_after_fill) or cash_after_fill < 0.0:
                date = asset_prices.index[row].date()
                raise RuntimeError(
                    f"{date}: fixed quantities would leave negative cash or a nonfinite balance ({cash_after_fill:.8g})."
                )
            daily_results[output_row, 1] = gross_traded / (cash + positions @ close)
            daily_results[output_row, 2] = np.count_nonzero(quantities)
            positions = planned_positions
            cash = cash_after_fill
        value = cash + positions @ close
        daily_results[output_row, 0] = value / previous_value - 1.0
        previous_value = value

        # Plan only the next session's fill, using today's closing prices and settled positions.
        next_row = row + 1
        next_target = targets.get(next_row) if next_row < len(closes) else None
        pending = None if next_target is None else size_rebalance(
            positions, cash, next_target, close, cash_buffer,
        )

    return pd.DataFrame(daily_results, index=calendar, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns):
    returns = daily_results["return"].to_numpy()
    sessions = len(returns)
    if not sessions:
        return dict(cagr=None, ann_vol=None, sharpe=None, sharpe_excess=None,
                    max_drawdown=None, total_return=None, turnover=None, n_trades=0)
    excess = returns - cash_returns.loc[daily_results.index].to_numpy()
    growth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    years = sessions / SESSIONS_PER_YEAR
    annual_scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1) if sessions > 1 else None
    excess_vol = excess.std(ddof=1) if sessions > 1 else None
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0),
        "ann_vol": float(vol * annual_scale) if vol is not None else None,
        "sharpe": float(annual_scale * returns.mean() / vol) if vol else None,
        "sharpe_excess": float(annual_scale * excess.mean() / excess_vol) if excess_vol else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum() / years),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, is_end):
    return {
        "is": metrics(daily_results.loc[:is_end], cash_returns),
        "oos": metrics(daily_results.loc[daily_results.index > is_end], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    config = load_config()
    prices, fetched_at = load_prices(config)
    returns = prices.pct_change(fill_method=None)
    asset_prices = prices[config["universe"]]
    asset_returns = returns[config["universe"]]
    cash_returns = returns[config["risk_free"]]
    start, is_end = config["backtest_start"], config["is_end"]
    lag, lookback = config["execution_lag_days"], config["lookback_days"]
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    cash_buffer = config["cash_buffer_fraction"]
    is_asset_returns = asset_returns.loc[:is_end]
    is_asset_prices = asset_prices.loc[:is_end]
    is_schedule = rebalance_schedule(is_asset_returns.index, start, lag)

    # Evaluate the alternative configurations using only the IS slice.
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, dividend- and split-adjusted closes, "
          "fractional total-return units for fills")
    print(f"RISK_FREE_PROXY: {config['risk_free']} adjusted daily total return")
    print("ROBUSTNESS (IS only)")
    robustness = config["robustness"]
    cases = [(days, cost, slippage) for days in robustness["lookback_days"]]
    cases.append((lookback, robustness["high_cost_bps"], robustness["high_slippage_bps"]))
    for days, case_cost, case_slippage in cases:
        targets = build_targets(is_asset_returns, is_schedule, days)
        run = simulate(is_asset_prices, targets, case_cost + case_slippage, cash_buffer, start)
        summary = metrics(run, cash_returns.loc[:is_end])
        print(f"  lookback_days={days} cost_bps={case_cost:g} slippage_bps={case_slippage:g} "
              f"{json.dumps(summary, allow_nan=False)}")
    schedule = rebalance_schedule(asset_returns.index, start, lag)
    targets = build_targets(asset_returns, schedule, lookback)
    friction = cost + slippage
    daily_results = simulate(asset_prices, targets, friction, cash_buffer, start)
    equal = np.full(len(config["universe"]), 1.0 / len(config["universe"]))
    benchmark_results = simulate(asset_prices, dict.fromkeys(schedule, equal), friction, cash_buffer, start)
    sample_targets = build_targets(asset_returns, schedule, lookback, shrink=False)
    sample_results = simulate(asset_prices, sample_targets, friction, cash_buffer, start)
    default_results = period_metrics(daily_results, cash_returns, is_end)
    benchmark = period_metrics(benchmark_results, cash_returns, is_end)
    sample_benchmark = period_metrics(sample_results, cash_returns, is_end)
    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(default_results[period], allow_nan=False)}")
        print(f"BENCHMARK equal_weight {period.upper()}: {json.dumps(benchmark[period], allow_nan=False)}")
        print(f"BENCHMARK sample_covariance_gmv {period.upper()}: "
              f"{json.dumps(sample_benchmark[period], allow_nan=False)}")

    result = {
        "strategy": config["strategy"],
        "data_range": {
            "start": str(daily_results.index[0].date()),
            "end": str(daily_results.index[-1].date()),
        },
        **default_results,
        "benchmark": {"name": config["benchmark"], **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def stop_on_timeout(_signum, _frame):
    raise SystemExit("Backtest stopped at the 9-minute time limit.")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(9 * 60)
    try:
        main()
    finally:
        signal.alarm(0)

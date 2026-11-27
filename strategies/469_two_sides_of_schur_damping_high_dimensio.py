"""469_two_sides_of_schur_damping_high_dimensio
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/469_two_sides_of_schur_damping_high_dimensio.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit c71364ad99535bbf59be998b004cff431ee83048 on 2026-11-27 11:54 Europe/Oslo, exit 0, output sha256 f2d1d193a394b3e69d01b9e5a58db90e06754700471f4e769ebd2f38073436bd
Data source: 通过 yfinance 读取 Yahoo Finance 的美元计价日频 ETF 收盘价：SPY、IWM、EFA、EEM、VNQ、XLU、IEF、TLT、TIP、LQD、HYG、EMB、GLD、DBC、UUP；另用 BIL 日总收益作为现金收益及超额夏普的代理（config.yaml:2-18；strategy.py:62-87、207-217、233-241；README.md:16）。
Data fetched: 2026-11-27T10:54:27+00:00
Sample period: 价格请求从 2009-01-01 至 2025-12-31，2009 年用于预热。报告配置从 2010-01-01 至 2025-12-31，本次日志记录的实际报告区间为 2010-01-04 至 2025-12-31（config.yaml:19-22；README.md:18；strategy.py:237-241、250-254；logs/run_20261127_080849.log:4）。   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: 按实际有报价的交易日切分。样本内为 2010-01-01 至 2017-12-31（含分界日），样本外为 2018-01-01 至 2025-12-31。先连续模拟整段路径，再按日期切分统计，样本外承接已有仓位（config.yaml:20-22；strategy.py:225-230、239-249；README.md:17-18）。
Costs and slippage: 买入和卖出的模拟成交额均扣除 2 bps 交易成本及 3 bps 固定滑点，共 5 bps。调仓时以各标的买卖成交额绝对值之和为计费基数，按扣费后的组合价值确定目标仓位。等权基准使用同一费率（config.yaml:27-28；strategy.py:176-198、242-246；README.md:20）。
Signal lag: 在月末交易日 t 收盘后，用截至该收盘的 45 个日收益率计算目标权重，于下一交易日 t+1 收盘模拟成交。t 到 t+1 的收盘到收盘收益仍由调仓前仓位获得，新仓位从 t+1 收盘到再下一交易日收盘的收益区间生效（config.yaml:23-25；strategy.py:157-170、176-198、237-241；README.md:19）。
Price adjustment: yfinance auto_adjust=True dividend- and split-adjusted close (total return)
Survivorship bias: 固定资产池按当前可取得完整历史的 ETF 回溯构建，README 明示存在存续偏差和事后选样偏差（config.yaml:2-17；strategy.py:55-58、233-237；README.md:24）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.040544726378848095, ann_vol=0.03221346612582867, sharpe=1.2499821439640646, sharpe_excess=1.2288560997675424, max_drawdown=-0.06230562697826347, total_return=0.3736640572958736, turnover=5.879882265299127, n_trades=1086
    oos: cagr=0.02952826909646311, ann_vol=0.0405551264991964, sharpe=0.7379225827987614, sharpe_excess=0.1390098721462251, max_drawdown=-0.10597918207395574, total_return=0.26140761773569654, turnover=5.2551256134649345, n_trades=1033
    full: cagr=0.03502457854325858, ann_vol=0.03661740923058102, sharpe=0.9585323162160332, sharpe_excess=0.6190473641482592, max_drawdown=-0.10597918207395607, total_return=0.7327503060827323, turnover=5.567659196999286, n_trades=2119
    benchmark: cagr=0.06111432606785949, ann_vol=0.07892624050719213, sharpe=0.7912290291726292, sharpe_excess=0.6336247940924836, max_drawdown=-0.18826723143806778, total_return=1.578558570337659, turnover=0.3188152914929827, n_trades=2880
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 469_two_sides_of_schur_damping_high_dimensio\nuniverse:\n  - SPY\n  - IWM\n  - EFA\n  - EEM\n  - VNQ\n  - XLU\n  - IEF\n  - TLT\n  - TIP\n  - LQD\n  - HYG\n  - EMB\n  - GLD\n  - DBC\n  - UUP\nrisk_free_ticker: BIL\nprice_start: '2009-01-01'\nbacktest_start: '2010-01-01'\nis_end: '2017-12-31'\nbacktest_end: '2025-12-31'\nrebalance: monthly\nlag_sessions: 1\nlookback_sessions: 45\ngamma: 0.86\ncash_buffer: 0.05\ncost_bps: 2.0\nslippage_bps: 3.0\n"
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
"""Rolling multi-asset allocation using Cotton's damped Schur blocks."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
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
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_BASIS = "yfinance auto_adjust=True dividend- and split-adjusted close (total return)"
SESSIONS_PER_YEAR = 252


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Regular holidays and the full-day emergency closures in the study period."""

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
        Holiday("Hurricane Sandy", year=2012, month=10, day=29),
        Holiday("Hurricane Sandy", year=2012, month=10, day=30),
        Holiday("Bush day of mourning", year=2018, month=12, day=5),
        Holiday("Carter day of mourning", year=2025, month=1, day=9),
    ]


def load_config():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    for key in ("price_start", "backtest_start", "is_end", "backtest_end"):
        config[key] = pd.Timestamp(config[key])
    if not config["price_start"] < config["backtest_start"] <= config["is_end"] < config["backtest_end"]:
        raise ValueError("Expected price_start < backtest_start <= is_end < backtest_end")
    universe = config["universe"]
    if len(universe) < 2 or len(set(universe)) != len(universe) or config["risk_free_ticker"] in universe:
        raise ValueError("Use distinct ETFs and a separate cash-return proxy")
    if config["rebalance"] != "monthly":
        raise ValueError("Expected monthly rebalancing")
    for key, minimum in (("lag_sessions", 1), ("lookback_sessions", 3)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    if not 0 <= config["gamma"] <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if not 0 < config["cash_buffer"] < 1:
        raise ValueError("cash_buffer must be in (0, 1)")
    fees = np.array([config["cost_bps"], config["slippage_bps"]], dtype=float)
    if not np.isfinite(fees).all() or (fees < 0).any() or fees.sum() >= 10_000:
        raise ValueError("Cost and slippage must be finite, nonnegative and total under 10,000 bps")
    return config


def validate_prices(prices, tickers, start, end):
    prices = prices.reindex(columns=tickers).sort_index()
    if prices.empty or prices.index.has_duplicates:
        raise ValueError("Price panel is empty or contains duplicate sessions")
    expected = pd.bdate_range(start, end).difference(NYSEHolidayCalendar().holidays(start, end))
    if not prices.index.equals(expected):
        missing = expected.difference(prices.index).strftime("%Y-%m-%d").tolist()
        unexpected = prices.index.difference(expected).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"NYSE session mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        counts = invalid.sum()
        raise ValueError(f"Missing or invalid adjusted closes: {counts[counts > 0].to_dict()}")
    return prices


def load_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    request = [PRICE_BASIS, tickers, str(start.date()), str(end.date())]
    cache_key = sha256(json.dumps(request).encode("utf-8")).hexdigest()[:20]
    path = CACHE / f"prices_{cache_key}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as cached:
            fetched_at = cached.readline().strip().removeprefix("# fetched_at=")
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(cached, index_col=0, parse_dates=True)
        return validate_prices(prices, tickers, start, end), fetched_at

    with TemporaryDirectory(prefix=".yfinance-", dir=CACHE) as yf_cache:
        yf.set_tz_cache_location(yf_cache)
        panel = yf.download(
            tickers,
            start=str(start.date()),
            end=str((end + pd.Timedelta(days=1)).date()),
            interval="1d",
            auto_adjust=True,
            progress=False,
            timeout=15,
        )
    if panel is None or panel.empty:
        raise RuntimeError("Yahoo Finance returned no adjusted prices")
    prices = validate_prices(panel["Close"], tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as pending:
        pending_path = Path(pending.name)
        try:
            pending.write(f"# fetched_at={fetched_at}\n")
            prices.to_csv(pending, index_label="date")
            pending.flush()
            pending_path.replace(path)
        finally:
            pending_path.unlink(missing_ok=True)
    return prices, fetched_at


def schur_direction(covariance, budget, gamma):
    """Return a direction scaled to the inverse variance of its unit-budget form."""
    if len(budget) == 1:
        return budget / covariance[0, 0]

    cut = len(budget) // 2
    left = covariance[:cut, :cut]
    right = covariance[cut:, cut:]
    cross = covariance[:cut, cut:]
    left_budget, right_budget = budget[:cut], budget[cut:]

    # Cotton (2411.05807), equations (10) and (11), in generalized budget coordinates.
    left_hedge = np.linalg.solve(right, cross.T).T
    right_hedge = np.linalg.solve(left, cross).T
    left_direction = schur_direction(
        left - gamma * (left_hedge @ cross.T),
        left_budget - gamma * (left_hedge @ right_budget),
        gamma,
    )
    right_direction = schur_direction(
        right - gamma * (right_hedge @ cross),
        right_budget - gamma * (right_hedge @ left_budget),
        gamma,
    )
    direction = np.concatenate([left_direction, right_direction])
    variance = float(direction @ covariance @ direction)
    if variance == 0.0:
        return direction
    return direction * (float(budget @ direction) / variance)


def allocation(window, gamma):
    covariance = np.cov(window, rowvar=False, ddof=1)
    variances = np.diag(covariance)
    if (variances <= 0).any():
        raise ValueError("A rolling estimation window contains a constant ETF return")
    scales = np.sqrt(variances)
    correlations = covariance / np.outer(scales, scales)
    distances = np.sqrt((1.0 - np.clip(correlations, -1.0, 1.0)) / 2.0)
    np.fill_diagonal(distances, 0.0)
    order = leaves_list(linkage(squareform(distances, checks=False), method="single"))

    # A machine-scale eigenvalue floor stabilizes block solves for singular inputs.
    floor = np.finfo(float).eps * len(variances) * variances.max()
    ridge = max(0.0, floor - np.linalg.eigvalsh(covariance)[0])
    covariance = covariance + ridge * np.eye(len(variances))
    direction = schur_direction(covariance[np.ix_(order, order)], np.ones(len(order)), gamma)
    budget = direction.sum()
    if not np.isfinite(direction).all() or budget <= 0:
        raise ValueError("Invalid Schur allocation")
    long_weights = np.maximum(direction / budget, 0.0)
    weights = np.empty_like(long_weights)
    weights[order] = long_weights / long_weights.sum()
    return weights


def scheduled_targets(returns, config):
    sessions = returns.index
    month_ends = pd.Series(np.arange(len(sessions)), index=sessions).groupby(sessions.to_period("M")).last()
    lookback = config["lookback_sessions"]
    lag = config["lag_sessions"]
    targets = {}
    for signal in month_ends:
        execution = signal + lag
        if signal + 1 < lookback or execution >= len(sessions):
            continue
        date = sessions[execution]
        if date >= config["backtest_start"]:
            window = returns.iloc[signal - lookback + 1 : signal + 1]
            targets[date] = allocation(window.to_numpy(), config["gamma"])
    return targets


def simulate(prices, cash_returns, targets, fee_rate, cash_buffer):
    daily_results = []
    sessions = cash_returns.index
    previous_close = prices.iloc[prices.index.get_loc(sessions[0]) - 1].to_numpy()
    positions = np.zeros(prices.shape[1])
    cash = previous_value = 1.0
    for (date, cash_return), close in zip(cash_returns.items(), prices.loc[sessions].to_numpy()):
        target = targets.get(date)
        if target is not None:
            # Lock fractional adjusted units using information through the prior close.
            target_positions = (1.0 - cash_buffer) * previous_value * target / previous_close
            orders = target_positions - positions

        daily_return = float(positions @ (close - previous_close)) / previous_value
        daily_return += (cash / previous_value) * cash_return
        cash *= 1.0 + cash_return
        turnover = 0.0
        trades = 0
        if target is not None:
            fill_amounts = orders * close
            traded_value = float(np.abs(fill_amounts).sum())
            fees = fee_rate * traded_value
            cash_after = cash - float(fill_amounts.sum()) - fees
            if cash_after < 0.0:
                raise ValueError(f"{date.date()}: predetermined fills and fees would leave negative cash ({cash_after:.8g})")
            turnover = traded_value / (float(positions @ close) + cash)
            trades = int(np.count_nonzero(orders))
            daily_return -= fees / previous_value
            cash = cash_after
            positions = target_positions

        daily_results.append((daily_return, turnover, trades))
        previous_value = float(positions @ close) + cash
        previous_close = close
    return pd.DataFrame(daily_results, index=sessions, columns=["return", "turnover", "n_trades"])


def annual_sharpe(daily_returns):
    if len(daily_returns) < 2 or daily_returns.min() == daily_returns.max():
        return None
    return float(np.sqrt(SESSIONS_PER_YEAR) * daily_returns.mean() / daily_returns.std(ddof=1))


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    growth = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    years = len(returns) / SESSIONS_PER_YEAR
    return {
        "cagr": float(growth[-1] ** (1.0 / years) - 1.0) if years else None,
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)) if len(returns) > 1 else None,
        "sharpe": annual_sharpe(returns),
        "sharpe_excess": annual_sharpe(returns - cash_returns.loc[returns.index]),
        "max_drawdown": float((growth / np.maximum.accumulate(growth) - 1.0).min()) if years else None,
        "total_return": float(growth[-1] - 1.0) if years else None,
        "turnover": float(daily_results["turnover"].sum() / years) if years else None,
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, split):
    return {
        "is": metrics(daily_results.loc[daily_results.index <= split], cash_returns),
        "oos": metrics(daily_results.loc[daily_results.index > split], cash_returns),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    config = load_config()
    universe = config["universe"]
    cash_ticker = config["risk_free_ticker"]
    prices, fetched_at = load_prices(universe + [cash_ticker], config["price_start"], config["backtest_end"])
    returns = prices.pct_change(fill_method=None).iloc[1:]
    targets = scheduled_targets(returns[universe], config)
    cash_returns = returns.loc[config["backtest_start"] :, cash_ticker]
    if cash_returns.empty:
        raise ValueError("The requested backtest window contains no reporting sessions")
    asset_prices = prices[universe]
    fee_rate = (config["cost_bps"] + config["slippage_bps"]) / 10_000
    cash_buffer = config["cash_buffer"]
    daily_results = simulate(asset_prices, cash_returns, targets, fee_rate, cash_buffer)
    equal_weights = np.full(len(universe), 1.0 / len(universe))
    benchmark_targets = dict.fromkeys(targets, equal_weights)
    benchmark_results = simulate(asset_prices, cash_returns, benchmark_targets, fee_rate, cash_buffer)

    strategy_metrics = period_metrics(daily_results, cash_returns, config["is_end"])
    benchmark_metrics = period_metrics(benchmark_results, cash_returns, config["is_end"])
    date_range = {"start": str(cash_returns.index[0].date()), "end": str(cash_returns.index[-1].date())}
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_BASIS}")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"Date Range: {date_range['start']} to {date_range['end']}")
    print(f"Risk-free proxy: {cash_ticker}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy_metrics[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()}: {json.dumps(benchmark_metrics[period], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": date_range,
        **strategy_metrics,
        "benchmark": {"name": "Monthly equal-weight ETF basket", **benchmark_metrics["full"]},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
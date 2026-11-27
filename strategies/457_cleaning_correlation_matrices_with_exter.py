"""457_cleaning_correlation_matrices_with_exter
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/457_cleaning_correlation_matrices_with_exter.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:05 Europe/Oslo, exit 0, output sha256 ca865531a121706f9a7076033ed8f06434155769c3b9ca33b0abf6fe96f2e74f
Data source: yfinance 免费美元日线。股票 ETF 为 SPY、IWM、EFA、EEM、EWJ、EWU，债券 ETF 为 IEF、TLT、IEI、TIP，外汇产品为 FXE、FXY、FXB、FXA、FXC。BIL 日总回报仅用于计算超额 Sharpe。代码优先读取同一请求的本地缓存，缓存缺失时通过 yfinance 下载。config.yaml:2-6，README.md:19-20，strategy.py:21-44、156-169、225-233。
Data fetched: 2026-11-27T11:05:11+00:00
Sample period: 请求价格区间为 2008-01-01 至 2025-12-31，评估门槛为 2012-01-03，并要求至少 150 个有效标准化交易周预热。日志记录的实际评估区间为 2012-01-04 至 2025-12-31，以首个合规建仓收盘开始。config.yaml:7-11、16-19，strategy.py:103-120、225-239，README.md:21，logs/run_20261127_074942.log:4。   data_range (from output): ['2012-01-04', '2025-12-31']
IS/OOS split: 样本内为 2012-01-04 至 2018-12-31（含分割日），样本外为分割日后至 2025-12-31（名义起点 2019-01-01，以实际交易日为准）。连续生成信号和模拟仓位，再按日期切分收益，组合和估计状态跨界延续。README 将其记为时间切分，整个回测区间早于论文发表。config.yaml:7-11，strategy.py:91-124、127-153、177-181、225-234，README.md:21，logs/run_20261127_074942.log:4。
Costs and slippage: 单边成本 2 bps，滑点按固定 3 bps 建模，合计 5 bps 乘以每次调仓的买卖绝对成交名义金额之和，初次建仓也计费。模拟求解费用扣减后的权益，再按该权益落实目标权重。滑点作为权益扣减项计入，成交价没有另行偏移。config.yaml:12-15，strategy.py:127-151、213-218，README.md:23。
Signal lag: 每个交易周首个交易日收盘 t 生成目标仓位，使用截至 t 收盘的日波动率和此前已结束交易周的相关矩阵。配置延迟一交易日，于 t+1 交易日收盘调仓。成交日的收盘到收盘收益先按旧仓位记账，随后计费并更新仓位，新仓位从下一段收盘到收盘收益开始生效。config.yaml:12-15，strategy.py:91-120、127-151，README.md:22。
Price adjustment: yfinance auto_adjust=True, split- and distribution-adjusted closes and fractional units
Survivorship bias: 资产池是预先列明的固定名单。README 说明名单由当前可取得完整历史的存续产品组成，存在幸存者偏差，遗漏样本期内已清盘或更换代码的产品。代码对固定名单检查价格有效性。config.yaml:2-6，README.md:33，strategy.py:46-55、188-202。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.028295873571517305, ann_vol=0.02612020022133279, sharpe=1.081367131795792, sharpe_excess=0.9557370002796821, max_drawdown=-0.03877131345017448, total_return=0.21502779478159106, turnover=0.7508899543527232, n_trades=5475
    oos: cagr=0.024241897040873317, ann_vol=0.031540407327709186, sharpe=0.775240141280264, sharpe_excess=-0.02646829503724673, max_drawdown=-0.07092106883880533, total_return=0.18209570226307115, turnover=0.6984313769120636, n_trades=5490
    full: cagr=0.02626630753540571, ann_vol=0.028954298935445676, sharpe=0.9099829659083963, sharpe_excess=0.4169850520209492, max_drawdown=-0.07092106883880611, total_return=0.4362791343414891, turnover=0.7246532120124103, n_trades=10965
    benchmark: cagr=0.010407005523610291, ann_vol=0.010911718642771366, sharpe=0.9542912593205897, sharpe_excess=-0.3413646140601073, max_drawdown=-0.03148996217897215, total_return=0.15554881797439712, turnover=1.242901701542152, n_trades=10965
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "457_ cleaning_correlation_matrices_with_exter"\nuniverse:\n  equity: [SPY, IWM, EFA, EEM, EWJ, EWU]\n  bonds: [IEF, TLT, IEI, TIP]\n  currency: [FXE, FXY, FXB, FXA, FXC]\nrisk_free: BIL\ndates:\n  data_start: 2008-01-01\n  evaluation_start: 2012-01-04\n  is_end: 2018-12-31\n  end: 2025-12-31\nexecution:\n  signal_lag_sessions: 1\n  cash_buffer: 0.10\n  cost_bps: 2.0\n  slippage_bps: 3.0\nestimator:\n  volatility_alpha_daily: 0.01\n  correlation_alpha_daily: 0.0013333333333333333\n  warmup_weeks: 150\n'
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
"""Research backtest for the grouped Maximum Variance correlation estimator."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

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
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
MIN_RELATIVE_EIGENVALUE = 1e-10


def validate_sessions(dates, start, end):
    # NYSE cash equity full-day holidays and exceptional closures for this sample.
    calendar = AbstractHolidayCalendar(rules=[
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
    ])
    exceptional_closures = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    closures = calendar.holidays(start, end).union(exceptional_closures)
    sessions = pd.bdate_range(start, end).difference(closures)
    missing = sessions.difference(dates)
    unexpected = dates.difference(sessions)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Adjusted closes disagree with the NYSE session calendar. "
            f"Missing ({len(missing)}): {missing[:10].strftime('%Y-%m-%d').tolist()}. "
            f"Unexpected ({len(unexpected)}): {unexpected[:10].strftime('%Y-%m-%d').tolist()}."
        )


def adjusted_closes(tickers, start, end):
    query = json.dumps(["yfinance", "auto_adjust", tickers, str(start), str(end)])
    key = hashlib.sha256(query.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"close_{key}.csv"
    cache_hit = cache_file.is_file()
    if cache_hit:
        with cache_file.open(encoding="utf-8") as cached:
            fetched_at = datetime.fromisoformat(cached.readline().strip()).isoformat()
            close = pd.read_csv(cached, index_col=0)
    else:
        CACHE_DIR.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=CACHE_DIR) as download_cache:
            yf.set_tz_cache_location(download_cache)
            prices = yf.download(
                tickers,
                start=str(start.date()),
                end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d",
                auto_adjust=True,
                progress=False,
                timeout=15,
            )
            fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        close = prices["Close"]

    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close = close.sort_index().loc[start:end].reindex(columns=tickers)
    if close.empty or close.index.has_duplicates:
        raise ValueError("The adjusted close panel is empty or has duplicate dates.")
    validate_sessions(close.index, start, end)
    bad = (~np.isfinite(close) | (close <= 0)).sum()
    if bad.any():
        raise ValueError(f"Missing or invalid adjusted closes by ticker: {bad[bad > 0].to_dict()}")

    if not cache_hit:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, delete=False) as tmp:
            pending = Path(tmp.name)
            tmp.write(f"{fetched_at}\n")
            close.to_csv(tmp, index_label="date")
        pending.replace(cache_file)
    return close, fetched_at


def to_correlation(matrix):
    diagonal = np.diag(matrix)
    if not np.isfinite(matrix).all() or (diagonal <= 0).any():
        raise ValueError("Correlation normalization requires finite moments and positive variances.")
    scale = np.sqrt(diagonal)
    return matrix / np.outer(scale, scale)


def maxvar_correlation(correlation, groups):
    # Disjoint groups make the leading group eigenvectors an orthonormal basis.
    n_assets = len(correlation)
    basis = np.zeros((n_assets, len(groups)))
    for column, group in enumerate(groups):
        _, eigenvectors = np.linalg.eigh(correlation[np.ix_(group, group)])
        basis[group, column] = eigenvectors[:, -1]

    # P (P.T C P) P.T equals Z M Z.T, with every constrained component retained.
    projected = basis.T @ correlation @ basis
    residual = (np.trace(correlation) - np.trace(projected)) / (n_assets - len(groups))
    completed = basis @ projected @ basis.T + residual * (np.eye(n_assets) - basis @ basis.T)
    return to_correlation(completed)


def portfolio_weights(correlation, volatility, expected_sharpe):
    if not np.isfinite(volatility).all() or (volatility <= 0).any():
        raise ValueError("Weight calculation requires positive finite asset volatilities.")
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues[0] <= MIN_RELATIVE_EIGENVALUE * eigenvalues[-1]:
        raise ValueError(f"Correlation matrix is singular or ill-conditioned (eigenvalues {eigenvalues[[0, -1]]}).")
    direction = np.linalg.solve(correlation, expected_sharpe) / volatility
    gross = np.abs(direction).sum()
    if not np.isfinite(gross) or gross <= 0:
        raise ValueError("Portfolio direction has invalid gross weight.")
    return direction / gross


def target_weights(returns, groups, expected_sharpe, start, execution, estimator):
    lag = execution["signal_lag_sessions"]
    daily_variance = returns.pow(2).ewm(alpha=estimator["volatility_alpha_daily"], adjust=False).mean()
    daily_vol = np.sqrt(daily_variance)
    prior_vol = daily_vol.shift()
    standardized = returns.div(prior_vol.where(prior_vol > 0))
    weekly_alpha = 1 - (1 - estimator["correlation_alpha_daily"]) ** 5
    weeks = returns.index.to_period("W-FRI")
    moment = np.zeros((returns.shape[1], returns.shape[1]))
    cleaned_targets, raw_targets = {}, {}
    n_weeks = 0

    for positions in returns.groupby(weeks, sort=True).indices.values():
        signal_pos = positions[0]
        fill_pos = signal_pos + lag
        # At the first close of this week, the previous week's moment is complete.
        if fill_pos < len(returns) and n_weeks >= estimator["warmup_weeks"]:
            fill_date = returns.index[fill_pos]
            if fill_date >= start:
                vol = daily_vol.iloc[signal_pos].to_numpy()
                try:
                    raw = to_correlation(moment)
                    raw_targets[fill_date] = portfolio_weights(raw, vol, expected_sharpe)
                    cleaned = maxvar_correlation(raw, groups)
                    cleaned_targets[fill_date] = portfolio_weights(cleaned, vol, expected_sharpe)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    raise ValueError(f"Cannot form weights for signal on {returns.index[signal_pos].date()}: {exc}") from exc

        block = standardized.iloc[positions].to_numpy()
        if np.isfinite(block).all():
            weekly = block.sum(axis=0)
            moment = (1 - weekly_alpha) * moment + weekly_alpha * np.outer(weekly, weekly)
            n_weeks += 1

    return cleaned_targets, raw_targets


def simulate(close, close_targets, cost_rate, cash_buffer):
    """Evaluate from the second close onward, with the first row for sizing."""
    positions = np.zeros(close.shape[1])
    cash = equity = 1.0
    daily_results = []
    prices = close.to_numpy()
    for date, sizing_close, fill_close in zip(close.index[1:], prices[:-1], prices[1:], strict=True):
        previous_equity = equity
        turnover = 0.0
        n_trades = 0
        # Freeze signed changes using the prior close and the positions then in place.
        target = close_targets.get(date)
        if target is not None:
            trade_units = (1 - cash_buffer) * previous_equity * target / sizing_close - positions

            # The fill price marks the old positions and values the frozen trades.
            equity_before_fill = cash + positions @ fill_close
            if equity_before_fill <= 0:
                raise ValueError(f"Portfolio equity became nonpositive before the fill on {date.date()}.")
            traded_value = trade_units * fill_close
            traded_gross = np.abs(traded_value).sum()
            cash -= traded_value.sum() + cost_rate * traded_gross
            if cash < 0:
                raise ValueError(f"Frozen trade quantities would make cash negative on {date.date()}.")
            positions += trade_units
            turnover = traded_gross / equity_before_fill
            n_trades = np.count_nonzero(trade_units)

        marked_positions = positions * fill_close
        equity = cash + marked_positions.sum()
        if equity <= 0:
            raise ValueError(f"Portfolio equity became nonpositive on {date.date()}.")
        # Gross exposure at most equity keeps shorts fully collateralized.
        if np.abs(marked_positions).sum() > equity:
            raise ValueError(f"Gross exposure exceeds portfolio equity at the close on {date.date()}.")
        daily_results.append((equity / previous_equity - 1, turnover, n_trades))

    return pd.DataFrame(daily_results, index=close.index[1:], columns=["return", "turnover", "n_trades"])


def metrics(daily_results, risk_free):
    returns = daily_results["return"]
    excess = returns - risk_free.loc[returns.index]
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    annual_scale = np.sqrt(SESSIONS_PER_YEAR)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    total = wealth.iloc[-1] if len(wealth) else None
    return {
        "cagr": float(total ** (SESSIONS_PER_YEAR / len(returns)) - 1) if total is not None else None,
        "ann_vol": float(vol * annual_scale) if np.isfinite(vol) else None,
        "sharpe": float(returns.mean() / vol * annual_scale) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_vol * annual_scale) if excess_vol > 0 else None,
        "max_drawdown": float((wealth / peak - 1).min()) if total is not None else None,
        "total_return": float(total - 1) if total is not None else None,
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR) if total is not None else None,
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, risk_free, split):
    periods = {
        "is": daily_results.loc[:split],
        "oos": daily_results.loc[daily_results.index > split],
        "full": daily_results,
    }
    return {name: metrics(period, risk_free) for name, period in periods.items()}


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    universe = config["universe"]
    group_names = ("equity", "bonds", "currency")
    tickers = [ticker for name in group_names for ticker in universe[name]]
    groups = [
        np.array([tickers.index(ticker) for ticker in universe[name]], dtype=int)
        for name in group_names
    ]
    cash_proxy = config["risk_free"]
    if (
        any(len(group) == 0 for group in groups)
        or len(tickers) <= len(groups)
        or len(set(tickers + [cash_proxy])) != len(tickers) + 1
        or cash_proxy != "BIL"
    ):
        raise ValueError("Use three disjoint nonempty asset groups, with BIL reserved for excess returns.")
    expected_sharpe = np.ones(len(tickers))
    expected_sharpe[groups[2]] = 0

    dates = config["dates"]
    data_start = pd.Timestamp(dates["data_start"])
    start = pd.Timestamp(dates["evaluation_start"])
    split = pd.Timestamp(dates["is_end"])
    end = pd.Timestamp(dates["end"])
    if not data_start < start <= split < end:
        raise ValueError("Check the order of the warmup, evaluation and split dates.")
    execution = config["execution"]
    lag = execution["signal_lag_sessions"]
    cash_buffer = execution["cash_buffer"]
    fees_bps = [execution["cost_bps"], execution["slippage_bps"]]
    cost_rate = sum(fees_bps) / 10_000
    if type(lag) is not int or lag < 1 or min(fees_bps) < 0 or not 0 <= cost_rate < 1:
        raise ValueError("Execution requires a positive session lag and nonnegative costs and slippage.")
    if not 0 < cash_buffer < 1:
        raise ValueError("Execution requires a cash buffer between zero and one.")
    estimator = config["estimator"]
    if estimator["warmup_weeks"] < len(tickers) or not all(
        0 < estimator[key] < 1 for key in ("volatility_alpha_daily", "correlation_alpha_daily")
    ):
        raise ValueError("Use enough warmup weeks for the raw matrix and EMA coefficients between 0 and 1.")

    close, fetched_at = adjusted_closes(tickers + [cash_proxy], data_start, end)
    daily_returns = close.pct_change(fill_method=None).iloc[1:]
    risky_returns = daily_returns[tickers]
    cleaned, raw = target_weights(risky_returns, groups, expected_sharpe, start, execution, estimator)
    first_eval_pos = close.index.searchsorted(start)
    if not 0 < first_eval_pos < len(close):
        raise ValueError("The evaluation needs a prior close for sizing and at least one evaluation session.")
    evaluation_closes = close[tickers].iloc[first_eval_pos - 1:]
    strategy_daily = simulate(evaluation_closes, cleaned, cost_rate, cash_buffer)
    benchmark_daily = simulate(evaluation_closes, raw, cost_rate, cash_buffer)
    stats = period_metrics(strategy_daily, daily_returns[cash_proxy], split)
    benchmark_stats = period_metrics(benchmark_daily, daily_returns[cash_proxy], split)
    data_range = [date.date().isoformat() for date in strategy_daily.index[[0, -1]]]

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True, split- and distribution-adjusted closes and fractional units")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"{config['strategy']} | {data_range[0]} to {data_range[1]}")
    for name, portfolio_stats in (("MaxVar", stats), ("Raw correlation benchmark", benchmark_stats)):
        for period, values in portfolio_stats.items():
            print(f"{name} {period.upper()}: {json.dumps(values, allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **stats,
        "benchmark": benchmark_stats["full"],
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

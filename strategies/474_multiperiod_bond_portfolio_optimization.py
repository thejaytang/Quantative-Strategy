"""474_multiperiod_bond_portfolio_optimization
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/474_multiperiod_bond_portfolio_optimization.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit cc50485fb68601d5f9b01f48262e3a778be06982 on 2026-11-27 09:02 Europe/Oslo, exit 0, output sha256 5e4f365a2cc76d28f98f594dc66f073261394fc71232554ad0abebc4b0b31cda
Data source: yfinance 日度 ETF 收盘价。SHY、IEI、IEF 为组合标的，BIL 为超额夏普的现金参照（config.yaml:2-6，strategy.py:41-55、324-326、336）。FRED CSV 提供日度美国国债恒定期限收益率 DGS1、DGS2、DGS3、DGS5、DGS7、DGS10、DGS20，使用七列均有值的观测日，百分数除以 100 转为小数（config.yaml:7，strategy.py:19、66-84、327-329）。
Data fetched: 2026-11-27T08:02:32+00:00
Sample period: 实际回测收益区间为 2010-01-04 至 2025-12-31，按 ETF 交易日统计（strategy.py:261-263、340-341，logs/run_20261127_081201.log:8、15）。配置的回测区间为 2010-01-01 至 2025-12-31，FRED 拟合数据请求从 1994-01-01 开始（config.yaml:8-12，strategy.py:302-304、327-329）。   data_range (from output): {'start': '2010-01-04', 'end': '2025-12-31'}
IS/OOS split: 切分阈值为 2019-01-01。IS 为阈值前的收益日，实际为 2010-01-04 至 2018-12-31。OOS 包含阈值当日及之后的收益日，实际为 2019-01-02 至 2025-12-31（config.yaml:11，strategy.py:288-296，logs/run_20261127_081201.log:8）。IS 用截至信号日的月末因子快照扩窗拟合。OOS 信号使用仅以切分前月末快照拟合的冻结动态参数，并使用当期按发布滞后对齐的曲线因子。持仓跨切分日延续（strategy.py:209-214、224-255，README.md:22）。
Costs and slippage: 模型计入每一买卖方向 2 bps 费用和 3 bps 固定比例滑点（config.yaml:13-14，README.md:24、30）。执行日先按当日收益更新原持仓，再将组合当日总增长因子乘以 1 − 0.0005 × Σ|目标权重 − 成交前漂移权重|。首次买入亦计入两项摩擦，等权买入持有基准仅在首次买入时计入。优化时同样使用合计摩擦（strategy.py:164-184、195、218-240，README.md:17、24）。
Signal lag: 每月最后一个 ETF 交易日收盘生成目标，排到下一 ETF 交易日收盘模拟成交。成交日截至收盘的价格变动仍按原持仓计算，新权重从下一段收盘到收盘的收益开始生效（config.yaml:16，strategy.py:200-207、216-240、242-255，README.md:23）。FRED 曲线观测从观测日后第 3 个美国联邦工作日起可用于收盘信号（config.yaml:15，strategy.py:93-102、209-214，README.md:23）。
Price adjustment: yfinance auto_adjust=True, dividend- and split-adjusted close (total return)
Survivorship bias: 使用事后选定且仍存续的固定 ETF 组合 SHY、IEI、IEF，未纳入已清盘或被更换的基金，存在存续及标的选择偏差（config.yaml:2-5，README.md:31）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.014129581800406577, ann_vol=0.01706471076091734, sharpe=0.83074989266368, sharpe_excess=0.6883606643804934, max_drawdown=-0.023799147121614128, total_return=0.13434285173424576, turnover=5.0, n_trades=5
    oos: cagr=0.025438581655849113, ann_vol=0.04614513389499357, sharpe=0.5674502582029434, sharpe_excess=0.019474279220779312, max_drawdown=-0.10127065477397768, total_return=0.1917753697832536, turnover=8.000000000000004, n_trades=8
    full: cagr=0.01906043687828962, ann_vol=0.03309057718214997, sharpe=0.5871387515970263, sharpe_excess=0.21303889481208527, max_drawdown=-0.10127065477397845, total_return=0.3518818715865677, turnover=13.000000000000004, n_trades=13
    benchmark equal_weight_buy_and_hold full: cagr=0.02135127714224816, ann_vol=0.040043999614059154, sharpe=0.5476165052139831, sharpe_excess=0.23848929051899345, max_drawdown=-0.15452597448022742, total_return=0.40123477408462005, turnover=1.0, n_trades=3
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 474_multiperiod_bond_portfolio_optimization\nuniverse:\n  SHY: {yield_maturity_years: 2, duration_years: 1.8}\n  IEI: {yield_maturity_years: 5, duration_years: 4.5}\n  IEF: {yield_maturity_years: 10, duration_years: 7.5}\ncash_proxy: BIL\nfred_series_years: {DGS1: 1, DGS2: 2, DGS3: 3, DGS5: 5, DGS7: 7, DGS10: 10, DGS20: 20}\ndates:\n  fit_start: 1994-01-01\n  start: 2010-01-01\n  oos_start: 2019-01-01\n  end: 2025-12-31\ncosts_bps: 2\nslippage_bps: 3\nfred_publication_lag_business_days: 3\nexecution_lag_sessions: 1\nmodel:\n  decay_per_year: 0.7308\n  horizon_months: 6\n  yield_grid_bps: 40\n  max_states_per_month: 80\n  max_successors: 25\n  simulation_paths: 10000\n  simulation_seed: 474\n  min_fit_months: 60\n  # Weights follow the ticker order above.\n  actions:\n    - [1, 0, 0]\n    - [0, 1, 0]\n    - [0, 0, 1]\n    - [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]\n    - [0.2, 0.3, 0.5]\n    - [0.5, 0.3, 0.2]\n'
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
"""Research backtest of cost-aware Treasury ETF allocation."""

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import requests
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True, dividend- and split-adjusted close (total return)"
PARAM_COMBOS_TRIED = 1
SESSIONS_PER_YEAR = 252


def cached_frame(path, fetch):
    if path.exists():
        frame = pd.read_csv(path, index_col=0, parse_dates=True, float_precision="round_trip")
    else:
        frame = fetch()
        with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temp_path = Path(stream.name)
        try:
            frame.to_csv(temp_path, index_label="date")
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    fetched_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return frame, fetched_at


def load_prices(tickers, start, end):
    key = "_".join(tickers)
    path = CACHE / f"yf_adjusted_{key}_{start.date()}_{end.date()}.csv"

    def fetch():
        with TemporaryDirectory(dir=CACHE) as session_path:
            yf.set_tz_cache_location(session_path)
            download = yf.download(
                tickers, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                auto_adjust=True, progress=False, threads=False, timeout=15,
            )
        prices = download["Close"].reindex(columns=tickers).sort_index()
        prices.index = prices.index.tz_localize(None).normalize()
        if prices.empty or not prices.index.is_unique:
            raise ValueError("Expected unique daily adjusted ETF closes.")
        values = prices.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("Missing or invalid adjusted ETF closes.")
        if prices.index[-1] < end - pd.Timedelta(days=7):
            raise ValueError("Adjusted ETF closes end before the requested sample.")
        return prices

    return cached_frame(path, fetch)


def load_yields(series, start, end):
    ids = ",".join(series)
    path = CACHE / f"fred_{'_'.join(series)}_{start.date()}_{end.date()}.csv"

    def fetch():
        response = requests.get(
            FRED_URL, params={"id": ids, "cosd": str(start.date()), "coed": str(end.date())},
            timeout=25,
        )
        response.raise_for_status()
        raw = pd.read_csv(BytesIO(response.content), index_col=0, parse_dates=True, na_values=".")
        yields = raw.loc[:, series].sort_index().loc[start:end].dropna().astype(float) / 100.0
        if yields.empty or not yields.index.is_unique or not np.isfinite(yields).all().all():
            raise ValueError("Expected complete daily FRED yield-curve observations.")
        if yields.index[-1] < end - pd.Timedelta(days=10):
            raise ValueError("FRED yield observations end before the requested sample.")
        return yields

    return cached_frame(path, fetch)


def dns_loadings(maturities, decay):
    scaled = decay * np.asarray(maturities)
    slope = -np.expm1(-scaled) / scaled
    return np.column_stack((np.ones_like(scaled), slope, slope - np.exp(-scaled)))


def available_factors(yields, maturity_years, decay, publication_lag):
    board_calendar = pd.offsets.CustomBusinessDay(calendar=USFederalHolidayCalendar())
    calendar = pd.date_range(
        yields.index[0], yields.index[-1] + publication_lag * board_calendar, freq=board_calendar,
    )
    release_rows = calendar.searchsorted(yields.index, side="right") + publication_lag - 1
    loadings = dns_loadings(maturity_years, decay)
    coefficients = np.linalg.lstsq(loadings, yields.to_numpy().T, rcond=None)[0].T
    factors = pd.DataFrame(coefficients, index=calendar[release_rows])
    return factors.loc[~factors.index.duplicated(keep="last")]


def fit_dynamics(history, min_months):
    if len(history) < min_months or not np.isfinite(history).all():
        raise ValueError("Insufficient monthly FRED observations for fitting factor dynamics.")
    previous, current = history[:-1], history[1:]
    centered = previous - previous.mean(axis=0)
    variation = (centered ** 2).sum(axis=0)
    if (variation == 0).any():
        raise ValueError("Monthly FRED factor observations must vary to estimate dynamics.")
    slope = (centered * (current - current.mean(axis=0))).sum(axis=0) / variation
    intercept = current.mean(axis=0) - slope * previous.mean(axis=0)
    sigma = (current - intercept - slope * previous).std(axis=0, ddof=2)
    if not np.isfinite(sigma).all():
        raise ValueError("Could not estimate diagonal factor dynamics.")
    return intercept, slope, sigma


def discretize(yields, grid_step, state_limit):
    # Keep frequent bins and map the rest to the nearest retained bin.
    bins, labels, counts = np.unique(
        np.rint(yields / grid_step).astype(np.int64), axis=0,
        return_inverse=True, return_counts=True,
    )
    kept = np.argsort(-counts, kind="stable")[:state_limit]
    distances = ((bins[:, None, :] - bins[kept][None, :, :]) ** 2).sum(axis=2)
    labels = distances.argmin(axis=1)[labels]
    counts = np.bincount(labels, minlength=len(kept))
    representatives = np.column_stack([
        np.bincount(labels, weights=column, minlength=len(kept)) / counts
        for column in yields.T
    ])
    return labels, representatives


def markov_steps(initial, dynamics, innovations, loadings, durations, model):
    intercept, diagonal, sigma = dynamics
    factors = initial
    previous_states = (initial @ loadings.T)[None, :]
    previous_labels = np.zeros(innovations.shape[1], dtype=int)
    grid_step = model["yield_grid_bps"] / 10_000
    steps = []
    for noise in innovations:
        factors = intercept + diagonal * factors + sigma * noise
        labels, states = discretize(factors @ loadings.T, grid_step, model["max_states_per_month"])
        counts = np.bincount(
            previous_labels * len(states) + labels, minlength=len(previous_states) * len(states),
        ).reshape(len(previous_states), len(states))
        successors = np.argsort(-counts, axis=1, kind="stable")[:, :model["max_successors"]]
        transition = np.zeros_like(counts, dtype=float)
        np.put_along_axis(transition, successors, np.take_along_axis(counts, successors, axis=1), axis=1)
        transition /= transition.sum(axis=1, keepdims=True)

        # Monthly carry and constant-duration sensitivity approximate ETF growth.
        old_yields = previous_states[:, None, :]
        gross = np.exp(old_yields / 12.0 - durations * (states[None, :, :] - old_yields))
        steps.append((transition, gross))
        previous_labels, previous_states = labels, states
    return steps


def cost_multiplier(weights, targets, friction):
    return 1.0 - friction * np.abs(weights - targets).sum(axis=-1)


def choose_target(steps, positions, actions, friction):
    # For rho=1, the Bellman value is the terminal growth factor per unit invested.
    continuation = 1.0
    for stage in range(len(steps) - 1, -1, -1):
        transition, asset_growth = steps[stage]
        action_growth = asset_growth @ actions.T
        expected = (transition[:, :, None] * action_growth * continuation).sum(axis=1)
        if stage == 0:
            scores = cost_multiplier(positions, actions, friction) * expected[0]
            return actions[scores.argmax()]

        # Index by previous yield state, current state and previous action.
        previous_growth = steps[stage - 1][1]
        drifted = previous_growth[:, :, None, :] * actions
        drifted /= drifted.sum(axis=-1, keepdims=True)
        after_cost = cost_multiplier(drifted[:, :, :, None, :], actions, friction)
        continuation = (after_cost * expected[None, :, None, :]).max(axis=-1)


def simulate(prices, factors, cfg, start, split):
    dates = prices.index
    tickers = list(cfg["universe"])
    model = cfg["model"]
    actions = np.asarray(model["actions"], dtype=float)
    durations = np.array([asset["duration_years"] for asset in cfg["universe"].values()])
    maturities = [asset["yield_maturity_years"] for asset in cfg["universe"].values()]
    loadings = dns_loadings(maturities, model["decay_per_year"])
    friction = (cfg["costs_bps"] + cfg["slippage_bps"]) / 10_000
    innovations = np.random.default_rng(model["simulation_seed"]).standard_normal(
        (model["horizon_months"], model["simulation_paths"], factors.shape[1]),
    )

    # Schedule each month-end close to execute at a later close.
    last_session = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")).last()
    lag = cfg["execution_lag_sessions"]
    first_row = dates.searchsorted(start)
    schedule = {i: i + lag for i in last_session if first_row <= i + lag < len(dates)}
    if not schedule:
        raise ValueError("No delayed monthly execution falls inside the requested sample.")
    first_fill = min(schedule.values())

    monthly_dates = pd.date_range(cfg["dates"]["fit_start"], dates[-1], freq=pd.offsets.MonthEnd())
    monthly = factors.reindex(monthly_dates, method="ffill", tolerance=pd.Timedelta(days=14))
    signal_dates = dates[list(schedule)]
    signals = factors.reindex(signal_dates, method="ffill", tolerance=pd.Timedelta(days=14))
    if signals.isna().any().any():
        raise ValueError("A signal close lacks a recent published Treasury yield curve.")

    gross = (prices[tickers] / prices[tickers].shift()).to_numpy(copy=True)
    gross[0] = 1.0
    positions = np.zeros((2, len(tickers)))
    equal_weights = np.full(len(tickers), 1.0 / len(tickers))
    daily_results = np.zeros((len(dates), 2, 3))
    pending = {}
    frozen_dynamics = None

    for row in range(min(first_row, min(schedule)), len(dates)):
        changes = positions * gross[row]
        growth = changes.sum(axis=1) + 1.0 - positions.sum(axis=1)
        positions = changes / growth[:, None]
        turnover = np.zeros(2)
        trade_count = np.zeros(2, dtype=int)
        if row in pending:
            targets = positions.copy()
            targets[0] = pending.pop(row)
            if row == first_fill:
                targets[1] = equal_weights
            differences = np.abs(targets - positions)
            differences[differences <= 1e-10] = 0.0
            turnover = differences.sum(axis=1)
            trade_count = np.count_nonzero(differences, axis=1)
            growth *= 1.0 - friction * turnover
            positions = targets

        if row in schedule:
            date = dates[row]
            if date < split:
                dynamics = fit_dynamics(monthly.loc[:date].to_numpy(), model["min_fit_months"])
            else:
                if frozen_dynamics is None:
                    frozen_dynamics = fit_dynamics(
                        monthly.loc[monthly.index < split].to_numpy(), model["min_fit_months"],
                    )
                dynamics = frozen_dynamics
            steps = markov_steps(
                signals.loc[date].to_numpy(), dynamics, innovations, loadings, durations, model,
            )
            pending[schedule[row]] = choose_target(steps, positions[0], actions, friction)

        daily_results[row, :, 0] = growth - 1.0
        daily_results[row, :, 1] = turnover
        daily_results[row, :, 2] = trade_count

    columns = ["return", "turnover", "n_trades"]
    sample_dates = dates[first_row:]
    return [pd.DataFrame(daily_results[first_row:, i], index=sample_dates, columns=columns) for i in range(2)]


def sharpe(returns):
    volatility = returns.std(ddof=1)
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility) if volatility > 0 else None


def metrics(daily_results, cash_returns):
    returns = daily_results["return"]
    cumulative_growth = (1.0 + returns).cumprod()
    drawdown = cumulative_growth / cumulative_growth.cummax().clip(lower=1.0) - 1.0
    terminal_growth = cumulative_growth.iloc[-1]
    return {
        "cagr": float(terminal_growth ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - cash_returns),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(terminal_growth - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, cash_returns, split):
    before_split = daily_results.index < split
    if min(before_split.sum(), (~before_split).sum()) < 2:
        raise ValueError("Both IS and OOS must contain at least two return dates.")
    return {
        "is": metrics(daily_results.loc[before_split], cash_returns.loc[before_split]),
        "oos": metrics(daily_results.loc[~before_split], cash_returns.loc[~before_split]),
        "full": metrics(daily_results, cash_returns),
    }


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    fit_start, start, split, end = (
        pd.Timestamp(cfg["dates"][key]) for key in ("fit_start", "start", "oos_start", "end")
    )
    if not fit_start < start < split <= end:
        raise ValueError("Require fit_start < start < oos_start <= end.")
    lag = cfg["execution_lag_sessions"]
    publication_lag = cfg["fred_publication_lag_business_days"]
    if lag < 1 or publication_lag < 2:
        raise ValueError("Require at least one ETF session and two federal business days of lag.")
    if min(cfg["costs_bps"], cfg["slippage_bps"]) < 0 or cfg["costs_bps"] + cfg["slippage_bps"] >= 5_000:
        raise ValueError("Invalid proportional cost or slippage.")
    model = cfg["model"]
    actions = np.asarray(model["actions"], dtype=float)
    if (actions.ndim != 2 or actions.shape[1] != len(cfg["universe"])
            or (actions < 0).any() or not np.allclose(actions.sum(axis=1), 1.0)):
        raise ValueError("Each action must specify long-only weights totaling one, in universe order.")
    positive_settings = ("decay_per_year", "horizon_months", "yield_grid_bps", "max_states_per_month",
                         "max_successors", "simulation_paths")
    if any(model[key] <= 0 for key in positive_settings) or model["min_fit_months"] < 4:
        raise ValueError("Model scales and simulation limits must be positive, with at least four fit months.")

    CACHE.mkdir(exist_ok=True)
    tickers = list(dict.fromkeys([*cfg["universe"], cfg["cash_proxy"]]))
    price_start = start - pd.offsets.MonthBegin(2) - pd.Timedelta(days=2 * lag)
    prices, prices_time = load_prices(tickers, price_start, end)
    fred_series = cfg["fred_series_years"]
    yields, yields_time = load_yields(list(fred_series), fit_start, end)
    factors = available_factors(yields, list(fred_series.values()), model["decay_per_year"], publication_lag)
    print(f"DATA_FETCHED_AT: {max(prices_time, yields_time).isoformat(timespec='seconds')}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"PARAM_COMBOS_TRIED: {PARAM_COMBOS_TRIED}")

    daily_results, benchmark = simulate(prices, factors, cfg, start, split)
    dates = daily_results.index
    cash_returns = prices[cfg["cash_proxy"]].pct_change(fill_method=None).loc[dates]
    strategy_metrics = period_metrics(daily_results, cash_returns, split)
    benchmark_name = "equal_weight_buy_and_hold"
    benchmark_metrics = period_metrics(benchmark, cash_returns, split)
    print(f"SAMPLE_SPLIT: IS {dates[0].date()} to {dates[dates < split][-1].date()}; "
          f"OOS {dates[dates >= split][0].date()} to {dates[-1].date()}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(strategy_metrics[period], allow_nan=False)}")
        print(f"BENCHMARK {benchmark_name} {period.upper()}: "
              f"{json.dumps(benchmark_metrics[period], allow_nan=False)}")
    result = {
        "strategy": cfg["strategy"],
        "data_range": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
        **strategy_metrics,
        "benchmark": {benchmark_name: benchmark_metrics},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

"""130_ml_risk_budget_allocation
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/130_ml_risk_budget_allocation.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:03 Europe/Oslo, exit 0, output sha256 226555ed7ab12181b895552ba2eed3f30ab7aebf609d8184d450ee15f83de3c2
Data source: 通过 yfinance 获取 Yahoo Finance 免费日频行情。八只 ETF 为 SPY、IWM、EFA、IEF、TLT、GLD、DBC、HYG，使用开盘价与收盘价。另取 ^VIX 日收盘水平。程序优先使用 data_cache/ 下的行情 CSV，本次日志显示走了 CSV 读取路径。（config.yaml:11-15；README.md:19；strategy.py:19-40、60；logs/run_20261125_103139.log:1-2）
Data fetched: 2026-11-27T11:03:37+00:00
Sample period: 行情请求区间为 2007-05-01 至 2025-12-31，包含终止日，2007-05-01 为配置的预热起点。本次日志记录的实际估值交易日范围为 2011-01-03 至 2025-12-31。（config.yaml:3-9；strategy.py:33-36、226-233、248-250、263-264；logs/run_20261125_103139.log:3）   data_range (from output): 2011-01-03 to 2025-12-31
IS/OOS split: 按时间连续划分。样本内为 2011-01-01 至 2016-12-31，样本外为 2017-01-01 至 2025-12-31，均含端点。模拟持仓跨边界延续，随后按日期切分报告。样本外沿用固定规则，每六个月使用当时已成熟的标签重训。（config.yaml:4-9、21-22；strategy.py:102-112、222-233、256-262；README.md:27）
Costs and slippage: 单边 5 bps，每次开盘调仓按各 ETF 买入与卖出金额的绝对值之和收费，首次建仓也计费。程序用扣费后净值确定目标份额。滑点未建模，成交价直接使用复权开盘价。（config.yaml:35；strategy.py:138-158、244；README.md:22、36；logs/run_20261125_103139.log:3）
Signal lag: 在月末最后一个交易日 t 收盘后，用截至 t 的 ETF 收盘特征与 ^VIX 收盘水平计算权重，在下一交易日（次月首个交易日）开盘模拟成交。（strategy.py:63-76、94-96、103-122、144-162；README.md:20、22）
Price adjustment: Yahoo Finance auto-adjusted ETF opens/closes and BIL close; VIX index close
Survivorship bias: 研究使用固定八只 ETF，存在基金存续和事后选基偏差。股票 ETF 使用基金历史价格，未用当前成分股名单回填历史。（config.yaml:11-14；README.md:9、13-15、33；strategy.py:235-248）
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.05687173510025234, ann_vol=0.06368240270312014, sharpe=0.9005067760073175, sharpe_excess=0.9059192057910772, max_drawdown=-0.10349565057078558, total_return=0.3929741170976824, turnover=8.64686831606922, n_trades=576
    oos: cagr=0.05458741801055522, ann_vol=0.07650145178245615, sharpe=0.73308520382926, sharpe_excess=0.4406827782410853, max_drawdown=-0.21780208969643167, total_return=0.611364104411966, turnover=14.856624376331958, n_trades=864
    full: cagr=0.05550127833165952, ann_vol=0.07163676620942787, sharpe=0.7899327218404332, sharpe_excess=0.6047238009283457, max_drawdown=-0.21780208969643033, total_return=1.244588490666139, turnover=23.50349269240118, n_trades=1440
    benchmark: cagr=0.060890242764187086, ann_vol=0.08477474224974597, sharpe=0.7397676058809167, sharpe_excess=0.5831379420877276, max_drawdown=-0.18769557442653106, total_return=1.4223778787867873, turnover=4.994656210395334, n_trades=1440
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 130_ml_risk_budget_allocation\n\ndata_start: 2007-05-01\nin_sample:\n  start: 2011-01-01\n  end: 2016-12-31\nout_of_sample:\n  start: 2017-01-01\n  end: 2025-12-31\n\nuniverse:\n  equity: [SPY, IWM, EFA]\n  treasury: [IEF, TLT]\n  other: [GLD, DBC, HYG]\nvix_ticker: '^VIX'\ncash_ticker: BIL\n\nmodel:\n  volatility_windows: [5, 21, 63]\n  return_window: 21\n  target_sessions: 21\n  training_weeks: 156\n  refit_months: 6\n  gradient_boosting:\n    n_estimators: 100\n    learning_rate: 0.1\n    max_depth: 3\n    random_state: 130\n\nallocation:\n  vix_threshold: 25\n  trailing_volatility_window: 21\n  baseline_budget: {equity: 0.40, treasury: 0.30, other: 0.30}\n  stress_budget: {equity: 0.20, treasury: 0.50, other: 0.30}\n\ncost_bps: 5\nslippage_bps: 5\ncash_buffer: 0.05\nsessions_per_year: 252\n"
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
"""Walk-forward volatility forecasts and monthly ETF allocation for C31."""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parent
PORTFOLIOS = ("ml_adaptive", "equal_weight", "trailing_adaptive", "ml_static")
FETCH_TIMESTAMP_PREFIX = "# DATA_FETCHED_AT: "


def us_equity_sessions(start, end):
    holidays = AbstractHolidayCalendar(rules=[
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]).holidays(start, end)
    # Full-day closures within the research window, in addition to regular holidays.
    special_closures = pd.to_datetime(["2007-01-02", "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(holidays.union(special_closures))


def read_snapshot(source):
    timestamp = source.readline().strip()
    if not timestamp.startswith(FETCH_TIMESTAMP_PREFIX):
        raise ValueError("The Yahoo Finance snapshot is missing its fetch timestamp.")
    fetched_at = datetime.fromisoformat(timestamp.removeprefix(FETCH_TIMESTAMP_PREFIX))
    return pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True), fetched_at


def load_prices(tickers, vix_ticker, cash_ticker, start, end, evaluation_start):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    symbols = tickers + [vix_ticker, cash_ticker]
    if any(not symbol or not set(symbol) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.^-=")
           for symbol in symbols):
        raise ValueError("Use uppercase Yahoo Finance ticker symbols in config.yaml.")
    path = cache_dir / f"yahoo_adjusted_{start.date()}_{end.date()}_{'_'.join(symbols)}.csv"
    cached = path.exists()
    if cached:
        with path.open() as source:
            panel, fetched_at = read_snapshot(source)
    else:
        with TemporaryDirectory(dir=cache_dir) as session_cache:
            yf.set_tz_cache_location(session_cache)
            panel = yf.download(
                symbols, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval="1d", auto_adjust=True, progress=False, timeout=20,
            )
        fetched_at = datetime.now(timezone.utc)

    columns = [("Open", symbol) for symbol in tickers]
    columns += [("Close", symbol) for symbol in tickers + [vix_ticker]]
    panel = panel.reindex(columns=pd.MultiIndex.from_tuples(columns + [("Close", cash_ticker)])).sort_index()
    if panel.empty or not isinstance(panel.index, pd.DatetimeIndex) or panel.index.has_duplicates:
        raise ValueError("Yahoo Finance returned empty, invalid or duplicate price dates.")
    expected = us_equity_sessions(start, end)
    if not panel.index.equals(expected):
        missing = expected.difference(panel.index).strftime("%Y-%m-%d").tolist()
        extra = panel.index.difference(expected).strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Yahoo Finance dates differ from the U.S. equity calendar. Missing: {missing}; extra: {extra}.")
    required_prices = panel[columns]
    bad_prices = ~(np.isfinite(required_prices) & required_prices.gt(0))
    if bad_prices.any().any():
        counts = bad_prices.sum()
        raise ValueError(f"Missing or invalid prices by field/ticker:\n{counts[counts.gt(0)]}")
    first_report_index = expected.searchsorted(evaluation_start)
    if not 0 < first_report_index < len(expected):
        raise ValueError("A price session before the reporting window is required.")
    cash_prices = panel["Close", cash_ticker].iloc[first_report_index - 1:]
    if not (np.isfinite(cash_prices) & cash_prices.gt(0)).all():
        raise ValueError(f"Missing or invalid {cash_ticker} closes for the cash comparison.")

    if not cached:
        with NamedTemporaryFile(mode="w+", dir=cache_dir, delete=False) as output:
            temporary_path = Path(output.name)
            try:
                output.write(FETCH_TIMESTAMP_PREFIX + fetched_at.isoformat(timespec="seconds") + "\n")
                panel.to_csv(output, index_label="date")
                # Use the same numeric parsing for the fresh run and subsequent runs.
                output.seek(0)
                panel, fetched_at = read_snapshot(output)
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
    cash_returns = panel["Close", cash_ticker].iloc[first_report_index - 1:].pct_change().iloc[1:]
    return panel["Open"], panel["Close"][tickers], panel["Close"][vix_ticker], cash_returns, fetched_at


def allocation_targets(opens, closes, vix, start, config):
    model_options = config["model"]
    annualization = np.sqrt(config["sessions_per_year"])
    daily = closes.pct_change()
    trailing_vol = {
        window: daily.rolling(window).std() * annualization
        for window in model_options["volatility_windows"]
    }
    allocation = config["allocation"]
    baseline_vol = daily.rolling(allocation["trailing_volatility_window"]).std() * annualization
    features = {f"rv_{window}": values.stack() for window, values in trailing_vol.items()}
    features["return"] = closes.pct_change(model_options["return_window"]).stack()
    features["vix"] = pd.DataFrame({symbol: vix for symbol in closes}).stack()
    features = pd.DataFrame(features).dropna()

    horizon = model_options["target_sessions"]
    # The return at open t+2 is the first one after opening the position at t+1.
    # The 21st return and the full label are observable at open t+22.
    future_vol = opens.pct_change().rolling(horizon).std().shift(-horizon - 1) * annualization
    log_target = np.log(future_vol.where(future_vol.gt(0)))
    maturity = opens.index.to_series().shift(-horizon - 1)
    dates = opens.index.to_series()
    weekly_dates = dates.groupby(opens.index.to_period("W-FRI")).last()
    feature_dates = features.groupby(level=0).size()
    complete_dates = feature_dates.index[feature_dates.eq(len(closes.columns))]
    weekly_dates = pd.DatetimeIndex(weekly_dates).intersection(complete_dates)
    weekly_dates = weekly_dates.intersection(log_target.dropna().index)
    weekly_targets = log_target.loc[weekly_dates].stack().rename("target")
    training = features.join(weekly_targets, how="inner")
    label_date = maturity.loc[weekly_dates]

    # Use month-end ETF and VIX closes after both series have been published.
    executions = dates.shift(-1).groupby(opens.index.to_period("M")).tail(1).dropna()
    executions = executions[executions.ge(start)]
    group = pd.Series({symbol: name for name, symbols in config["universe"].items()
                       for symbol in symbols})
    targets = {name: {} for name in PORTFOLIOS}
    previous_refit = None
    model = None
    refit_months = model_options["refit_months"]
    for signal, execution in executions.items():
        refit = (execution.year * 12 + execution.month - 1) // refit_months
        if refit != previous_refit:
            eligible = label_date[label_date.le(signal)].tail(model_options["training_weeks"])
            if len(eligible) < model_options["training_weeks"]:
                raise ValueError(f"Insufficient completed weekly labels at {signal.date()}; move data_start earlier.")
            sample = training.loc[eligible.index]
            model = GradientBoostingRegressor(**model_options["gradient_boosting"])
            model.fit(sample[features.columns], sample["target"])
            previous_refit = refit

        snapshot = features.loc[signal].reindex(closes.columns)
        forecast = pd.Series(np.exp(model.predict(snapshot)), index=snapshot.index)
        realized = baseline_vol.loc[signal]
        baseline = allocation["baseline_budget"]
        adaptive = allocation["stress_budget"] if vix.loc[signal] >= allocation["vix_threshold"] else baseline
        targets["ml_adaptive"][execution] = budget_weights(forecast, group, adaptive)
        targets["trailing_adaptive"][execution] = budget_weights(realized, group, adaptive)
        targets["ml_static"][execution] = budget_weights(forecast, group, baseline)
        targets["equal_weight"][execution] = pd.Series(1 / len(closes.columns), index=closes.columns)

    return {
        name: pd.DataFrame.from_dict(weights, orient="index", columns=closes.columns)
        for name, weights in targets.items()
    }


def budget_weights(volatility, groups, budgets):
    if not (np.isfinite(volatility) & volatility.gt(0)).all():
        raise ValueError("All input volatilities must be positive and finite.")
    inverse_vol = 1 / volatility
    group = groups.reindex(volatility.index)
    return inverse_vol / inverse_vol.groupby(group).transform("sum") * group.map(budgets)


def simulate(opens, closes, targets, start, fee, slippage, cash_buffer):
    """Size from the previous close and fill unchanged quantities at the next open."""
    opens = opens.loc[start:]
    sizing_closes = closes.shift(1).loc[opens.index]
    closes = closes.loc[opens.index]
    by_date = {date: weights for date, weights in zip(targets.index, targets.to_numpy())}
    positions = np.zeros(len(opens.columns))
    cash = previous_value = 1.0
    rows = []
    for date, reference_prices, open_prices, close_prices in zip(
        opens.index, sizing_closes.to_numpy(), opens.to_numpy(), closes.to_numpy()
    ):
        turnover = 0.0
        trades = 0
        if date in by_date:
            weights = by_date[date]
            desired_positions = (1 - cash_buffer) * previous_value * weights / reference_prices
            share_changes = desired_positions - positions

            value_before_fill = cash + positions @ open_prices
            fill_prices = open_prices * (1 + slippage * np.sign(share_changes))
            trade_values = share_changes * fill_prices
            gross_traded = np.abs(trade_values).sum()
            cash_after_fill = cash - trade_values.sum() - fee * gross_traded
            if not np.isfinite(cash_after_fill) or cash_after_fill < 0:
                raise ValueError(f"Cash shortfall at {date.date()} open with fixed share quantities: {cash_after_fill}.")
            positions = desired_positions
            cash = cash_after_fill
            turnover = gross_traded / value_before_fill
            trades = np.count_nonzero(share_changes)

        value = cash + positions @ close_prices
        rows.append((value / previous_value - 1, turnover, trades))
        previous_value = value
    return pd.DataFrame(rows, index=opens.index, columns=["return", "turnover", "n_trades"])


def metrics(daily, cash_returns, sessions_per_year):
    returns = daily["return"]
    excess_returns = returns - cash_returns.loc[returns.index]
    wealth = (1 + returns).cumprod().to_numpy()
    volatility = returns.std(ddof=1)
    excess_volatility = excess_returns.std(ddof=1)
    annualization = np.sqrt(sessions_per_year)
    values = {
        "cagr": wealth[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": volatility * annualization,
        "sharpe": returns.mean() / volatility * annualization if volatility > 0 else None,
        "sharpe_excess": excess_returns.mean() / excess_volatility * annualization if excess_volatility > 0 else None,
        "max_drawdown": (wealth / np.maximum.accumulate(np.r_[1.0, wealth])[1:] - 1).min(),
        "total_return": wealth[-1] - 1,
        "turnover": daily["turnover"].sum(),
        "n_trades": int(daily["n_trades"].sum()),
        "sessions": len(returns),
    }
    return {
        key: value if isinstance(value, int) else float(value) if value is not None and np.isfinite(value) else None
        for key, value in values.items()
    }


def print_results(config, data_range, reports, fetched_at):
    main_report = reports["ml_adaptive"]
    equal_weight = reports["equal_weight"]

    def display(value, spec):
        return "n/a" if value is None else format(value, spec)

    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()}")
    print(f"PRICE_ADJUSTMENT: Yahoo Finance auto-adjusted ETF opens/closes and {config['cash_ticker']} close; VIX index close")
    print(f"{config['strategy']} | valuation dates {data_range} | {config['cost_bps']} bps fee and "
          f"{config['slippage_bps']} bps slippage per side | {config['cash_buffer']:.0%} cash buffer")
    print(f"{'Portfolio':<19} {'Period':>6} {'CAGR':>9} {'Ann vol':>9} {'Sharpe':>8} "
          f"{'Sharpe-' + config['cash_ticker']:>10} {'Max DD':>9} {'Total':>9} {'Turnover':>10} {'Trades':>7}")
    for name, report in reports.items():
        for period, row in report.items():
            print(f"{name:<19} {period.upper():>6} {display(row['cagr'], '.2%'):>9} "
                  f"{display(row['ann_vol'], '.2%'):>9} {display(row['sharpe'], '.2f'):>8} "
                  f"{display(row['sharpe_excess'], '.2f'):>10} "
                  f"{display(row['max_drawdown'], '.2%'):>9} {display(row['total_return'], '.2%'):>9} "
                  f"{display(row['turnover'], '.2f'):>9}x {row['n_trades']:>7}")

    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **main_report,
        "benchmark": {
            "name": "equal_weight",
            **equal_weight["full"],
            **equal_weight,
            "comparisons": {name: reports[name] for name in ("trailing_adaptive", "ml_static")},
        },
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    periods = {
        short: tuple(pd.Timestamp(config[long][endpoint]) for endpoint in ("start", "end"))
        for short, long in (("is", "in_sample"), ("oos", "out_of_sample"))
    }
    start, is_end = periods["is"]
    oos_start, end = periods["oos"]
    data_start = pd.Timestamp(config["data_start"])
    if not data_start < start <= is_end < oos_start <= end:
        raise ValueError("Dates must be ordered from warm-up through non-overlapping IS and OOS.")
    if oos_start != is_end + pd.Timedelta(days=1):
        raise ValueError("IS and OOS date ranges must be contiguous.")
    periods["full"] = (start, end)

    groups = config["universe"]
    tickers = [symbol for symbols in groups.values() for symbol in symbols]
    vix_ticker, cash_ticker = config["vix_ticker"], config["cash_ticker"]
    if not tickers or len(set(tickers + [vix_ticker, cash_ticker])) != len(tickers) + 2:
        raise ValueError("The allocation ETFs, VIX, and cash comparison must have distinct tickers.")
    for key in ("baseline_budget", "stress_budget"):
        budgets = config["allocation"][key]
        if (budgets.keys() != groups.keys() or not all(groups.values())
                or min(budgets.values()) < 0 or not np.isclose(sum(budgets.values()), 1)):
            raise ValueError(f"{key} must assign all capital across the nonempty ETF groups.")
    fee = config["cost_bps"] / 10_000
    slippage = config["slippage_bps"] / 10_000
    cash_buffer = config["cash_buffer"]
    if not (0 <= fee < 1 and 0 <= slippage < 1 and 0 < cash_buffer < 1):
        raise ValueError("Cost and slippage rates must be in [0, 1), and the cash buffer in (0, 1).")

    opens, closes, vix, cash_returns, fetched_at = load_prices(
        tickers, vix_ticker, cash_ticker, data_start, end, start
    )
    targets = allocation_targets(opens, closes, vix, start, config)
    if any(opens.loc[first:last].empty for first, last in periods.values()):
        raise ValueError("Each configured reporting period must contain trading sessions.")

    reports = {}
    for name, weights in targets.items():
        daily_results = simulate(opens, closes, weights, start, fee, slippage, cash_buffer)
        reports[name] = {
            period: metrics(daily_results.loc[first:last], cash_returns, config["sessions_per_year"])
            for period, (first, last) in periods.items()
        }
    data_range = f"{cash_returns.index[0].date()} to {cash_returns.index[-1].date()}"
    print_results(config, data_range, reports, fetched_at)


if __name__ == "__main__":
    main()

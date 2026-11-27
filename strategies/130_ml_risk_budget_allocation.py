"""130_ml_risk_budget_allocation
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/130_ml_risk_budget_allocation.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit ad594150a0142009ae8f6ffab0995de08b2f8304 on 2026-11-26 13:50 Europe/Oslo, exit 0, output sha256 e4fa0e47a802137bc5f79b7810d844c8aca4f1a33db9a6c19a72afd4a6301dbd
Data source: 通过 yfinance 获取 Yahoo Finance 免费日频行情。八只 ETF 为 SPY、IWM、EFA、IEF、TLT、GLD、DBC、HYG，使用开盘价与收盘价。另取 ^VIX 日收盘水平。程序优先使用 data_cache/ 下的行情 CSV，本次日志显示走了 CSV 读取路径。
Data fetched: 2026-11-26 13:50 Europe/Oslo (run time)
Sample period: 行情请求区间为 2007-05-01 至 2025-12-31，包含终止日，2007-05-01 为配置的预热起点。本次日志记录的实际估值交易日范围为 2011-01-03 至 2025-12-31。 data_range (from output): 2011-01-03 to 2025-12-31
IS/OOS split: 按时间连续划分。样本内为 2011-01-01 至 2016-12-31，样本外为 2017-01-01 至 2025-12-31，均含端点。模拟持仓跨边界延续，随后按日期切分报告。样本外沿用固定规则，每六个月使用当时已成熟的标签重训。
Costs and slippage: 单边 5 bps，每次开盘调仓按各 ETF 买入与卖出金额的绝对值之和收费，首次建仓也计费。程序用扣费后净值确定目标份额。滑点未建模，成交价直接使用复权开盘价。
Signal lag: 在月末最后一个交易日 t 收盘后，用截至 t 的 ETF 收盘特征与 ^VIX 收盘水平计算权重，在下一交易日（次月首个交易日）开盘模拟成交。
Price adjustment: 下载时设置 auto_adjust=True，ETF 使用复权开盘价和收盘价，^VIX 使用收盘水平。历史复权数据可能修订。
Survivorship bias: 研究使用固定八只 ETF，存在基金存续和事后选基偏差。股票 ETF 使用基金历史价格，未用当前成分股名单回填历史。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.060246578012403296, ann_vol=0.06717378433618088, sharpe=0.9045744966384669, max_drawdown=-0.1084135871265347, total_return=0.41984074426360696, turnover=9.093535076654744, n_trades=576
    oos: cagr=0.05854973170810318, ann_vol=0.0805170245128774, sharpe=0.7470271327399245, max_drawdown=-0.225621810072425, total_return=0.6665296095794255, turnover=15.696748102457093, n_trades=864
    full: cagr=0.05922868384241897, ann_vol=0.0754499125324828, sharpe=0.8004610178300348, max_drawdown=-0.22562181007242366, total_return=1.3662066412026181, turnover=24.79028317911184, n_trades=1440
    benchmark: cagr=0.06435521561627122, ann_vol=0.08935695243658176, sharpe=0.742806374068546, max_drawdown=-0.19682673018115504, total_return=1.5435424961947066, turnover=5.405496454938333, n_trades=1440
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 130_ml_risk_budget_allocation\n\ndata_start: 2007-05-01\nin_sample:\n  start: 2011-01-01\n  end: 2016-12-31\nout_of_sample:\n  start: 2017-01-01\n  end: 2025-12-31\n\nuniverse:\n  equity: [SPY, IWM, EFA]\n  treasury: [IEF, TLT]\n  other: [GLD, DBC, HYG]\nvix_ticker: '^VIX'\n\nmodel:\n  volatility_windows: [5, 21, 63]\n  return_window: 21\n  target_sessions: 21\n  training_weeks: 156\n  refit_months: 6\n  gradient_boosting:\n    n_estimators: 100\n    learning_rate: 0.1\n    max_depth: 3\n    random_state: 130\n\nallocation:\n  vix_threshold: 25\n  trailing_volatility_window: 21\n  baseline_budget: {equity: 0.40, treasury: 0.30, other: 0.30}\n  stress_budget: {equity: 0.20, treasury: 0.50, other: 0.30}\n\ncost_bps: 5\nsessions_per_year: 252\n"
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
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from scipy.optimize import brentq
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parent
PORTFOLIOS = ("ml_adaptive", "equal_weight", "trailing_adaptive", "ml_static")


def load_prices(tickers, vix_ticker, start, end):
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    symbols = tickers + [vix_ticker]
    if any(not symbol or not set(symbol) <= set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.^-=")
           for symbol in symbols):
        raise ValueError("Use uppercase Yahoo Finance ticker symbols in config.yaml.")
    path = cache_dir / f"yahoo_adjusted_{start.date()}_{end.date()}_{'_'.join(symbols)}.csv"
    if path.exists():
        panel = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True,
                            float_precision="round_trip")
    else:
        with TemporaryDirectory(dir=cache_dir) as session_cache:
            yf.set_tz_cache_location(session_cache)
            panel = yf.download(
                symbols, start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                auto_adjust=True, progress=False, timeout=20,
            )

    columns = [("Open", symbol) for symbol in tickers]
    columns += [("Close", symbol) for symbol in symbols]
    panel = panel.reindex(columns=pd.MultiIndex.from_tuples(columns)).sort_index()
    if panel.empty or panel.index.has_duplicates:
        raise ValueError("Yahoo Finance returned empty or duplicate price dates.")
    bad_prices = ~(np.isfinite(panel) & panel.gt(0))
    if bad_prices.any().any():
        counts = bad_prices.sum()
        raise ValueError(f"Missing or invalid prices by field/ticker:\n{counts[counts.gt(0)]}")
    if panel.index[0] > start + pd.Timedelta(days=7) or panel.index[-1] < end - pd.Timedelta(days=7):
        raise ValueError("Yahoo Finance prices do not span the configured data window.")

    if not path.exists():
        with NamedTemporaryFile(mode="w", dir=cache_dir, delete=False) as output:
            temporary_path = Path(output.name)
            try:
                panel.to_csv(output, index_label="date")
                output.flush()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)
    return panel["Open"], panel["Close"][tickers], panel["Close"][vix_ticker]


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

    # Targets are created on the last session of a month and booked at the next open.
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


def simulate(opens, closes, targets, fee):
    """Hold ETF shares between rebalances and value them at each close."""
    by_date = {date: weights for date, weights in zip(targets.index, targets.to_numpy())}
    holdings = np.zeros(len(opens.columns))
    cash = previous_nav = 1.0
    rows = []
    for date, open_prices, close_prices in zip(opens.index, opens.to_numpy(), closes.to_numpy()):
        turnover = 0.0
        trades = 0
        if date in by_date:
            weights = by_date[date]
            positions = holdings * open_prices
            equity = cash + positions.sum()

            def financing_error(nav):
                return nav + fee * np.abs(nav * weights - positions).sum() - equity

            net_equity = brentq(financing_error, 0.0, equity)
            trade_values = np.abs(net_equity * weights - positions)
            holdings = net_equity * weights / open_prices
            cash = 0.0
            turnover = trade_values.sum() / equity
            trades = np.count_nonzero(trade_values > equity * 1e-12)

        nav = cash + holdings @ close_prices
        rows.append((nav / previous_nav - 1, turnover, trades))
        previous_nav = nav
    return pd.DataFrame(rows, index=opens.index, columns=["return", "turnover", "n_trades"])


def metrics(daily, sessions_per_year):
    returns = daily["return"]
    wealth = (1 + returns).cumprod().to_numpy()
    volatility = returns.std(ddof=1)
    values = {
        "cagr": wealth[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": volatility * np.sqrt(sessions_per_year),
        "sharpe": returns.mean() / volatility * np.sqrt(sessions_per_year) if volatility > 0 else None,
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


def print_results(config, data_range, reports):
    main_report = reports["ml_adaptive"]
    equal_weight = reports["equal_weight"]

    def display(value, spec):
        return "n/a" if value is None else format(value, spec)

    print(f"{config['strategy']} | valuation dates {data_range} | {config['cost_bps']} bps per side")
    print(f"{'Portfolio':<19} {'Period':>6} {'CAGR':>9} {'Ann vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total':>9} {'Turnover':>10} {'Trades':>7}")
    for name, report in reports.items():
        for period, row in report.items():
            print(f"{name:<19} {period.upper():>6} {display(row['cagr'], '.2%'):>9} "
                  f"{display(row['ann_vol'], '.2%'):>9} {display(row['sharpe'], '.2f'):>8} "
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
    if not tickers or len(set(tickers)) != len(tickers) or config["vix_ticker"] in tickers:
        raise ValueError("The strategy requires distinct ETF tickers and a separate VIX series.")
    for key in ("baseline_budget", "stress_budget"):
        budgets = config["allocation"][key]
        if (budgets.keys() != groups.keys() or not all(groups.values())
                or min(budgets.values()) < 0 or not np.isclose(sum(budgets.values()), 1)):
            raise ValueError(f"{key} must assign all capital across the nonempty ETF groups.")
    fee = config["cost_bps"] / 10_000
    if not 0 <= fee < 1:
        raise ValueError("cost_bps must be nonnegative and less than 10000.")

    opens, closes, vix = load_prices(tickers, config["vix_ticker"], data_start, end)
    targets = allocation_targets(opens, closes, vix, start, config)
    opens, closes = opens.loc[start:end], closes.loc[start:end]
    if opens.empty or any(opens.loc[first:last].empty for first, last in periods.values()):
        raise ValueError("Each configured reporting period must contain trading sessions.")
    if targets["ml_adaptive"].empty:
        raise ValueError("The evaluation period contains no scheduled rebalances.")

    reports = {}
    for name, weights in targets.items():
        daily = simulate(opens, closes, weights, fee)
        reports[name] = {
            period: metrics(daily.loc[first:last], config["sessions_per_year"])
            for period, (first, last) in periods.items()
        }
    data_range = f"{opens.index[0].date()} to {opens.index[-1].date()}"
    print_results(config, data_range, reports)


if __name__ == "__main__":
    main()
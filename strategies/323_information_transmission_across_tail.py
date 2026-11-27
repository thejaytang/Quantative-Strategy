"""323_information_transmission_across_tail
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/323_information_transmission_across_tail.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2585fac6f6b194ec028771df2a48e498092fc067 on 2026-11-27 11:14 Europe/Oslo, exit 0, output sha256 ad8fd84c64a5b48269c65e69cf5705c97bc06f17457ecc72ffddfe168c108ee0
Data source: Yahoo Finance 公开历史行情网页的 USO、SPY、BIL 日线开盘价、收盘价及复权收盘价，以及 FRED 公开 CSV 的 OVXCLS 日度序列。优先读取同标的、同区间的本地 CSV 缓存。
Data fetched: 2026-11-27T11:14:36+01:00
Sample period: 行情及预热请求区间为 2008-01-01 至 2026-11-20。日志记录的实际组合收益日期为 2014-01-02 至 2026-11-20。 data_range (from output): 2014-01-02 to 2026-11-20
IS/OOS split: 配置以日历日期划分，样本内为 2014-01-01 至 2019-12-31，样本外为 2020-01-01 至 2026-11-20，均含首尾。组合连续模拟后按日切片，样本外沿用同一规则滚动拟合。预测误差要求信号日和五日目标终点均在对应区间内。
Costs and slippage: 初始建仓及后续再平衡均按每腿买卖绝对成交金额单边收取 5 bps，从组合权益扣除，并按扣费后权益迭代求目标持仓。滑点和买卖价差均未单独建模。
Signal lag: 在新一周首个交易日之前最近一个交易日的收盘 t 计算周度信号，于下一交易日 t+1 开盘按目标权重再平衡。回测首日开盘使用截至该开盘日已有的有效目标。后续缺少有效周度信号时，持仓随价格变动。
Price adjustment: Yahoo split-adjusted Open/Close for fills and valuation, Adj Close for signals, and separate cash distributions
Survivorship bias: 固定使用 USO、SPY、BIL 三只长期存续 ETF。README 明确披露事后产品选择与存续偏差。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.07886750537418874, ann_vol=0.3013243490506808, sharpe=-0.122253568179605, sharpe_excess=-0.14625415274861384, max_drawdown=-0.7153344760600222, total_return=-0.38875435690747295, turnover=29.82191740514346, n_trades=383
    oos: cagr=0.0774520321688732, ann_vol=0.3929235037930702, sharpe=0.38935587469712635, sharpe_excess=0.319561294372033, max_drawdown=-0.7734449933090879, total_return=0.6698274172072458, turnover=45.88653378402627, n_trades=460
    full: cagr=0.0015919214787900593, ann_vol=0.35322563221661507, sharpe=0.18281176874662353, sharpe_excess=0.13180723716243528, max_drawdown=-0.8741810001318548, total_return=0.02067473348438087, turnover=75.70845118916972, n_trades=843
    benchmark buy_and_hold_uso full: cagr=-0.050493942915678325, ann_vol=0.3562429865175857, sharpe=0.034926193602012955, sharpe_excess=-0.0156360033224963, max_drawdown=-0.9275982045326717, total_return=-0.4865399802527268, turnover=0.9777408608798147, n_trades=1
    benchmark own_market_gate full: cagr=-0.0016124519139468774, ann_vol=0.32746289541082396, sharpe=0.15992257649648506, sharpe_excess=0.10490003121079541, max_drawdown=-0.8403269798238995, total_return=-0.020547037260637002, turnover=58.977335397250286, n_trades=811
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "323_information_transmission_across_tail"\nuniverse:\n  oil: USO\n  equity: SPY\n  defensive: BIL\n  implied_vol: OVXCLS\ndata_start: "2008-01-01"\nin_sample:\n  start: "2014-01-01"\n  end: "2019-12-31"\nout_of_sample:\n  start: "2020-01-01"\n  end: "2026-11-20"\nmodel:\n  training_sessions: 504\n  forecast_sessions: 5\n  oil_variance_windows: [1, 5, 22]\n  equity_quantile_window: 252\n  equity_tail_quantile: 0.05\n  equity_tail_window: 20\n  ovx_first_public_date: "2008-07-15"\n  ovx_lag_sessions: 2\n  ovx_max_fill_sessions: 5\n  ridge_alpha: 1.0\nallocation:\n  gate_history_weeks: 52\n  gate_quantile: 0.80\n  defensive_weight: 0.50\n  transaction_cost_bps: 5.0\n  slippage_bps: 5.0\n  cash_buffer: 0.005\n  distribution_payment_lag_sessions: 10\nsessions_per_year: 252\n'
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
"""Weekly USO allocation using an incremental SPY downside-tail forecast."""

import sys

sys.dont_write_bytecode = True

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import yaml
from curl_cffi.requests.exceptions import RequestException
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from yfinance.exceptions import YFException


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
HTTP_TIMEOUT = (10, 30)
OSLO = ZoneInfo("Europe/Oslo")
ETF_COLUMNS = ["open", "close", "adj_close", "distribution"]
PRICE_ADJUSTMENT = (
    "Yahoo split-adjusted Open/Close for fills and valuation, Adj Close for signals, "
    "and separate cash distributions"
)


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def nyse_sessions(start, end):
    if start.year < 2008 or end.year > 2026:
        raise ValueError("Update the NYSE closure dates before extending the configured sample")
    closures = pd.to_datetime(["2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"])
    return pd.bdate_range(start, end).difference(NyseHolidays().holidays(start, end).union(closures))


def yahoo_history_page(symbol, start, stop):
    response = requests.get(
        f"https://finance.yahoo.com/quote/{symbol}/history/",
        params={
            "period1": int(start.tz_localize("UTC").timestamp()),
            "period2": int(stop.tz_localize("UTC").timestamp()),
        },
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    table = pd.read_html(StringIO(response.text), flavor="lxml")[0]
    table.index = pd.to_datetime(table["Date"], format="%b %d, %Y")
    opens = pd.to_numeric(table["Open"], errors="coerce")
    quotes = table.loc[opens.notna()]
    events = table.loc[opens.isna(), "Open"].astype(str)
    amounts = events.str.extract(r"^([\d,.]+)\s+(?:Dividend|Capital Gains?)\b", expand=False)
    splits = events.str.match(r"^\d+:\d+\s+Stock Splits?\b")
    if (amounts.isna() & ~splits).any() or not events.index.isin(quotes.index).all():
        raise ValueError(f"Unrecognized Yahoo Finance price or corporate-action row for {symbol}")
    distributions = pd.to_numeric(amounts.str.replace(",", "", regex=False)).groupby(level=0).sum()
    return pd.DataFrame({
        "open": opens.loc[opens.notna()],
        "close": pd.to_numeric(quotes[next(c for c in table if c.startswith("Close "))]),
        "adj_close": pd.to_numeric(quotes[next(c for c in table if c.startswith("Adj Close "))]),
        "distribution": distributions.reindex(quotes.index, fill_value=0.0),
    })


def yahoo_prices(symbol, start, end):
    stop = end + pd.Timedelta(days=1)
    try:
        history = yf.Ticker(symbol).history(
            start=str(start.date()), end=str(stop.date()), interval="1d",
            auto_adjust=False, actions=True, repair=False, timeout=15,
        )
        if history.empty:
            raise ValueError(f"Empty Yahoo chart history for {symbol}")
        frame = history.rename(columns={"Open": "open", "Close": "close", "Adj Close": "adj_close"})
        frame["distribution"] = history["Dividends"] + history.get("Capital Gains", 0.0)
        frame.index = frame.index.tz_localize(None)
        return frame[ETF_COLUMNS], "Yahoo Finance via yfinance"
    except (YFException, RequestException, requests.RequestException, ValueError, KeyError):
        return yahoo_history_page(symbol, start, stop), "Yahoo Finance public history page"


def fred_series(symbol, start, end):
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": symbol, "cosd": str(start.date()), "coed": str(end.date())},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), index_col=0, parse_dates=True)
    frame[symbol] = pd.to_numeric(frame[symbol])
    return frame, "FRED graph CSV"


def cached_data(fetch, symbol, start, end, expected_sessions=None):
    path = CACHE / f"{fetch.__name__}_v2_{symbol}_{start.date()}_{end.date()}.csv"
    cache_hit = path.is_file()
    if cache_hit:
        with path.open(encoding="utf-8") as stream:
            metadata = json.loads(stream.readline())
            frame = pd.read_csv(stream, index_col="date", parse_dates=True)
    else:
        frame, source = fetch(symbol, start, end)
        metadata = {"source": source, "fetched_at": datetime.now(OSLO).isoformat(timespec="seconds")}

    frame = frame.sort_index().loc[start:end].rename_axis("date")
    if frame.empty or frame.index.has_duplicates:
        raise ValueError(f"Empty or duplicate history for {symbol}")
    if expected_sessions is not None:
        if not frame.index.equals(expected_sessions):
            missing = expected_sessions.difference(frame.index).strftime("%Y-%m-%d").tolist()
            extra = frame.index.difference(expected_sessions).strftime("%Y-%m-%d").tolist()
            raise ValueError(f"{symbol}: missing NYSE sessions {missing[:10]}, extra sessions {extra[:10]}")
        if (not np.isfinite(frame[ETF_COLUMNS]).all().all()
                or (frame[ETF_COLUMNS[:3]] <= 0).any().any()
                or (frame["distribution"] < 0).any()):
            raise ValueError(f"Invalid price or distribution history for {symbol}")
    else:
        observations = frame[symbol].dropna()
        grace = pd.Timedelta(days=5)
        if (observations.empty or observations.index[0] > start + grace
                or observations.index[-1] < end - grace
                or not np.isfinite(observations).all() or (observations <= 0).any()):
            raise ValueError(f"Incomplete or invalid history for {symbol}")

    frame.attrs.update(metadata)
    if not cache_hit:
        with TemporaryDirectory(dir=CACHE) as directory:
            temporary_path = Path(directory) / path.name
            with temporary_path.open("w", encoding="utf-8") as stream:
                stream.write(json.dumps(metadata) + "\n")
                frame.to_csv(stream)
            temporary_path.replace(path)
    return frame


def load_market_data(universe, start, end):
    CACHE.mkdir(exist_ok=True)
    sessions = nyse_sessions(start, end)
    with TemporaryDirectory(dir=CACHE) as directory:
        yf.set_tz_cache_location(directory)
        with ThreadPoolExecutor(max_workers=4) as pool:
            downloads = {
                role: pool.submit(
                    cached_data,
                    fred_series if role == "implied_vol" else yahoo_prices,
                    symbol, start, end, None if role == "implied_vol" else sessions,
                )
                for role, symbol in universe.items()
            }
            return {role: future.result() for role, future in downloads.items()}


def ridge_predict(training_features, targets, current_features, alpha):
    center = training_features.mean(axis=0)
    scale = training_features.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (training_features - center) / scale
    current = (current_features - center) / scale
    target_mean = targets.mean()
    penalty = alpha * np.eye(standardized.shape[1])
    coefficients = np.linalg.solve(
        standardized.T @ standardized + penalty,
        standardized.T @ (targets - target_mean),
    )
    return float(target_mean + current @ coefficients)


def weekly_forecasts(data, model, annual_sessions):
    oil_variance = data["oil"]["adj_close"].pct_change(fill_method=None).pow(2)
    sessions = oil_variance.index
    equity_returns = data["equity"]["adj_close"].pct_change(fill_method=None)
    cutoff = equity_returns.shift(1).rolling(model["equity_quantile_window"]).quantile(
        model["equity_tail_quantile"]
    )
    tail_days = (equity_returns < cutoff).astype(float).where(cutoff.notna())
    equity_tail = tail_days.rolling(model["equity_tail_window"]).mean()
    ovx = data["implied_vol"].iloc[:, 0].loc[pd.Timestamp(model["ovx_first_public_date"]):]
    ovx = ovx.reindex(sessions).shift(model["ovx_lag_sessions"]).ffill(limit=model["ovx_max_fill_sessions"])
    features = pd.DataFrame({
        f"oil_{n}": oil_variance.rolling(n).mean() for n in model["oil_variance_windows"]
    })
    features["ovx"] = (ovx / 100).pow(2) / annual_sessions
    features["equity_tail"] = equity_tail

    horizon = model["forecast_sessions"]
    target = oil_variance.rolling(horizon).mean().shift(-horizon).to_numpy()
    feature_values = features.to_numpy()
    width = model["training_sessions"]
    week = sessions.to_period("W-FRI")
    # The first session of a new week trades on the preceding session's signal.
    prior_week_closes = np.flatnonzero(week[:-1] != week[1:])
    forecasts = []
    for position in prior_week_closes:
        # Origin s has a known label at t exactly when s + horizon <= t.
        train_end = position - horizon + 1
        train_start = train_end - width
        if train_start < 0:
            continue
        train = feature_values[train_start:train_end]
        labels = target[train_start:train_end]
        current = feature_values[position]
        if not (np.isfinite(train).all() and np.isfinite(labels).all() and np.isfinite(current).all()):
            continue
        own = ridge_predict(train[:, :-1], labels, current[:-1], model["ridge_alpha"])
        cross = ridge_predict(train, labels, current, model["ridge_alpha"])
        target_end = position + horizon
        forecasts.append({
            "signal": sessions[position],
            "execution": sessions[position + 1],
            "target_end": sessions[target_end] if target_end < len(sessions) else pd.NaT,
            "realized": target[position],
            "own": own,
            "cross": cross,
        })
    if not forecasts:
        raise ValueError("There are no weekly forecasts with a complete training window")
    weekly = pd.DataFrame(forecasts).set_index("execution")
    expected = sessions[prior_week_closes + 1]
    return weekly.reindex(expected[expected >= weekly.index[0]])


def allocation_signals(weekly, config):
    history = config["gate_history_weeks"]
    incremental = weekly["cross"] - weekly["own"]
    tail_cutoff = incremental.shift(1).rolling(history).quantile(config["gate_quantile"])
    tail_gate = (incremental > tail_cutoff).astype(float).where(
        incremental.notna() & tail_cutoff.notna()
    )

    # Match the cross gate's observed past activation rate using own forecast ranks.
    target_rate = tail_gate.shift(1).rolling(history).mean()
    own_gate = pd.Series(index=weekly.index, dtype=float)
    own_forecasts = weekly["own"].to_numpy()
    for i in np.flatnonzero(target_rate.notna() & weekly["own"].notna()):
        rate = target_rate.iloc[i]
        past_forecasts = own_forecasts[i - history:i]
        if rate == 0 or rate == 1:
            own_gate.iloc[i] = rate
        else:
            own_gate.iloc[i] = float(own_forecasts[i] > np.quantile(past_forecasts, 1 - rate))

    gates = pd.DataFrame({"tail": tail_gate, "own": own_gate})
    return 1 - config["defensive_weight"] * gates


def fill_at_open(positions, cash, requested_positions, opening, fee, slippage):
    change = requested_positions - positions
    sales = np.maximum(-change, 0.0)
    purchases = np.maximum(change, 0.0)
    sale_value = sales * opening * (1 - slippage)
    purchase_value = purchases * opening * (1 + slippage)
    cash_after_sales = cash + sale_value.sum() * (1 - fee)
    purchase_cost = purchase_value.sum() * (1 + fee)
    if purchase_cost > cash_after_sales:
        fraction = cash_after_sales / purchase_cost
        purchases *= fraction
        purchase_value *= fraction
        purchase_cost = purchase_value.sum() * (1 + fee)
    unspent = cash_after_sales - purchase_cost
    if unspent < -1e-12 * cash_after_sales:
        raise ArithmeticError("Opening fills exceeded available cash")
    return positions - sales + purchases, max(unspent, 0.0), sale_value + purchase_value


def simulate(oil, defensive, oil_target, rebalance_dates, allocation):
    opening, closing, prior_close, distributions = (
        np.column_stack((oil[column].loc[oil_target.index], defensive[column].loc[oil_target.index]))
        for column in ("open", "close", "previous_close", "distribution")
    )
    fee = allocation["transaction_cost_bps"] / 10_000
    slippage = allocation["slippage_bps"] / 10_000
    invested_fraction = 1 - allocation["cash_buffer"]
    delay = allocation["distribution_payment_lag_sessions"]
    positions = np.zeros(2)
    cash = previous_value = 1.0
    due_payments = np.zeros(len(oil_target) + delay)
    records = []
    for i, target in enumerate(oil_target):
        rebalance_today = i == 0 or oil_target.index[i] in rebalance_dates
        if rebalance_today:
            # Set the quantities using the previous close and cash then available.
            sizing_value = cash + positions @ prior_close[i]
            weights = np.array([target, 1 - target])
            requested_positions = sizing_value * invested_fraction * weights / prior_close[i]

        # Only shares carried into the ex-date earn the distribution.
        due_payments[i + delay] = positions @ distributions[i]
        cash += due_payments[i]
        unpaid_distributions = due_payments[i + 1:i + delay + 1].sum()
        open_value = cash + positions @ opening[i] + unpaid_distributions
        filled = np.zeros(2)
        if rebalance_today:
            positions, cash, filled = fill_at_open(
                positions, cash, requested_positions, opening[i], fee, slippage,
            )
        post_fill_value = cash + positions @ opening[i] + unpaid_distributions
        oil_weight = positions[0] * opening[i, 0] / post_fill_value
        close_value = cash + positions @ closing[i] + unpaid_distributions
        records.append((close_value / previous_value - 1, filled.sum() / open_value,
                        np.count_nonzero(filled), oil_weight, target < 1))
        previous_value = close_value
    return pd.DataFrame(records, index=oil_target.index, columns=[
        "return", "turnover", "n_trades", "avg_uso_weight", "de_risk_share",
    ])


def performance(daily, cash_returns, annual_sessions):
    returns = daily["return"]
    excess_returns = returns - cash_returns.reindex(returns.index)
    growth = (1 + returns).cumprod()
    drawdown = growth / growth.cummax().clip(lower=1) - 1
    daily_vol = returns.std()
    excess_vol = excess_returns.std()
    annualization = np.sqrt(annual_sessions)
    return {
        "cagr": float(growth.iloc[-1] ** (annual_sessions / len(returns)) - 1) if len(returns) else None,
        "ann_vol": float(daily_vol * annualization) if np.isfinite(daily_vol) else None,
        "sharpe": float(returns.mean() / daily_vol * annualization) if daily_vol > 0 else None,
        "sharpe_excess": float(excess_returns.mean() / excess_vol * annualization) if excess_vol > 0 else None,
        "max_drawdown": float(drawdown.min()) if len(returns) else None,
        "total_return": float(growth.iloc[-1] - 1) if len(returns) else None,
        "n_trades": int(daily["n_trades"].sum()),
        "turnover": float(daily["turnover"].sum()),
        "avg_uso_weight": float(daily["avg_uso_weight"].mean()) if len(returns) else None,
        "de_risk_share": float(daily["de_risk_share"].mean()) if len(returns) else None,
    }


def forecast_error(weekly, periods):
    rows = {}
    for period, (start, end) in periods.items():
        sample = weekly.loc[weekly["signal"].between(start, end) & (weekly["target_end"] <= end)]
        scores = {"n_forecasts": len(sample)}
        for model in ("cross", "own"):
            residuals = (sample[model].clip(lower=0) - sample["realized"]) * 10_000
            scores[f"{model}_rmse"] = float(np.sqrt(residuals.pow(2).mean())) if len(sample) else None
            scores[f"{model}_mae"] = float(residuals.abs().mean()) if len(sample) else None
        rows[period] = scores
    return rows


def print_results(results, portfolios, universe, data):
    rows = [
        {"portfolio": name, "period": period.upper(), **values}
        for name, metrics in portfolios.items()
        for period, values in metrics.items()
    ]
    table = pd.DataFrame(rows).set_index(["portfolio", "period"])
    percentage = ["cagr", "ann_vol", "max_drawdown", "total_return", "avg_uso_weight", "de_risk_share"]
    formatters = {column: "{:.2%}".format for column in percentage}
    print(f"{results['strategy']} | return dates {results['data_range']}")
    print("PARAM_COMBOS_TRIED: 1")
    print("DATA_FETCHED_AT: " + max(frame.attrs["fetched_at"] for frame in data.values()))
    print("DATA_SOURCES: " + ", ".join(
        f"{universe[role]}={frame.attrs['source']} ({frame.attrs['fetched_at']})"
        for role, frame in data.items()
    ))
    print("PRICE_ADJUSTMENT: " + PRICE_ADJUSTMENT)
    print("\nNet portfolio performance (turnover in multiples of capital)")
    print(table.to_string(formatters=formatters, float_format="{:.3f}".format, na_rep="n/a"))
    print("\nWeekly five-session variance forecast errors (squared percentage points)")
    print(pd.DataFrame.from_dict(results["forecast_error"], orient="index").rename(index=str.upper).to_string(
        float_format="{:.4f}".format, na_rep="n/a"
    ))
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not is_start < is_end < oos_start < oos_end or oos_start - is_end != pd.Timedelta(days=1):
        raise ValueError("Configure adjacent, non-overlapping IS and OOS date ranges")
    periods = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    annual_sessions = config["sessions_per_year"]
    universe = config["universe"]
    allocation = config["allocation"]
    if not all(0 <= allocation[key] < 100 for key in ("transaction_cost_bps", "slippage_bps")):
        raise ValueError("transaction_cost_bps and slippage_bps must be in the range [0, 100)")
    if not 0 <= allocation["defensive_weight"] <= 1 or not 0 < allocation["cash_buffer"] < 1:
        raise ValueError("Invalid defensive_weight or cash_buffer")
    delay = allocation["distribution_payment_lag_sessions"]
    if not isinstance(delay, int) or delay < 1:
        raise ValueError("distribution_payment_lag_sessions must be a positive integer")
    if config["model"]["ovx_lag_sessions"] < 1:
        raise ValueError("OVX must have at least one session of publication lag")
    data = load_market_data(universe, pd.Timestamp(config["data_start"]), oos_end)
    weekly = weekly_forecasts(data, config["model"], annual_sessions)
    signals = allocation_signals(weekly, allocation)

    oil, defensive = (data[role].assign(previous_close=data[role]["close"].shift())
                      .loc[is_start:oos_end] for role in ("oil", "defensive"))
    if oil.empty or oil["previous_close"].isna().any():
        raise ValueError("The return sample requires a prior session's close for sizing")
    cash_returns = (defensive["close"] + defensive["distribution"]) / defensive["previous_close"] - 1
    cash_returns.iloc[0] = defensive["close"].iloc[0] / defensive["open"].iloc[0] - 1
    portfolios = {}
    for name, scheduled in {
        "tail_gate": signals["tail"].dropna(),
        "own_market_gate": signals["own"].dropna(),
        "buy_and_hold_uso": pd.Series(1.0, index=oil.index[:1]),
    }.items():
        target = scheduled.reindex(oil.index, method="ffill")
        if target.empty or pd.isna(target.iloc[0]):
            raise ValueError(f"Insufficient pre-IS history for {name}")
        daily = simulate(oil, defensive, target, frozenset(scheduled.index), allocation)
        portfolios[name] = {
            period: performance(daily.loc[start:end], cash_returns, annual_sessions)
            for period, (start, end) in periods.items()
        }

    results = {
        "strategy": config["strategy"],
        "data_range": f"{oil.index[0].date()} to {oil.index[-1].date()}",
        **portfolios["tail_gate"],
        "benchmark": {name: portfolios[name] for name in ("buy_and_hold_uso", "own_market_gate")},
        "forecast_error": forecast_error(weekly, periods),
    }
    print_results(results, portfolios, universe, data)


if __name__ == "__main__":
    main()

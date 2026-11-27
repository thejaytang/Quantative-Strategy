"""022_sector_residual_statarb
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Gemini; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/022_sector_residual_statarb.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit ed11995234d0b7af91b8ab7ab035ba02e738dbe6 on 2026-11-27 06:30 Europe/Oslo, exit 0, output sha256 626480c9bb8bfff75595071ccd0828c8f3c11b6b9a46be50ed0238da86c54a36
Data source: Yahoo Finance，经 yfinance 下载日频收盘价。行业 ETF 为 XLB、XLE、XLF、XLI、XLK、XLP、XLU、XLV、XLY、XLC、XLRE，另取 SPY 和 BIL 的行情及 ETF 现金分配。SPY 用作回归因子和买入持有基准，BIL 用作超额收益参照，Yahoo 的 ^IRX 日收盘报价用于现金计息和融资。（config.yaml:9-28，strategy.py:32-42、63-71、74-91、108-111、238-244、286-317，logs/codex_fix_d2_run_20261126_104407.log:1、4）
Data fetched: 2026-11-27 06:30 Europe/Oslo (run time)
Sample period: 请求区间为 2015-01-01 起至 2026-11-01（不含）。日志记录实际行情为 2015-01-02 至 2026-10-30，共 2975 个交易日。信号预热后的评估为 2015-10-05 至 2026-10-30，共 2785 个交易日。（config.yaml:2-3，strategy.py:78-89、275-286，logs/codex_fix_d2_run_20261126_104407.log:3-5）   data_range (from output): {'start': '2015-01-02', 'end': '2026-10-30', 'evaluation_start': '2015-10-05', 'oos_start': '2022-01-01'}
IS/OOS split: 评估样本按 2022-01-01 切分，之前为 IS，当日及以后为 OOS。两个区间使用同一组参数和连续持仓。这是 2026-11-26 修复时在已有全样本结果后加上的回溯切分，并非事先固定的样本外划分。（config.yaml:4，strategy.py:260-265、286-287、316，README.md:19、43-45，backtest_results.md:18，logs/codex_fix_d2_run_20261126_104407.log:4）
Costs and slippage: 单边成交费为每次实际绝对成交额的 5 bps，按成交收盘价计算并扣减现金。SPY 基准首次买入也支付同一费率。借券费为前一收盘空头市值的年化 50 bps，按实际日历天数除以 365 计提。按空头市值的 102% 预留不计息抵押品，剩余现金按滞后的 ^IRX 换算利率计息，负余额按同率计融资费。滑点未建模，买卖价差和市场冲击也未建模。（config.yaml:5-7，strategy.py:63-70、159-199、221-234、290-294，README.md:13、47-49）
Signal lag: 交易日 t 收盘后，以当日收益生成信号，用 t 日收盘价及交易后净值确定目标股数。该股数在下一交易日 t+1 收盘模拟成交。t 至 t+1 的价变、现金分配及融资先计入原持仓。（strategy.py:115-128、154-156、173-194、196-216、275-280，README.md:11）
Price adjustment: yfinance 设置 auto_adjust=False、actions=True，使用 Close。README 将该口径描述为 Yahoo 拆股调整后的等价股数。信号收益用（当日 Close 加当日股息及资本利得分配）除以前一日 Close 再减一。成交按 Close 记账，除息日的分配另按原持仓股数计入现金，空头承担对应分配。（strategy.py:40-42、63-71、87-91、175-193、275-278，README.md:11-13）
Survivorship bias: 使用当前固定的 11 只行业 ETF 回看历史，存在事后选池偏差，退市或未入名单的产品不纳入。XLC 自 2018-06-19、XLRE 自 2015-10-08 纳入，各自积累信号窗口。未重建历史行业分类或替代标的，偏差未量化。（config.yaml:9-24，strategy.py:53-61、115-128，README.md:17、51-53）
Parameter combinations tried: 未记录
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.00021474464000736937, ann_vol=0.0730299711719651, sharpe=0.03357337037517022, sharpe_excess=-0.07140371805968677, max_drawdown=-0.14557761758118148, total_return=-0.0013396954310883613, turnover=219.6766880287238, n_trades=7106
    oos: cagr=-0.01398185216580139, ann_vol=0.09249547555466735, sharpe=-0.10602302705132025, sharpe_excess=-0.5177004814056095, max_drawdown=-0.25455296077062617, total_return=-0.06547844969081662, turnover=164.8153447823205, n_trades=6166
    full: cagr=-0.006229499073650224, ann_vol=0.0820557559953671, sharpe=-0.03513344696253851, sharpe_excess=-0.2898060401472896, max_drawdown=-0.2980056274224895, total_return=-0.06673042394201334, turnover=384.4920328110443, n_trades=13272
    benchmark SPY full: cagr=0.13850423565782966, ann_vol=0.16308703252029264, sharpe=0.8773250771901916, sharpe_excess=0.7489055608373023, max_drawdown=-0.3162502476374127, total_return=3.193602960659941, turnover=0.9995002498750625, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy_name: "022_sector_residual_statarb"\nstart_date: "2015-01-01"\nend_date: "2026-11-01"\noos_start: "2022-01-01"\ncosts_bps: 5.0\nshort_borrow_bps: 50.0\nshort_collateral_ratio: 1.02\n\nsector_tickers:\n  - "XLB"\n  - "XLE"\n  - "XLF"\n  - "XLI"\n  - "XLK"\n  - "XLP"\n  - "XLU"\n  - "XLV"\n  - "XLY"\n  - "XLC"\n  - "XLRE"\n\nsector_listing_dates:\n  XLC: "2018-06-19"\n  XLRE: "2015-10-08"\n\nmarket_ticker: "SPY"\ncash_ticker: "BIL"\nfinancing_ticker: "^IRX"\n\nols_window: 126\nzscore_window: 63\nentry_z: 2.0\nexit_z: 0.5\nstop_z: 4.0\ngross_exposure: 1.0\n'
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
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


SESSIONS_PER_YEAR = 252
BILL_MATURITY_DAYS = 91


def load_config(config_path="config.yaml"):
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if min(config["ols_window"], config["zscore_window"]) < 2:
        raise ValueError("OLS and z-score windows require at least two sessions")
    if not 0 <= config["exit_z"] < config["entry_z"] < config["stop_z"]:
        raise ValueError("Require 0 <= exit_z < entry_z < stop_z")
    if not config["sector_tickers"] or config["gross_exposure"] < 0:
        raise ValueError("Require sector tickers and a nonnegative gross exposure target")
    if min(config["costs_bps"], config["short_borrow_bps"]) < 0 or config["short_collateral_ratio"] < 1:
        raise ValueError("Costs must be nonnegative and collateral must cover at least 100% of shorts")
    return config


def validate_market_data(data, config):
    sectors = config["sector_tickers"]
    etfs = sectors + [config["market_ticker"], config["cash_ticker"]]
    rate_ticker = config["financing_ticker"]
    start, end = (pd.Timestamp(config[key]) for key in ("start_date", "end_date"))
    if data is None or data.empty or not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("yfinance returned no usable daily ETF data")

    close = data["Close"][etfs].copy()
    dividends = data["Dividends"][etfs]
    capital_gains = data["Capital Gains"][etfs]
    index = close.index
    if (
        index.hasnans or not index.is_unique or not index.is_monotonic_increasing
        or not (index.day_of_week < 5).all() or index[0] < start or index[-1] >= end
        or index[0] - start > pd.Timedelta(days=7)
        or end - index[-1] > pd.Timedelta(days=7)
        or index.to_series().diff().max() > pd.Timedelta(days=7)
    ):
        raise ValueError("Yahoo session index is invalid or does not span the requested range")

    listing_dates = config["sector_listing_dates"]
    for ticker in etfs:
        listed = index >= pd.Timestamp(listing_dates.get(ticker, config["start_date"]))
        if close.loc[~listed, ticker].notna().any():
            raise ValueError(f"{ticker} has a quote before its configured listing date")
        for field, values in (("Close", close), ("Dividends", dividends), ("Capital Gains", capital_gains)):
            observed = values.loc[listed, ticker]
            if not np.isfinite(observed).all() or (field == "Close" and (observed <= 0).any()):
                raise ValueError(f"Incomplete or invalid {field} for listed ETF {ticker}")

    irx = data["Close"][rate_ticker].rename(rate_ticker)
    # Use a quote already at least one equity session old at the start of the accrual interval.
    discount = irx.ffill(limit=5).shift(2) / 100.0
    bill_price = 1.0 - discount * BILL_MATURITY_DAYS / 360.0
    if not np.isfinite(bill_price.iloc[2:]).all() or (bill_price.iloc[2:] <= 0).any():
        raise ValueError(f"Missing or invalid lagged {rate_ticker} financing quotes")
    days = index.to_series().diff().dt.days
    cash_returns = np.expm1(-np.log1p(bill_price - 1.0) * days / BILL_MATURITY_DAYS)
    return close, dividends + capital_gains, irx, cash_returns


def load_market_data(config):
    tickers = config["sector_tickers"] + [
        config["market_ticker"], config["cash_ticker"], config["financing_ticker"],
    ]
    today_ny = datetime.now(ZoneInfo("America/New_York")).date()
    start, end = (pd.Timestamp(config[key]).date() for key in ("start_date", "end_date"))
    if not start < end <= today_ny:
        raise ValueError("Use a nonempty date range whose exclusive end is no later than today in New York")

    with TemporaryDirectory(prefix=".yfinance-", dir=".") as cache_dir:
        yf.set_tz_cache_location(cache_dir)
        for attempt in range(2):
            try:
                raw = yf.download(
                    tickers, start=config["start_date"], end=config["end_date"],
                    interval="1d", ignore_tz=True, auto_adjust=False, actions=True,
                    progress=False, timeout=12,
                )
                close, distributions, irx, cash_returns = validate_market_data(raw, config)
            except (KeyError, ValueError) as error:
                if attempt:
                    raise RuntimeError(f"Yahoo data failed validation: {error}") from error
            else:
                break

    fields = pd.concat({"Close": close, "Distributions": distributions, "Financing quote": irx.to_frame()}, axis=1)
    digest = sha256(fields.to_csv().encode("utf-8")).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Data source: Yahoo Finance via yfinance {yf.__version__}, daily Close and cash distributions, {timestamp}")
    print(f"Input SHA-256: {digest}")
    print(f"Price sessions: {len(close)} ({close.index[0].date()} to {close.index[-1].date()})")
    return close, distributions, cash_returns


def build_signals(total_returns, config):
    sectors = config["sector_tickers"]
    market = total_returns[config["market_ticker"]]
    sector_returns = total_returns[sectors]
    ols_window = config["ols_window"]
    z_window = config["zscore_window"]

    # Roll each sector on a complete paired window ending the session before the signal.
    residuals = {}
    for ticker in sectors:
        sector = sector_returns[ticker]
        paired_market = market.where(sector.notna())
        market_window = paired_market.rolling(ols_window)
        variance = market_window.var(ddof=0).where(lambda value: value > 1e-12)
        beta = sector.rolling(ols_window).cov(paired_market, ddof=0) / variance
        alpha = sector.rolling(ols_window).mean() - beta * market_window.mean()
        residuals[ticker] = sector - alpha.shift(1) - beta.shift(1) * market

    cumulative = pd.DataFrame(residuals).cumsum()
    recent = cumulative.rolling(z_window)
    scores = (cumulative - recent.mean()) / recent.std().where(lambda value: value > 0)

    entry, exit_, stop = (config[key] for key in ("entry_z", "exit_z", "stop_z"))
    side_weight = config["gross_exposure"] / 2.0
    state = np.zeros(len(sectors))
    targets = []
    for score in scores.to_numpy():
        flat = state == 0
        long_entry = flat & (score > -stop) & (score < -entry)
        short_entry = flat & (score > entry) & (score < stop)
        long_exit = (state > 0) & ((score >= -exit_) | (score <= -stop))
        short_exit = (state < 0) & ((score <= exit_) | (score >= stop))
        state[long_exit | short_exit | ~np.isfinite(score)] = 0
        state[long_entry] = 1
        state[short_entry] = -1

        target = np.zeros_like(state)
        for side in (1, -1):
            active = state == side
            if active.any():
                target[active] = side * side_weight / active.sum()
        targets.append(target)
    weights = pd.DataFrame(targets, index=scores.index, columns=sectors)
    return weights, scores.notna().any(axis=1)


def size_next_close_order(weights, marks, nav):
    # Before listing, a zero target needs no quote or position.
    return np.divide(nav * weights, marks, where=weights != 0, out=np.zeros_like(weights))


def run_backtest(prices, distributions, cash_returns, weights, start_row, config):
    """Simulate a close-to-close daily_results in dollars per $1 of starting capital."""
    cost_rate = config["costs_bps"] / 10_000.0
    borrow_rate = config["short_borrow_bps"] / 10_000.0
    collateral_ratio = config["short_collateral_ratio"]
    marks = prices[weights.columns].fillna(0).to_numpy()
    payouts = distributions[weights.columns].fillna(0).to_numpy()
    cash_rates = cash_returns.to_numpy()
    targets = weights.to_numpy()
    days = prices.index.to_series().diff().dt.days.to_numpy()

    cash = nav = 1.0
    collateral = short_value = 0.0
    shares = np.zeros(marks.shape[1])
    pending_shares = size_next_close_order(targets[start_row - 1], marks[start_row - 1], nav)
    daily_results = []
    for row in range(start_row, len(prices)):
        # Today's close, ex-date cash flows and financing belong to yesterday's positions.
        mark = marks[row]
        distribution = shares @ payouts[row]
        cash_interest = (cash - collateral) * cash_rates[row]
        borrow_fee = short_value * borrow_rate * days[row] / 365.0
        cash += distribution + cash_interest - borrow_fee
        pretrade_nav = cash + shares @ mark
        if not np.isfinite(pretrade_nav) or pretrade_nav <= 0:
            raise RuntimeError(f"Nonpositive or invalid pre-trade NAV on {prices.index[row].date()}")

        # This share quantity was set at the previous close and fills at today's close.
        trade_value = (pending_shares - shares) * mark
        trade_value[np.abs(trade_value) <= pretrade_nav * 1e-12] = 0
        next_shares = np.where(trade_value != 0, pending_shares, shares)
        short_change = (np.minimum(next_shares, 0) - np.minimum(shares, 0)) * mark
        turnover = np.abs(trade_value).sum() / pretrade_nav
        trading_fee = cost_rate * np.abs(trade_value).sum()
        cash -= trade_value.sum() + trading_fee
        shares = next_shares

        values = shares * mark
        long_value = np.maximum(values, 0).sum()
        short_value = -np.minimum(values, 0).sum()
        collateral = collateral_ratio * short_value
        previous_nav, nav = nav, cash + long_value - short_value
        if not np.isfinite(nav) or nav <= 0:
            raise RuntimeError(f"Nonpositive or invalid post-trade NAV on {prices.index[row].date()}")
        gross = long_value + short_value
        daily_results.append({
            "cash": cash, "free_cash": cash - collateral, "long_value": long_value,
            "short_value": short_value, "collateral": collateral,
            "short_sale_proceeds": -np.minimum(short_change, 0).sum(),
            "short_cover_cost": np.maximum(short_change, 0).sum(),
            "gross_exposure": gross, "gross_leverage": gross / nav,
            "net_leverage": (long_value - short_value) / nav,
            "cash_interest": cash_interest, "borrow_fee": borrow_fee,
            "net_distributions": distribution, "trading_fee": trading_fee,
            "nav": nav, "return": nav / previous_nav - 1.0,
            "turnover": turnover, "n_trades": int(np.count_nonzero(trade_value)),
        })
        pending_shares = size_next_close_order(targets[row], mark, nav)

    return pd.DataFrame(daily_results, index=prices.index[start_row:])


def buy_and_hold(prices, distributions, cash_returns, cost_rate):
    benchmark = pd.DataFrame({"return": 0.0, "turnover": 0.0, "n_trades": 0}, index=prices.index)
    if benchmark.empty:
        return benchmark

    # Enter on the first evaluation close. Subsequent distributions remain in interest-bearing cash.
    notional = nav = 1.0 / (1.0 + cost_rate)
    shares = notional / prices.iloc[0]
    cash = 0.0
    benchmark.iloc[0] = (nav - 1.0, notional, 1)
    for date in prices.index[1:]:
        cash = cash * (1.0 + cash_returns.loc[date]) + shares * distributions.loc[date]
        previous_nav, nav = nav, cash + shares * prices.loc[date]
        benchmark.loc[date, "return"] = nav / previous_nav - 1.0
    return benchmark


def period_metrics(book, cash_returns):
    if book.empty:
        fields = ("cagr", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "total_return", "turnover")
        return dict.fromkeys(fields) | {"n_trades": 0}

    returns = book["return"]
    excess = returns - cash_returns.loc[book.index]
    growth = np.r_[1.0, (1.0 + returns).cumprod()]
    vol, excess_vol = returns.std(), excess.std()
    annualizer = np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": growth[-1] ** (SESSIONS_PER_YEAR / len(book)) - 1.0,
        "ann_vol": vol * annualizer if np.isfinite(vol) else None,
        "sharpe": annualizer * returns.mean() / vol if vol > 0 else None,
        "sharpe_excess": annualizer * excess.mean() / excess_vol if excess_vol > 0 else None,
        "max_drawdown": (growth / np.maximum.accumulate(growth) - 1.0).min(),
        "total_return": growth[-1] - 1.0,
        "turnover": book["turnover"].sum(),
        "n_trades": int(book["n_trades"].sum()),
    }


def summarize_periods(daily_results, cash_returns, oos_start):
    return {
        "is": period_metrics(daily_results.loc[daily_results.index < oos_start], cash_returns),
        "oos": period_metrics(daily_results.loc[daily_results.index >= oos_start], cash_returns),
        "full": period_metrics(daily_results, cash_returns),
    }


def main():
    parser = argparse.ArgumentParser(description="022 Sector Residual Statistical Arbitrage Strategy")
    parser.add_argument("--smoke", action="store_true", help="Check public data and signal construction")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--refresh", action="store_true", help="Fetch Yahoo data (also the default)")
    args = parser.parse_args()

    config = load_config(args.config)
    prices, distributions, cash_rates = load_market_data(config)
    total_returns = (prices + distributions).div(prices.shift(1)) - 1.0
    weights, ready = build_signals(total_returns, config)
    ready_to_fill = np.flatnonzero(ready.to_numpy()[:-1])
    start_row = int(ready_to_fill[0] + 1) if ready_to_fill.size else len(prices)
    if args.smoke:
        print(f"Tickers: {', '.join(prices.columns)}")
        print(f"Signal dimensions: {weights.shape}; nonzero targets: {np.count_nonzero(weights)}")
        return

    book = run_backtest(prices, distributions, cash_rates, weights, start_row, config)
    oos_start = pd.Timestamp(config["oos_start"])
    market = config["market_ticker"]
    cash_ticker = config["cash_ticker"]
    cash_returns = total_returns[cash_ticker]
    benchmark = buy_and_hold(
        prices.loc[book.index, market], distributions.loc[book.index, market],
        cash_rates, config["costs_bps"] / 10_000.0,
    )
    print(f"Split: IS before {oos_start.date()}, OOS on or after that date; excess Sharpe vs {cash_ticker}")
    if not book.empty:
        print(f"Evaluation sessions: {len(book)} ({book.index[0].date()} to {book.index[-1].date()})")
        print(f"Ending cash: {book['cash'].iloc[-1]:.6f}; long value: {book['long_value'].iloc[-1]:.6f}; "
              f"short value: {book['short_value'].iloc[-1]:.6f}; collateral: {book['collateral'].iloc[-1]:.6f}; "
              f"NAV: {book['nav'].iloc[-1]:.6f}")
        print(f"Gross leverage mean/max: {book['gross_leverage'].mean():.4f}/{book['gross_leverage'].max():.4f}; "
              f"net leverage min/max: {book['net_leverage'].min():.4f}/{book['net_leverage'].max():.4f}")
        print(f"Cash flows per $1 initial NAV: trading fees {book['trading_fee'].sum():.6f}; "
              f"short borrow fees {book['borrow_fee'].sum():.6f}; cash interest {book['cash_interest'].sum():.6f}; "
              f"net distributions {book['net_distributions'].sum():.6f}")
    else:
        print("No session available for execution after signal warm-up")

    result = {
        "strategy": config["strategy_name"],
        "data_range": {
            "start": str(prices.index[0].date()), "end": str(prices.index[-1].date()),
            "evaluation_start": str(book.index[0].date()) if not book.empty else None,
            "oos_start": str(oos_start.date()),
        },
        **summarize_periods(book, cash_returns, oos_start),
        "benchmark": {"ticker": market, **summarize_periods(benchmark, cash_returns, oos_start)},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

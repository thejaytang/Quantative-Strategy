"""167_bayesian_cointegration_lowvol
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; non-author review: PASS; publication gate: Codex.
Run (one command): python strategies/167_bayesian_cointegration_lowvol.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 2b23c5bbf706aea933c0cba5f5805efb450ee206 on 2026-11-26 14:26 Europe/Oslo, exit 0, output sha256 26d4da732c7448dc553f90216271623856d3431aec866db7be6b9a8a7bc32736
Data source: Yahoo Finance 日频复权收盘价，优先用 yfinance 下载，必要时直接向 Yahoo 公共 chart 端点取价（strategy.py:28-47,83-101）。股票池为 MSFT、AAPL、ORCL、IBM、JPM、BAC、GS、MS、XOM、CVX、COP、KO、PEP、PG、CL、WMT、COST、TGT、HD、LOW、JNJ、MRK、PFE、CAT、DE，基准为 SPY（config.yaml:5-31）。该次运行从 CSV 缓存读取 3,039 个交易日的数据（logs/run_20261125_114811.log:1）。
Data fetched: 2026-11-26 14:26 Europe/Oslo (run time)
Sample period: 配置数据范围为 2013-12-01 至 2025-12-31，结束日包含在内（config.yaml:2-4；strategy.py:57-58,68-69,85-86）。实际报告区间为 2016-02-01 至 2025-12-31（logs/run_20261125_114811.log:3,11）。报告起点之前的数据仅用于模型预热，信号窗口为 504 个交易日（README.md:21；config.yaml:39；strategy.py:176-186,279-284）。   data_range (from output): 2016-02-01 to 2025-12-31
IS/OOS split: 样本内 2016-02-01 至 2024-07-31，样本外 2024-08-01 至 2025-12-31（config.yaml:32-37）。同一连续模拟在完成后按日期切分统计，持仓跨边界延续；样本外继续以截至信号日的滚动窗口重估（strategy.py:176-187,271-289；README.md:22）。README 记载参数预先固定，未按两段的回测表现调参（README.md:22-24）。
Costs and slippage: 策略对买入和卖出按成交绝对金额收取 5 bps，在模拟成交时从现金扣除；借券费为年化 100 bps，按空头前收盘市值乘以两次收盘间实际日历天数除以 365，从现金扣除（config.yaml:49-51；strategy.py:190-212）。SPY 仅在首个报告交易日收盘买入时计 5 bps（README.md:13,24；strategy.py:223-228,280-283）。滑点未建模，模拟使用复权收盘价，未建模冲击或买卖价差变化（strategy.py:190-212；README.md:28）。
Signal lag: 每月在报告区间内的首个交易日 t，用截至当日收盘的 504 个交易日价格生成目标，按 t 日收盘净值换算为目标复权股数，在下一交易日收盘模拟成交；原持仓承担至成交收盘的价格变动（config.yaml:39；strategy.py:168-186,199-219；README.md:11-13）。
Price adjustment: 使用 Yahoo Finance 股息及拆股复权收盘价，yfinance 设 auto_adjust=False 后显式提取 Adj Close，备用端点取 adjclose；模型以窗口首日复权价归一，模拟以复权股数持仓并按复权收盘价成交和估值（README.md:11-13,28；strategy.py:42-47,83-88,137-142,190-219）。
Survivorship bias: 固定列明的 25 只美国股票按长期历史和行情仍可取得而选定。README 明确记录事后选股及幸存者偏差，缺少历史成分、退市收益及逐日可借券信息（config.yaml:6-31；strategy.py:53-56,68-73,278-280；README.md:17,28）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.03201570788057151, ann_vol=0.06052577913027534, sharpe=-0.5073183954079026, max_drawdown=-0.3494570177227665, total_return=-0.2413370621300851, turnover=155.35778296994084, n_trades=1457
    oos: cagr=0.09998148175726262, ann_vol=0.07240907181343482, sharpe=1.3524319962993203, max_drawdown=-0.08693758038951682, total_return=0.1441028091708001, turnover=21.84308273101449, n_trades=213
    full: cagr=-0.01419780064634768, ann_vol=0.06240828493700318, sharpe=-0.19791642317160424, max_drawdown=-0.3494570177227665, total_return=-0.13201160156925673, turnover=177.20086570095535, n_trades=1670
    benchmark SPY full: cagr=0.15465840317744628, ann_vol=0.17957175543374543, sharpe=0.8910605638618914, max_drawdown=-0.3371727197642552, total_return=3.152784736701223, turnover=0.9995002498750625, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 167_bayesian_cointegration_lowvol\ndata:\n  start: '2013-12-01'\n  end: '2025-12-31'\n  benchmark: SPY\n  tickers:\n    - MSFT\n    - AAPL\n    - ORCL\n    - IBM\n    - JPM\n    - BAC\n    - GS\n    - MS\n    - XOM\n    - CVX\n    - COP\n    - KO\n    - PEP\n    - PG\n    - CL\n    - WMT\n    - COST\n    - TGT\n    - HD\n    - LOW\n    - JNJ\n    - MRK\n    - PFE\n    - CAT\n    - DE\nin_sample:\n  start: '2016-02-01'\n  end: '2024-07-31'\nout_of_sample:\n  start: '2024-08-01'\n  end: '2025-12-31'\nmodel:\n  lookback_sessions: 504\n  adjustment_directions: 6\n  spike_rate: 40.0\n  slab_rate: 2.0\n  slab_probability: 0.20\n  em_max_iterations: 30\n  em_convergence_tolerance: 0.0001\n  adf_lags: 1\n  spread_adf_cutoff: -3.0\n  gross_exposure: 1.0\ncosts:\n  transaction_bps: 5.0  # Per dollar traded, buys and sells.\n  short_borrow_annual_bps: 100.0\nsessions_per_year: 252\n"
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
"""Walk-forward backtest of a sparse Bayesian cointegration basket."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from inspect import signature
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from urllib.parse import quote

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import requests
from scipy.special import expit, logit
from sklearn.linear_model import Lasso
from statsmodels.tsa.stattools import adfuller
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
ADF_OPTIONS = {"result_object": False} if "result_object" in signature(adfuller).parameters else {}


def yahoo_chart_close(ticker, start, end_exclusive):
    """Read the same Yahoo adjusted close when yfinance cannot retrieve a symbol."""
    params = {
        "period1": int(start.tz_localize("UTC").timestamp()),
        "period2": int(end_exclusive.tz_localize("UTC").timestamp()),
        "interval": "1d",
    }
    failures = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        url = f"https://{host}/v8/finance/chart/{quote(ticker, safe='')}"
        try:
            response = requests.get(url, params=params, timeout=10,
                                    headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            chart = response.json()["chart"]["result"][0]
            dates = pd.to_datetime(chart["timestamp"], unit="s", utc=True)
            dates = dates.tz_convert(chart["meta"]["exchangeTimezoneName"])
            adjusted = chart["indicators"]["adjclose"][0]["adjclose"]
            return pd.Series(adjusted, index=dates.tz_localize(None).normalize(),
                             name=ticker, dtype=float).dropna()
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
            failures.append(str(error))
    raise RuntimeError(f"Yahoo adjusted close unavailable for {ticker}: {'; '.join(failures)}")


def load_prices(data):
    if len(data["tickers"]) != len(set(data["tickers"])):
        raise ValueError("Equity universe contains duplicate tickers")
    tickers = list(dict.fromkeys([*data["tickers"], data["benchmark"]]))
    start, end = (pd.Timestamp(data[key]) for key in ("start", "end"))
    end_exclusive = end + pd.Timedelta(days=1)
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    ticker_hash = sha256(",".join(tickers).encode()).hexdigest()[:12]
    cache = cache_dir / f"yahoo_adjusted_{start.date()}_{end.date()}_{ticker_hash}.csv"

    def complete_range(series):
        return (not series.empty and series.index[0] <= start + pd.Timedelta(days=7)
                and series.index[-1] >= end - pd.Timedelta(days=7))

    def validate_panel(panel):
        panel = panel.loc[start:end, tickers]
        if (not panel.index.is_unique or not complete_range(panel)
                or not np.isfinite(panel.to_numpy()).all() or (panel <= 0).any().any()):
            raise ValueError("Incomplete, duplicate, or invalid adjusted prices in the panel")
        return panel

    if cache.exists():
        try:
            prices = validate_panel(pd.read_csv(cache, index_col="Date", parse_dates=["Date"]).sort_index())
            print(f"Loaded {len(prices)} sessions of Yahoo adjusted closes from {cache.name}.")
            return prices
        except (ValueError, KeyError) as error:
            print(f"Refreshing an invalid price cache: {error}", file=sys.stderr)

    yf.set_tz_cache_location(str(cache_dir / "yfinance"))
    try:
        history = yf.download(tickers, start=start.date(), end=end_exclusive.date(),
                              auto_adjust=False, progress=False, threads=6, timeout=8)
        adjusted = history.get("Adj Close", pd.DataFrame()).reindex(columns=tickers)
        adjusted.index = pd.DatetimeIndex(adjusted.index).tz_localize(None).normalize()
    except (OSError, ValueError, KeyError) as error:
        print(f"yfinance download failed: {error}", file=sys.stderr)
        adjusted = pd.DataFrame(columns=tickers, index=pd.DatetimeIndex([]), dtype=float)

    sessions = adjusted.dropna(how="all").index
    missing = [ticker for ticker in tickers
               if not complete_range(adjusted[ticker].dropna())
               or adjusted[ticker].reindex(sessions).isna().any()]
    if missing:
        print(f"Trying Yahoo's public chart endpoint for {len(missing)} symbols.")
        with ThreadPoolExecutor(max_workers=6) as pool:
            recovered = list(pool.map(lambda ticker: yahoo_chart_close(ticker, start, end_exclusive), missing))
        adjusted = pd.concat([adjusted.drop(columns=missing).dropna(how="all"), *recovered], axis=1)

    prices = validate_panel(adjusted.sort_index())
    with NamedTemporaryFile(mode="w", dir=cache_dir, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            prices.to_csv(handle, index_label="Date")
            handle.close()
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Cached {len(prices)} sessions of Yahoo adjusted closes in {cache.name}.")
    return prices


def spike_slab_map(predictors, response, initial, model):
    """EM for a two-Laplace prior, with plug-in Gaussian noise variance."""
    spike, slab = model["spike_rate"], model["slab_rate"]
    prior_log_odds = logit(model["slab_probability"]) + np.log(slab / spike)
    residual = response - predictors @ initial
    noise_variance = max(np.mean(residual ** 2), 1e-8)
    estimator = Lasso(alpha=noise_variance / len(response), fit_intercept=False,
                      max_iter=8000, tol=1e-6, warm_start=True)
    coefficients = initial
    for _ in range(model["em_max_iterations"]):
        slab_posterior = expit(prior_log_odds + (spike - slab) * np.abs(coefficients))
        penalty = spike + (slab - spike) * slab_posterior
        estimator.coef_ = coefficients * penalty
        estimator.fit(predictors / penalty, response)
        updated = estimator.coef_ / penalty
        if np.max(np.abs(updated - coefficients)) <= model["em_convergence_tolerance"]:
            return updated
        coefficients = updated
    return coefficients


def fit_basket(window, model):
    """Return signed adjusted share units per dollar of signal-date equity."""
    base = window[0]
    levels = window / base
    lagged = levels[:-1]
    changes = np.diff(levels, axis=0)
    level_scale = lagged.std(axis=0)
    change_scale = changes.std(axis=0)
    if (level_scale == 0).any() or (change_scale == 0).any():
        return np.zeros(window.shape[1])
    predictors = (lagged - lagged.mean(axis=0)) / level_scale
    responses = (changes - changes.mean(axis=0)) / change_scale
    preliminary = np.linalg.lstsq(predictors, responses, rcond=None)[0]
    _, _, rotation = np.linalg.svd(predictors @ preliminary, full_matrices=False)

    best_statistic = model["spread_adf_cutoff"]
    basket = np.zeros(window.shape[1])
    for direction in rotation[:model["adjustment_directions"]]:
        beta = spike_slab_map(predictors, responses @ direction, preliminary @ direction, model)
        if not ((beta > 0).any() and (beta < 0).any()):
            continue
        price_coefficients = beta / (base * level_scale)
        spread = window @ price_coefficients
        if np.ptp(spread) == 0:
            continue
        statistic = adfuller(spread, maxlag=model["adf_lags"], regression="c",
                             autolag=None, **ADF_OPTIONS)[0]
        if statistic < best_statistic:
            best_statistic = statistic
            basket = price_coefficients

    gross = np.abs(basket) @ window[-1]
    if gross > 0:
        # The likelihood identifies a spread only up to sign. Use its largest leg as the anchor.
        anchor = np.argmax(np.abs(basket) * window[-1])
        basket *= np.sign(basket[anchor]) * model["gross_exposure"] / gross
    return basket


def portfolio_signals(prices, start, end, model):
    lookback = model["lookback_sessions"]
    eligible = prices.loc[start:end].index
    monthly = eligible[~eligible.to_period("M").duplicated()]
    if monthly.empty or prices.index.get_loc(monthly[0]) + 1 < lookback:
        raise ValueError("Insufficient warm-up history at the first in-sample session")
    values = prices.to_numpy()
    targets = {}
    for date in monthly:
        stop = prices.index.get_loc(date) + 1
        targets[date] = fit_basket(values[stop - lookback:stop], model)
    return targets


def simulate(prices, targets, costs):
    """Hold fixed shares between close fills, and mark returns at each close."""
    fee_rate = costs["transaction_bps"] / 10000
    borrow_rate = costs["short_borrow_annual_bps"] / 10000
    shares = np.zeros(prices.shape[1])
    cash = equity = 1.0
    pending = None
    previous_date, previous_close = None, None
    records = []
    for date, close in zip(prices.index, prices.to_numpy()):
        if previous_date is not None:
            short_value = np.maximum(-shares, 0) @ previous_close
            cash -= short_value * borrow_rate * (date - previous_date).days / 365
        pre_trade_equity = cash + shares @ close
        if pre_trade_equity <= 0:
            raise ValueError(f"Portfolio equity depleted on {date.date()}")
        traded, trades = 0.0, 0
        if pending is not None:
            orders = (pending - shares) * close
            traded = np.abs(orders).sum()
            trades = np.count_nonzero(np.abs(orders) > 1e-10 * pre_trade_equity)
            cash -= orders.sum() + traded * fee_rate
            shares = pending
        next_equity = cash + shares @ close
        if next_equity <= 0:
            raise ValueError(f"Portfolio equity depleted after costs on {date.date()}")
        records.append((next_equity / equity - 1, traded / pre_trade_equity, trades))
        equity = next_equity
        pending = equity * targets[date] if date in targets else None
        previous_date, previous_close = date, close
    return pd.DataFrame(records, index=prices.index, columns=["return", "turnover", "n_trades"])


def buy_and_hold(prices, fee_bps):
    fee_rate = fee_bps / 10000
    invested = 1 / (1 + fee_rate)
    benchmark = prices.pct_change().to_frame("return").assign(turnover=0.0, n_trades=0)
    benchmark.iloc[0] = (invested - 1, invested, 1)
    return benchmark


def metrics(history, sessions_per_year):
    returns = history["return"].to_numpy()
    if len(returns) == 0:
        raise ValueError("No trading sessions in a configured reporting period")
    equity = np.r_[1.0, np.cumprod(1 + returns)]
    stdev = returns.std(ddof=1) if len(returns) > 1 else np.nan
    numbers = {
        "cagr": equity[-1] ** (sessions_per_year / len(returns)) - 1,
        "ann_vol": stdev * np.sqrt(sessions_per_year),
        "sharpe": returns.mean() / stdev * np.sqrt(sessions_per_year) if stdev > 0 else np.nan,
        "max_drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
        "total_return": equity[-1] - 1,
        "turnover": history["turnover"].sum(),
    }
    return {key: float(value) if np.isfinite(value) else None for key, value in numbers.items()} | {
        "n_trades": int(history["n_trades"].sum()),
    }


def print_report(result):
    print(f"\n{result['strategy']} | {result['data_range']}")
    print(f"{'Portfolio':<17} {'Period':<6} {'CAGR':>9} {'Ann vol':>9} {'Sharpe':>8} "
          f"{'Max DD':>9} {'Total ret':>10} {'Turnover':>10} {'Trades':>7}")
    for label, entries in (("Strategy", result), (result["benchmark"]["ticker"], result["benchmark"])):
        for period in ("is", "oos", "full"):
            row = entries[period]
            cells = []
            for key in ("cagr", "ann_vol", "sharpe", "max_drawdown", "total_return", "turnover", "n_trades"):
                value = row[key]
                fmt = {"sharpe": ".2f", "turnover": ".2f", "n_trades": "d"}.get(key, ".2%")
                cells.append(format(value, fmt) if value is not None else "—")
            print(f"{label:<17} {period.upper():<6} {cells[0]:>9} {cells[1]:>9} {cells[2]:>8} "
                  f"{cells[3]:>9} {cells[4]:>10} {cells[5]:>10} {cells[6]:>7}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data = config["data"]
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not pd.Timestamp(data["start"]) < is_start <= is_end < oos_start <= oos_end <= pd.Timestamp(data["end"]):
        raise ValueError("Expected ordered data, in-sample, and out-of-sample date ranges")
    prices = load_prices(data)
    if ((prices.index > is_end) & (prices.index < oos_start)).any():
        raise ValueError("In-sample and out-of-sample ranges must cover consecutive trading sessions")
    equity_prices = prices[data["tickers"]]
    targets = portfolio_signals(equity_prices, is_start, oos_end, config["model"])
    assets = equity_prices.loc[is_start:oos_end]
    reference = prices.loc[is_start:oos_end, data["benchmark"]]
    strategy = simulate(assets, targets, config["costs"])
    benchmark = buy_and_hold(reference, config["costs"]["transaction_bps"])
    periods = {"is": slice(is_start, is_end), "oos": slice(oos_start, oos_end), "full": slice(None)}
    annualization = config["sessions_per_year"]
    result = {
        "strategy": config["strategy"],
        "data_range": f"{assets.index[0].date()} to {assets.index[-1].date()}",
        **{key: metrics(strategy.loc[period], annualization) for key, period in periods.items()},
        "benchmark": {
            "ticker": data["benchmark"],
            **{key: metrics(benchmark.loc[period], annualization) for key, period in periods.items()},
        },
    }
    print_report(result)


if __name__ == "__main__":
    main()

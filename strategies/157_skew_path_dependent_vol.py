"""157_skew_path_dependent_vol
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/157_skew_path_dependent_vol.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 498f8135d2067ed389c0a52a60055b65c4b14836 on 2026-11-27 06:36 Europe/Oslo, exit 0, output sha256 f1e4e0eded62958a3277b0d8311e417d771d311b65c348eb7ff0b02331a83613
Data source: Yahoo Finance 经 yfinance 提供的免费 SPY ETF 日频复权开盘价与收盘价，优先读取本地 CSV 缓存。来源见 config.yaml:3、strategy.py:15-32、README.md:19。
Data fetched: 2026-11-27 06:36 Europe/Oslo (run time)
Sample period: 日志记录的回测交易日为 2004-01-02 至 2025-12-31。请求自 1993-01-01 起的可用历史，用于初始化路径状态。来源见 logs/run_20261125_113338.log:1、backtest_results.md:14、config.yaml:4、README.md:19、strategy.py:179-181、strategy.py:191-202。   data_range (from output): 2004-01-02 to 2025-12-31
IS/OOS split: 配置样本内为 2004-01-01 至 2015-12-31，样本外为 2016-01-01 至 2025-12-31。只用样本内扣费后 Sharpe 选择仓位下限，同分取较高下限，日志记录选中 25%。样本外沿用所选下限，仓位与路径状态跨分界连续。来源见 config.yaml:5-10、strategy.py:110-119、strategy.py:198-212、README.md:20-22、logs/run_20261125_113338.log:2-3。
Costs and slippage: 策略与买入持有基准在开盘按实际成交名义金额扣除单边 5 bps，首次买入也扣费。成交名义金额以开盘交易前净值计量，费用进入当日净收益。未单独建模滑点，模拟使用当日复权开盘价。来源见 config.yaml:18、strategy.py:76-101、strategy.py:177、strategy.py:201-203、README.md:20、README.md:26、logs/run_20261125_113338.log:3。
Signal lag: 交易日 t 的 SPY 复权收盘价更新路径状态，目标仓位移后一行，于下一交易日开盘模拟成交。隔夜价格变动按旧仓位计入，开盘后至收盘按新仓位计入。来源见 strategy.py:59-80、strategy.py:83-101、strategy.py:198-202、README.md:20。
Price adjustment: 下载时设置 auto_adjust=True、actions=False，保留复权 Open 和 Close 两列。命中缓存时直接读取文件名以 _adjusted.csv 结尾的 CSV。README 说明供应商可能修订复权价格。来源见 strategy.py:15-38、README.md:19。
Survivorship bias: 使用单一 SPY ETF 自身历史，避免用当前指数成分股回溯选股带来的成分股幸存者偏差。事后选取长期存续的美国 ETF 仍存在市场与产品选择偏差。来源见 config.yaml:3、strategy.py:175、strategy.py:191、README.md:26。
Parameter combinations tried: 3
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.06882383508370718, ann_vol=0.13107841169395812, sharpe=0.5735059568085301, max_drawdown=-0.2954963523301738, total_return=1.2209022342272315, turnover=19.885597326794425, n_trades=1357
    oos: cagr=0.11940629886530238, ann_vol=0.14916328591834913, sharpe=0.8313003143284088, max_drawdown=-0.267228496112917, total_return=2.08113756203267, turnover=16.64702108011502, n_trades=703
    full: cagr=0.09150872373465474, ann_vol=0.13957879871343096, sharpe=0.697460500588218, max_drawdown=-0.2954963523301738, total_return=5.842905295479801, turnover=36.53261840690945, n_trades=2060
    benchmark  full: cagr=0.10612051457947011, ann_vol=0.18765471433884495, sharpe=0.6314628779222705, max_drawdown=-0.5518946581545358, total_return=8.164092434848788, turnover=0.9995002498750625, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 157_skew_path_dependent_vol\ncategory: volatility\nuniverse: [SPY]\ndata_start: '1993-01-01'\nin_sample:\n  start: '2004-01-01'\n  end: '2015-12-31'\nout_of_sample:\n  start: '2016-01-01'\n  end: '2025-12-31'\nallocation:\n  drawdown_start: 0.10\n  drawdown_full: 0.25\n  duration_full_sessions: 63\n  recovery_start: 0.25\n  recovery_full: 0.75\n  min_weight_candidates: [0.25, 0.50, 0.75]\none_way_cost_bps: 5.0\nsessions_per_year: 252\n"
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
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"


def load_prices(ticker, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / f"{ticker}_{start.date()}_{end.date()}_adjusted.csv"
    if cache_file.exists():
        prices = pd.read_csv(cache_file, index_col="Date", parse_dates=["Date"])
    else:
        import yfinance as yf

        with TemporaryDirectory(dir=CACHE_DIR) as library_cache:
            yf.set_tz_cache_location(library_cache)
            prices = yf.Ticker(ticker).history(
                start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
                interval="1d", auto_adjust=True, actions=False, timeout=15,
            )
        if prices.empty:
            raise RuntimeError(f"Yahoo Finance returned no daily prices for {ticker}")
        prices = prices[["Open", "Close"]].copy()
        prices.index = prices.index.tz_localize(None).rename("Date")

    values = prices.to_numpy(dtype=float)
    if (prices.empty or not prices.index.is_monotonic_increasing
            or not prices.index.is_unique or prices.index.hasnans
            or not np.isfinite(values).all() or (values <= 0).any()):
        raise ValueError(f"Invalid adjusted daily prices for {ticker}")
    if prices.index[0] < start or prices.index[-1] > end or (
        end - prices.index[-1] > pd.Timedelta(days=7)
    ):
        raise ValueError(f"Daily prices do not cover the requested end date for {ticker}")

    if not cache_file.exists():
        with NamedTemporaryFile(mode="w", dir=CACHE_DIR, suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
        try:
            prices.to_csv(temporary_path)
            temporary_path.replace(cache_file)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices


def recovery_gain(drawdown):
    return drawdown / (1 - drawdown)


def allocation_stress(close, parameters):
    """Fraction of the allowed equity reduction indicated at each close."""
    drawdown = 1 - close / close.cummax()
    episode = drawdown.eq(0).cumsum()
    by_episode = drawdown.groupby(episode, sort=False)
    worst_depth = by_episode.cummax()
    underwater_days = by_episode.cumcount()

    start, full = (recovery_gain(parameters[key]) for key in ("drawdown_start", "drawdown_full"))
    depth_scale = ((recovery_gain(worst_depth) - start) / (full - start)).clip(0, 1)
    duration_scale = (underwater_days / parameters["duration_full_sessions"]).clip(0, 1)
    recovered = 1 - drawdown / worst_depth.where(worst_depth > 0)
    start, full = (parameters[key] for key in ("recovery_start", "recovery_full"))
    restored = ((recovered - start) / (full - start)).clip(0, 1).fillna(0)
    return depth_scale * (0.5 + 0.5 * duration_scale) * (1 - restored)


def market_relatives(prices):
    return pd.DataFrame({
        "overnight": prices["Open"] / prices["Close"].shift(),
        "intraday": prices["Close"] / prices["Open"],
    })


def simulate(relatives, targets, cost_rate):
    """Hold the old allocation overnight, rebalance at the open, mark at the close."""
    close_weight = 0.0
    rows = []
    for (overnight, intraday), target in zip(relatives.to_numpy(), targets, strict=True):
        overnight_factor = 1 + close_weight * (overnight - 1)
        pretrade_weight = close_weight * overnight / overnight_factor
        weight_change = target - pretrade_weight
        if abs(weight_change) < 1e-12:
            traded = 0.0
            target = pretrade_weight
        else:
            # Signed trade value as a share of pre-trade NAV, with target weights after fees.
            traded = abs(weight_change / (1 + math.copysign(cost_rate * target, weight_change)))
        intraday_factor = 1 + target * (intraday - 1)
        close_weight = target * intraday / intraday_factor
        net_return = overnight_factor * (1 - cost_rate * traded) * intraday_factor - 1
        rows.append((net_return, traded))
    return pd.DataFrame(rows, index=relatives.index, columns=["return", "turnover"])


def risk_stats(returns, sessions_per_year):
    daily_vol = returns.std()
    scale = math.sqrt(sessions_per_year)
    return daily_vol * scale, returns.mean() * scale / daily_vol if daily_vol > 0 else np.nan


def select_min_weight(relatives, stress_at_open, candidates, cost_rate, sessions_per_year):
    scores = []
    for floor in candidates:
        trial = simulate(relatives, 1 - (1 - floor) * stress_at_open, cost_rate)
        _, sharpe = risk_stats(trial["return"], sessions_per_year)
        if math.isfinite(sharpe):
            scores.append((sharpe, floor))
    if not scores:
        raise ValueError("No candidate had a defined in-sample Sharpe ratio")
    return max(scores)[1]


def geometric_capture(returns, reference, mask):
    denominator = np.expm1(np.log1p(reference[mask]).mean())
    if not math.isfinite(denominator) or denominator == 0:
        return np.nan
    return np.expm1(np.log1p(returns[mask]).mean()) / denominator


def metrics(path, reference, sessions_per_year):
    returns = path["return"]
    log_growth = np.log1p(returns)
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1) - 1
    underwater = drawdown < 0
    ann_vol, sharpe = risk_stats(returns, sessions_per_year)
    upside = geometric_capture(returns, reference, reference > 0)
    downside = geometric_capture(returns, reference, reference < 0)
    values = {
        "cagr": np.expm1(log_growth.mean() * sessions_per_year),
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "total_return": np.expm1(log_growth.sum()),
        "turnover": path["turnover"].sum(),
        "upside_capture": upside,
        "downside_capture": downside,
        "capture_asymmetry": upside - downside,
    }
    result = {key: float(value) if math.isfinite(value) else None for key, value in values.items()}
    result["n_trades"] = int(path["turnover"].gt(0).sum())
    result["max_underwater_sessions"] = int(underwater.groupby((~underwater).cumsum()).sum().max())
    return result


def print_report(results):
    header = ("Period", "Portfolio", "CAGR", "Ann vol", "Sharpe", "Max DD", "Total return",
              "Turnover", "Trades", "Underwater d", "Up capture", "Down capture")
    layout = "{:<12} {:<16} {:>8} {:>8} {:>7} {:>8} {:>12} {:>10} {:>7} {:>12} {:>11} {:>12}"
    print(layout.format(*header))
    fields = (
        ("cagr", ".2%"), ("ann_vol", ".2%"), ("sharpe", ".2f"), ("max_drawdown", ".2%"),
        ("total_return", ".2%"), ("turnover", ".2%"), ("n_trades", "d"),
        ("max_underwater_sessions", "d"), ("upside_capture", ".2%"), ("downside_capture", ".2%"),
    )
    for period in ("is", "oos", "full"):
        for label, row in (("Strategy", results[period]),
                           ("Buy and hold", results["benchmark"][period])):
            formatted = [format(row[key], spec) if row[key] is not None else "n/a" for key, spec in fields]
            print(layout.format(period.upper(), label, *formatted))


def main():
    with (ROOT / "config.yaml").open() as file:
        config = yaml.safe_load(file)
    ticker, = config["universe"]
    parameters = config["allocation"]
    cost_rate = config["one_way_cost_bps"] / 10_000
    sessions_per_year = config["sessions_per_year"]
    data_start = pd.Timestamp(config["data_start"])
    is_start, is_end = (pd.Timestamp(config["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(config["out_of_sample"][key]) for key in ("start", "end"))
    if not data_start < is_start <= is_end < oos_start <= oos_end:
        raise ValueError("Expected warm-up followed by ordered, disjoint IS and OOS dates")
    if not (0 <= parameters["drawdown_start"] < parameters["drawdown_full"] < 1
            and 0 <= parameters["recovery_start"] < parameters["recovery_full"] <= 1
            and parameters["duration_full_sessions"] > 0
            and all(0 <= floor <= 1 for floor in parameters["min_weight_candidates"])
            and 0 <= cost_rate < 1 and sessions_per_year > 0):
        raise ValueError("Invalid allocation, transaction cost, or annualization parameters")

    prices = load_prices(ticker, data_start, oos_end)
    relatives = market_relatives(prices).loc[is_start:oos_end]
    if relatives.loc[:is_end].empty or relatives.loc[oos_start:].empty or relatives.isna().any().any():
        raise ValueError("Need warm-up prices and at least one session in both IS and OOS")
    if not relatives.loc[is_end + pd.Timedelta(days=1):oos_start - pd.Timedelta(days=1)].empty:
        raise ValueError("IS and OOS must cover adjacent trading sessions")

    is_stress = allocation_stress(prices.loc[:is_end, "Close"], parameters).shift().loc[is_start:is_end]
    floor = select_min_weight(relatives.loc[:is_end], is_stress, parameters["min_weight_candidates"],
                              cost_rate, sessions_per_year)
    stress_at_open = allocation_stress(prices["Close"], parameters).shift().reindex(relatives.index)
    strategy = simulate(relatives, 1 - (1 - floor) * stress_at_open, cost_rate)
    benchmark = simulate(relatives, np.ones(len(relatives)), cost_rate)
    windows = {"is": slice(is_start, is_end), "oos": slice(oos_start, oos_end),
               "full": slice(is_start, oos_end)}

    results = {
        "strategy": config["strategy"],
        "data_range": f"{strategy.index[0].date()} to {strategy.index[-1].date()}",
    }
    for name, dates in windows.items():
        results[name] = metrics(strategy.loc[dates], benchmark.loc[dates, "return"], sessions_per_year)
    results["benchmark"] = {"name": f"{ticker} buy and hold"}
    for name, dates in windows.items():
        results["benchmark"][name] = metrics(benchmark.loc[dates], benchmark.loc[dates, "return"],
                                             sessions_per_year)

    print(f"{config['strategy']} | {ticker} | {results['data_range']}")
    print(f"IS {is_start.date()} to {is_end.date()} | OOS {oos_start.date()} to {oos_end.date()}")
    print(f"IS-selected minimum equity weight {floor:.0%} | {config['one_way_cost_bps']:g} bps one way")
    print_report(results)
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

"""472_breaking_the_trend_how_to_avoid_cherry_p
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/472_breaking_the_trend_how_to_avoid_cherry_p.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit e0683ac343a175cb2458afc860f8d1a1066c8e3a on 2026-11-27 08:56 Europe/Oslo, exit 0, output sha256 9fa7880cb8362894dd09717cf9bdcd65b0c364af38a028774917f555ba597aa5
Data source: yfinance 免费日频美国上市 ETF 收盘价。标的为 SPY、EFA、IEF、TLT、GLD、DBC、UUP、FXE，现金代理为 BIL（config.yaml:2-7，strategy.py:36-46、192-195，README.md:15）。运行日志记录的价格下载时间为 2026-11-27T07:06:46+00:00（logs/run_20261127_080852.log:1，strategy.py:26-29、53-59、231）。
Data fetched: 2026-11-27T07:56:23+00:00
Sample period: 配置的下载窗口为 2008-01-01 至 2026-10-31，2008 至 2011 年用于预热，评估窗口为 2012-01-01 至 2026-10-31（config.yaml:8-12，README.md:17，strategy.py:218-224）。日志报告的实际评估交易日范围为 2012-01-03 至 2026-10-30（logs/run_20261127_080852.log:12，strategy.py:236-238）。   data_range (from output): {'start': '2012-01-03', 'end': '2026-10-30'}
IS/OOS split: 按日期切分，IS 为 2012-01-01 至 2023-12-31（含），OOS 为 2024-01-01 至 2026-10-31（含）。先连续模拟，再分段统计，切分处沿用组合状态（config.yaml:8-12，strategy.py:218-226、240-248）。README 记载部分 OOS 日期早于论文首次公开（README.md:17）。
Costs and slippage: 配置显式建模单边 1 bp 交易成本及 2 bps 固定滑点。每次调仓按 ETF 实际交易名义金额绝对值之和扣除合计 3 bps。模拟器以调仓前组合价值为基数计算扣费比例，并迭代使目标权重按扣费后净值计量。基准的首次买入采用相同费率（config.yaml:13-16，strategy.py:139-165、201-216、226-229，README.md:21）。
Signal lag: 用交易日 t 的收盘价计算目标权重，配置将目标后移 1 个交易日，于 t+1 收盘执行。直到 t+1 收盘的收益由旧权重承担，扣除调仓费用后，新权重自下一收益区间起生效（config.yaml:13-14，strategy.py:79-90、133-165、208-209、218-226，README.md:21）。
Price adjustment: yfinance auto_adjust=True total-return adjusted close
Survivorship bias: 八只 ETF 的名单事后固定，覆盖存续且有长历史的基金。README 披露存续偏差和资产选择偏差（config.yaml:2-7，strategy.py:192-195，README.md:25）。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.038585417207515516, ann_vol=0.05478770435153061, sharpe=0.7185013752802843, sharpe_excess=0.5550262906784298, max_drawdown=-0.08397768135521233, total_return=0.5736754256141616, turnover=5.085605736658835, n_trades=24144
    oos: cagr=0.10771048132982042, ann_vol=0.06564504769539736, sharpe=1.5914615375338634, sharpe_excess=0.9386632265187207, max_drawdown=-0.04697540638228814, total_return=0.33458283152837964, turnover=5.277652609306083, n_trades=5688
    full: cagr=0.05142397295818801, ann_vol=0.057030808875734626, sharpe=0.9079084863390342, sharpe_excess=0.6377005461323761, max_drawdown=-0.08397768135521233, total_return=1.1002002054227757, turnover=5.1222228797138625, n_trades=29832
    benchmark: cagr=0.060740543311831985, ann_vol=0.07609381738171149, sharpe=0.8130867160481745, sharpe_excess=0.6100628789576021, max_drawdown=-0.16906578280270168, total_return=1.3930662492999342, turnover=0.06755817180831214, n_trades=8
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 472_breaking_the_trend_how_to_avoid_cherry_p\nuniverse:\n  equity: [SPY, EFA]\n  bonds: [IEF, TLT]\n  commodities: [GLD, DBC]\n  currencies: [UUP, FXE]\ncash_proxy: BIL\ndates:\n  data_start: 2008-01-01\n  start: 2012-01-01\n  is_end: 2023-12-31\n  end: 2026-10-31\nexecution:\n  lag_sessions: 1\n  cost_bps: 1.0\n  slippage_bps: 2.0\nmodel:\n  trend_decay_sessions: 112\n  risk_decay_sessions: 40\n  correlation_decay_weeks: 150\n  min_correlation_weeks: 52\n  correlation_shrinkage: 0.10\n  smoothing_sessions: 20\n  annual_vol_target: 0.10\n  max_gross_weight: 1.0\n  sessions_per_year: 252\n'
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
"""Research backtest for a single-EMA cross-asset trend strategy."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True total-return adjusted close"


def load_prices(tickers, start, end):
    end_exclusive = end + pd.Timedelta(days=1)
    request = json.dumps([sorted(tickers), str(start.date()), str(end_exclusive.date()), PRICE_ADJUSTMENT])
    cache_file = CACHE / f"prices_{hashlib.sha256(request.encode()).hexdigest()[:16]}.csv"
    if cache_file.exists():
        with cache_file.open() as source:
            fetched_at = source.readline().removeprefix("# fetched_at=").strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col=0, parse_dates=True)
        return check_prices(prices, tickers, start, end), fetched_at

    CACHE.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance"))
    error = None
    for _ in range(3):
        try:
            downloaded = yf.download(
                tickers,
                start=str(start.date()),
                end=str(end_exclusive.date()),
                interval="1d",
                auto_adjust=True,
                ignore_tz=True,
                progress=False,
                timeout=8,
            )
            prices = check_prices(downloaded["Close"], tickers, start, end)
            break
        except (KeyError, RuntimeError, ValueError, OSError) as exc:
            error = exc
    else:
        raise RuntimeError(f"Unable to download complete yfinance prices: {error}") from error

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as target:
        temp_path = Path(target.name)
        target.write(f"# fetched_at={fetched_at}\n")
        prices.to_csv(target, index_label="date")
    temp_path.replace(cache_file)
    return prices, fetched_at


def check_prices(prices, tickers, start, end):
    prices = prices.reindex(columns=tickers).sort_index()
    if not isinstance(prices.index, pd.DatetimeIndex) or prices.index.has_duplicates:
        raise ValueError("Expected a unique daily price index")
    prices.index = prices.index.as_unit("ns").rename("date")
    prices.columns.name = None
    prices = prices.loc[start:end]
    if prices.empty:
        raise ValueError("No daily prices in the requested date range")
    invalid = ~np.isfinite(prices) | (prices <= 0)
    if invalid.any().any():
        raise ValueError(f"Missing or invalid prices for {', '.join(invalid.columns[invalid.any()])}")
    if prices.index[0] - start > pd.Timedelta(days=7) or end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError("Price history does not cover the requested date range")
    return prices


def trend_targets(prices, tickers, cash_proxy, model):
    returns = prices.pct_change(fill_method=None)
    excess = returns[tickers].sub(returns[cash_proxy], axis=0)
    trend_days = model["trend_decay_sessions"]
    risk_days = model["risk_decay_sessions"]
    squared = excess.pow(2)
    signal_vol = np.sqrt(squared.ewm(alpha=1 / trend_days, adjust=False, min_periods=trend_days).mean())
    normalized = excess.div(signal_vol.shift(1).where(signal_vol.shift(1) > 0))
    # In equation (6), the newest observation has weight sqrt(alpha).
    signal = normalized.ewm(alpha=1 / trend_days, adjust=False, min_periods=trend_days).mean()
    signal *= np.sqrt(trend_days)
    risk_vol = np.sqrt(squared.ewm(alpha=1 / risk_days, adjust=False, min_periods=risk_days).mean())

    weekly = prices.resample("W-FRI").last().pct_change(fill_method=None)
    weekly_excess = weekly[tickers].sub(weekly[cash_proxy], axis=0)
    weekly_corr = weekly_excess.ewm(
        alpha=1 / model["correlation_decay_weeks"],
        adjust=False,
        min_periods=model["min_correlation_weeks"],
    ).corr()
    # A short week's estimate first becomes usable on or after its Friday label.
    corr_dates = weekly.index.get_indexer(prices.index, method="pad")
    corr_values = weekly_corr.to_numpy().reshape(len(weekly), len(tickers), len(tickers))
    identity = np.eye(len(tickers))
    shrinkage = model["correlation_shrinkage"]
    smoothing = 1 / model["smoothing_sessions"]
    daily_vol_target = model["annual_vol_target"] / np.sqrt(model["sessions_per_year"])
    max_gross = model["max_gross_weight"]
    smoothed = np.zeros(len(tickers))
    targets = np.full(signal.shape, np.nan)

    for row, (current_signal, vol, corr_row) in enumerate(
        zip(signal.to_numpy(), risk_vol.to_numpy(), corr_dates)
    ):
        if corr_row < 0 or not np.isfinite(current_signal).all() or not (vol > 0).all():
            continue
        correlation = corr_values[corr_row]
        if not np.isfinite(correlation).all():
            continue
        correlation = (1 - shrinkage) * correlation + shrinkage * identity
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        decorrelated = eigenvectors @ ((eigenvectors.T @ current_signal) / np.sqrt(eigenvalues))
        smoothed = (1 - smoothing) * smoothed + smoothing * decorrelated / vol
        gross = np.abs(smoothed).sum()
        if gross == 0:
            targets[row] = 0.0
            continue
        risk_units = vol * smoothed
        predicted_vol = np.sqrt(risk_units @ correlation @ risk_units)
        targets[row] = smoothed * min(daily_vol_target / predicted_vol, max_gross / gross)

    return pd.DataFrame(targets, index=prices.index, columns=tickers)


def simulate(returns, cash_returns, targets, friction):
    weights = np.zeros(returns.shape[1])
    rows = []
    for asset_returns, cash_return, target in zip(
        returns.to_numpy(), cash_returns.to_numpy(), targets.to_numpy()
    ):
        asset_values = weights * (1 + asset_returns)
        growth = asset_values.sum() + (1 - weights.sum()) * (1 + cash_return)
        if not np.isfinite(growth) or growth <= 0:
            raise ValueError("Portfolio value became nonpositive or invalid")
        drifted = asset_values / growth
        if np.isnan(target).all():
            weights = drifted
            rows.append((growth - 1, 0.0, 0))
            continue

        # Target weights are fractions of value after the execution charge.
        charge = 0.0
        for _ in range(20):
            traded = target * (1 - charge) - drifted
            turnover = np.abs(traded).sum()
            next_charge = friction * turnover
            if abs(next_charge - charge) < 1e-13:
                break
            charge = next_charge
        else:
            raise ValueError("Execution charge computation did not converge")

        net_growth = growth * (1 - next_charge)
        if net_growth <= 0:
            raise ValueError("Execution charges exhausted portfolio value")
        weights = target.copy()
        rows.append((net_growth - 1, turnover, np.count_nonzero(np.abs(traded) > 1e-10)))

    return pd.DataFrame(rows, index=returns.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, cash_returns, sessions_per_year):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Expected at least two sessions per reporting period")
    excess = returns - cash_returns.loc[returns.index]
    values = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    scale = np.sqrt(sessions_per_year)
    vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    return {
        "cagr": float(values[-1] ** (sessions_per_year / len(returns)) - 1),
        "ann_vol": float(vol * scale),
        "sharpe": float(returns.mean() * scale / vol) if vol > 0 else None,
        "sharpe_excess": float(excess.mean() * scale / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((values / np.maximum.accumulate(values) - 1).min()),
        "total_return": float(values[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * sessions_per_year),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    tickers = [ticker for group in config["universe"].values() for ticker in group]
    cash_proxy = config["cash_proxy"]
    symbols = [*tickers, cash_proxy]
    if not tickers or len(set(symbols)) != len(symbols):
        raise ValueError("The universe and cash proxy must use distinct symbols")
    data_start, start, is_end, end = (
        pd.Timestamp(config["dates"][key]) for key in ("data_start", "start", "is_end", "end")
    )
    execution = config["execution"]
    lag = execution["lag_sessions"]
    cost_bps, slippage_bps = execution["cost_bps"], execution["slippage_bps"]
    model = config["model"]
    sessions_per_year = model["sessions_per_year"]
    if not (data_start < start <= is_end < end):
        raise ValueError("Expected a warm-up period followed by ordered IS and OOS dates")
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Execution lag must be at least one session")
    if min(cost_bps, slippage_bps) < 0 or cost_bps + slippage_bps >= 10_000:
        raise ValueError("Invalid transaction cost or slippage")
    if not (0 < model["correlation_shrinkage"] <= 1 and 0 < model["max_gross_weight"] <= 1):
        raise ValueError("Correlation shrinkage and gross exposure cap must be in (0, 1]")
    if min(model["annual_vol_target"], sessions_per_year) <= 0:
        raise ValueError("Volatility target and annualization factor must be positive")
    friction = (cost_bps + slippage_bps) / 10_000

    prices, fetched_at = load_prices(symbols, data_start, end)
    realized = prices.pct_change(fill_method=None).loc[start:end]
    cash_returns = realized[cash_proxy]
    asset_returns = realized[tickers]
    targets = trend_targets(prices, tickers, cash_proxy, model).shift(lag).loc[realized.index]
    if not np.isfinite(targets).all().all():
        raise ValueError("Insufficient warm-up history for one or more requested sessions")

    daily_results = simulate(asset_returns, cash_returns, targets, friction)
    passive_targets = pd.DataFrame(np.nan, index=targets.index, columns=tickers)
    passive_targets.iloc[0] = 1 / len(tickers)
    passive_results = simulate(asset_returns, cash_returns, passive_targets, friction)

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"EXCESS_SHARPE_REFERENCE: {cash_proxy}")
    print(f"BENCHMARK: Equal-weight buy-and-hold ({', '.join(tickers)})")
    output = {
        "strategy": config["strategy"],
        "data_range": {"start": str(realized.index[0].date()), "end": str(realized.index[-1].date())},
    }
    for name, mask in (
        ("is", realized.index <= is_end),
        ("oos", realized.index > is_end),
        ("full", slice(None)),
    ):
        output[name] = metrics(daily_results.loc[mask], cash_returns, sessions_per_year)
        passive = metrics(passive_results.loc[mask], cash_returns, sessions_per_year)
        print(f"{name.upper()}: {json.dumps(output[name], allow_nan=False)}")
        print(f"BENCHMARK_{name.upper()}: {json.dumps(passive, allow_nan=False)}")
    output["benchmark"] = passive
    print(f"RESULTS_JSON: {json.dumps(output, allow_nan=False)}")


if __name__ == "__main__":
    main()
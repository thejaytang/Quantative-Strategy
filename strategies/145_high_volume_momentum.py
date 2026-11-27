# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 145_high_volume_momentum\n\n# Ex-post survivor basket for a case study; not historical index membership.\nuniverse:\n  - AAPL\n  - MSFT\n  - AMZN\n  - GOOGL\n  - META\n  - NVDA\n  - JPM\n  - BAC\n  - GS\n  - MS\n  - XOM\n  - CVX\n  - COP\n  - SLB\n  - WMT\n  - COST\n  - PG\n  - KO\n  - PEP\n  - JNJ\n  - PFE\n  - MRK\n  - HD\n  - LOW\nbenchmark: SPY\ncash_proxy: BIL\n\ndata_start: '2012-01-01'\nin_sample:\n  start: '2014-01-01'\n  end: '2020-12-31'\nout_of_sample:\n  start: '2021-01-01'\n  end: '2025-12-31'\ntrading_days_per_year: 252\n\nsignal:\n  reference_sessions: 126\n  momentum_sessions: 126\n  volume_median_sessions: 63\n  relative_volume_cap: 3.0\n  activity_sessions: 21\n  min_valid_stocks: 16\n  max_holdings: 6\n\ncommission_bps: 5.0\nslippage_bps: 5.0\nsensitivity_commission_bps: [0.0, 15.0]\n"
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
from __future__ import annotations

import gzip
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
FIELDS = ["Open", "Close", "Volume"]
PORTFOLIOS = (
    "High activity / overhang",
    "High activity / return",
    "Low activity / overhang",
    "Low activity / return",
)


def load_market_data(cfg, start, end):
    tickers = sorted({*cfg["universe"], cfg["benchmark"], cfg["cash_proxy"]})
    columns = pd.MultiIndex.from_product([FIELDS, tickers], names=["Price", "Ticker"])
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    cache_dir = ROOT / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_key = hashlib.sha256(
        repr(("yahoo-adjusted-v1", tickers, str(start.date()), end_exclusive)).encode()
    ).hexdigest()[:16]
    cache_file = cache_dir / f"yahoo_{cache_key}.csv.gz"
    required = [(field, ticker) for field in ("Open", "Close") for ticker in tickers]
    required += [("Volume", ticker) for ticker in cfg["universe"]]

    if cache_file.exists():
        data = pd.read_csv(
            cache_file, header=[0, 1], index_col=0, parse_dates=True,
        ).reindex(columns=columns)
    else:
        yf.set_tz_cache_location(str(cache_dir / "yfinance"))
        data = pd.DataFrame(columns=columns, dtype=float)
        missing = tickers
        for _ in range(3):
            batch = yf.download(
                missing, start=str(start.date()), end=end_exclusive, interval="1d",
                auto_adjust=True, ignore_tz=True, progress=False, threads=8,
                timeout=12, multi_level_index=True,
            )
            if not batch.empty:
                data = batch.reindex(columns=columns).combine_first(data)
            missing = sorted({ticker for field, ticker in required
                              if not data[field, ticker].gt(0).any()})
            if not missing:
                break
        if missing:
            raise RuntimeError(f"Yahoo Finance did not supply required data for {', '.join(missing)}")

    data = data.sort_index().loc[start:end]
    data.index = data.index.astype("datetime64[ns]")
    data = data.where(np.isfinite(data) & data.gt(0))
    required_etfs = data.loc[:, (["Open", "Close"], [cfg["benchmark"], cfg["cash_proxy"]])]
    data = data.loc[required_etfs.notna().any(axis=1)]
    if (data.empty or not data.index.is_unique
            or data.index[0] > start + pd.Timedelta(days=7)
            or data.index[-1] < end - pd.Timedelta(days=7)
            or required_etfs.loc[data.index].isna().any().any()):
        raise RuntimeError("Yahoo Finance benchmark or cash proxy is missing required session prices")
    if not data.loc[:, required].notna().any().all():
        raise RuntimeError("Yahoo Finance input is missing a required price or volume series")

    if not cache_file.exists():
        with NamedTemporaryFile(dir=cache_dir, suffix=".csv.gz", delete=False) as temp:
            temp_file = Path(temp.name)
        try:
            data.to_csv(temp_file, compression="gzip")
            temp_file.replace(cache_file)
        finally:
            temp_file.unlink(missing_ok=True)
    # The gzip header timestamps the snapshot written immediately after download.
    with gzip.open(cache_file, "rb") as cached:
        cached.read(1)
        if not cached.mtime:
            raise RuntimeError("Yahoo Finance cache lacks a fetch timestamp")
        fetched_at = datetime.fromtimestamp(cached.mtime, ZoneInfo("Europe/Oslo"))
    return data, fetched_at


def monthly_targets(data, sessions, cfg):
    params = cfg["signal"]
    universe = sorted(cfg["universe"])
    cash_proxy = cfg["cash_proxy"]
    close = data["Close"][universe]
    volume = data["Volume"][universe]
    relative_volume = (
        volume / volume.rolling(params["volume_median_sessions"]).median()
    ).clip(upper=params["relative_volume_cap"])
    window = params["reference_sessions"]
    reference = ((close * relative_volume).rolling(window).sum()
                 / relative_volume.rolling(window).sum())
    overhang = close / reference - 1
    momentum = close / close.shift(params["momentum_sessions"]) - 1
    activity = relative_volume.shift(1).rolling(params["activity_sessions"]).mean()

    months = sessions.to_period("M")
    rebalance_days = sessions[np.r_[True, months[1:] != months[:-1]]]
    # Use the last session before the execution month, even if evaluation starts mid-month.
    signal_positions = data.index.searchsorted(rebalance_days.to_period("M").to_timestamp()) - 1
    if (signal_positions < 0).any():
        raise ValueError("data_start must provide a prior month-end for the first rebalance")
    targets = {
        label: pd.DataFrame(0.0, index=rebalance_days, columns=[*universe, cash_proxy])
        for label in PORTFOLIOS
    }
    for day, signal_pos in zip(rebalance_days, signal_positions):
        scores = pd.DataFrame({
            "overhang": overhang.iloc[signal_pos],
            "return": momentum.iloc[signal_pos],
            "activity": activity.iloc[signal_pos],
        }).dropna()
        groups = (scores.iloc[:0], scores.iloc[:0])
        if len(scores) >= params["min_valid_stocks"]:
            median = scores["activity"].median()
            rising = scores[(scores["return"] > 0) & (scores["overhang"] > 0)]
            groups = (rising[rising["activity"] > median], rising[rising["activity"] < median])

        slots = params["max_holdings"]
        for label, eligible, rank in zip(
            PORTFOLIOS, (groups[0], groups[0], groups[1], groups[1]),
            ("overhang", "return", "overhang", "return"),
        ):
            chosen = eligible[rank].nlargest(slots, keep="first").index
            targets[label].loc[day, chosen] = 1.0 / slots
            targets[label].loc[day, cash_proxy] = 1.0 - len(chosen) / slots
    return targets


def simulate(opens, closes, targets, cost_bps):
    symbols = targets.columns
    opens = opens[symbols]
    closes = closes[symbols]
    positions = np.zeros(len(symbols))
    marks = np.zeros(len(symbols))
    cash = previous_nav = 1.0
    pending = None
    daily = []
    cost_rate = cost_bps / 10_000

    for day, opening, closing in zip(opens.index, opens.to_numpy(), closes.to_numpy()):
        tradeable = np.isfinite(opening)
        marks = np.where(tradeable, opening, marks)
        turnover, n_trades = 0.0, 0
        if day in targets.index:
            pending = targets.loc[day].to_numpy()

        # Wait for a common open if a held asset or a target lacks an opening price.
        if pending is not None and tradeable[(positions > 0) | (pending > 0)].all():
            holdings = positions * marks
            equity = cash + holdings.sum()
            post_cost_equity = equity
            # Self-financing targets: costs reduce the dollars available to allocate.
            for _ in range(12):
                traded = np.abs(pending * post_cost_equity - holdings).sum()
                post_cost_equity = equity - cost_rate * traded
            allocations = pending * post_cost_equity
            traded_by_asset = np.abs(allocations - holdings)
            turnover = traded_by_asset.sum() / equity
            n_trades = int((traded_by_asset > 1e-10 * equity).sum())
            positions = np.divide(allocations, marks, out=np.zeros_like(positions), where=marks > 0)
            cash = post_cost_equity * max(0.0, 1.0 - pending.sum())
            pending = None

        marks = np.where(np.isfinite(closing), closing, marks)
        nav = cash + (positions * marks).sum()
        daily.append((nav / previous_nav - 1, turnover, n_trades))
        previous_nav = nav

    return pd.DataFrame(daily, index=opens.index, columns=["return", "turnover", "n_trades"])


def metrics(daily, cash_returns, annual_days):
    returns = daily["return"]
    excess = returns - cash_returns.loc[returns.index]
    wealth = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    result = {
        "cagr": wealth[-1] ** (annual_days / len(daily)) - 1,
        "ann_vol": volatility * np.sqrt(annual_days),
        "sharpe": returns.mean() / volatility * np.sqrt(annual_days) if volatility > 0 else None,
        "sharpe_excess": (excess.mean() / excess_volatility * np.sqrt(annual_days)
                          if excess_volatility > 0 else None),
        "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
        "total_return": wealth[-1] - 1,
        "n_trades": int(daily["n_trades"].sum()),
        "turnover": daily["turnover"].mean() * annual_days,
    }
    return {
        key: value if value is not None and np.isfinite(value) else None
        for key, value in result.items()
    }


def print_table(rows):
    def percent(value):
        return f"{value:.2%}" if value is not None else "n/a"

    def ratio(value):
        return f"{value:.2f}" if value is not None else "n/a"

    print(f"{'Portfolio':25} {'bps':>5} {'Period':>6} {'CAGR':>9} {'Ann. vol':>9} "
          f"{'Sharpe':>7} {'Excess':>7} {'Max DD':>9} {'Total ret':>10} {'Trades':>7} {'Ann. t/o':>10}")
    for label, bps, period, m in rows:
        print(f"{label:25} {bps:5g} {period.upper():>6} {percent(m['cagr']):>9} "
              f"{percent(m['ann_vol']):>9} {ratio(m['sharpe']):>7} {ratio(m['sharpe_excess']):>7} "
              f"{percent(m['max_drawdown']):>9} {percent(m['total_return']):>10} "
              f"{m['n_trades']:7d} {percent(m['turnover']):>10}")


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    is_start, is_end = (pd.Timestamp(cfg["in_sample"][key]) for key in ("start", "end"))
    oos_start, oos_end = (pd.Timestamp(cfg["out_of_sample"][key]) for key in ("start", "end"))
    data_start = pd.Timestamp(cfg["data_start"])
    if not (data_start < is_start <= is_end and
            is_end + pd.Timedelta(days=1) == oos_start <= oos_end):
        raise ValueError("Configure ordered, adjacent IS and OOS date ranges after data_start")

    commissions = dict.fromkeys([cfg["commission_bps"], *cfg["sensitivity_commission_bps"]])
    slippage = cfg["slippage_bps"]
    if not 0 <= slippage <= 500 or not all(0 <= bps <= 500 - slippage for bps in commissions):
        raise ValueError("Commission and slippage must be nonnegative and total at most 500 bps")
    data, fetched_at = load_market_data(cfg, data_start, oos_end)
    opens = data["Open"].loc[is_start:oos_end]
    closes = data["Close"].reindex(opens.index)
    dates = {"is": (is_start, is_end), "oos": (oos_start, oos_end), "full": (is_start, oos_end)}
    if any(opens.loc[start:end].empty for start, end in dates.values()):
        raise ValueError("Each evaluation period must contain at least one session")

    # The cash hurdle is a frictionless BIL investment starting at the first open.
    cash_proxy = cfg["cash_proxy"]
    cash_close = closes[cash_proxy]
    cash_base = cash_close.shift()
    cash_base.iloc[0] = opens[cash_proxy].iloc[0]
    cash_returns = cash_close / cash_base - 1

    targets = monthly_targets(data, opens.index, cfg)
    benchmark_label = f"{cfg['benchmark']} buy and hold"
    targets[benchmark_label] = pd.DataFrame(
        1.0, index=opens.index[:1], columns=[cfg["benchmark"]]
    )
    data_range = f"{opens.index[0].date()} to {opens.index[-1].date()}"
    print(f"{cfg['strategy']} | {data_range}")
    print(f"UNIVERSE: retrospective case study of {len(cfg['universe'])} stocks with survivorship bias")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat()} (Yahoo Finance cache snapshot, Europe/Oslo)")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True; split/dividend-adjusted Open and Close; Yahoo Volume as supplied")
    print(f"COSTS: {cfg['commission_bps']:g} bps commission + {slippage:g} bps slippage per side; table bps are total")
    print(f"EXCESS_SHARPE_CASH: {cash_proxy} gross total return")
    for period, (start, end) in dates.items():
        index = opens.loc[start:end].index
        print(f"{period.upper():>4} {index[0].date()} to {index[-1].date()}")

    rows = []
    baseline = {}
    for commission in commissions:
        # Both deductions are proportional to the quoted opening notional.
        bps = commission + slippage
        for label, orders in targets.items():
            daily = simulate(opens, closes, orders, bps)
            summary = {period: metrics(daily.loc[start:end], cash_returns, cfg["trading_days_per_year"])
                       for period, (start, end) in dates.items()}
            rows.extend((label, bps, period, m) for period, m in summary.items())
            if commission == cfg["commission_bps"]:
                baseline[label] = summary
    print_table(rows)
    results = {
        "strategy": cfg["strategy"],
        "data_range": data_range,
        **baseline[PORTFOLIOS[0]],
        "benchmark": {"ticker": cfg["benchmark"], **baseline[benchmark_label]},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

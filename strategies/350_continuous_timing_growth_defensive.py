# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 350_continuous_timing_growth_defensive\n\nuniverse:\n  growth: [QQQ, XLK, VGT, SPYG, VUG]\n  defensive: [SCHD, VYM, VTV, FDVV, COWZ]\n  market: SPY\n  volatility: ^VIX\n  risk_free: BIL\n  treasury_series: DGS10\n\ndates:\n  history_start: 2013-01-02\n  backtest_start: 2017-10-02\n  is_end: 2018-06-27\n  end: 2026-05-15\n\nsignal:\n  rate_change_sessions: 21\n  vix_change_sessions: 21\n  vix_percentile_sessions: 756\n  crowding_sessions: 126\n  zscore_min_observations: 21\n  softplus_temperature: 1.0\n  # Paper v1, equation (32), selected on the author's first training window.\n  alpha: 0.50\n  stress_weight: 0.50\n  crowding_weight: 0.25\n  max_tilt: 0.20\n  weight_temperature: 1.50\n  ewma_alpha: 0.03\n\nexecution:\n  signal_lag_sessions: 1\n  treasury_publication_lag_sessions: 1\n  vix_publication_lag_sessions: 1\n  costs_bps: 10.0\n  slippage_bps: 2.0\n\nmetrics:\n  sessions_per_year: 252\n"
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
"""C006 continuous Growth–Defensive ETF allocation research backtest."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust total-return ETF opens and closes, ^VIX index level"


def cached_frame(source, request_key, fetch):
    digest = hashlib.sha256(repr(request_key).encode()).hexdigest()[:16]
    path = CACHE / f"{source}_{digest}.csv"
    if path.exists():
        with path.open() as cached:
            fetched_at = cached.readline().strip()
            frame = pd.read_csv(cached, index_col=0, parse_dates=True)
        return frame, fetched_at

    frame = fetch()
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as staged:
        staged_path = Path(staged.name)
        try:
            staged.write(fetched_at + "\n")
            frame.to_csv(staged, index_label="date")
            staged.close()
            staged_path.replace(path)
        finally:
            staged_path.unlink(missing_ok=True)
    return frame, fetched_at


def yahoo_prices(tickers, start, end):
    with TemporaryDirectory(dir=CACHE, prefix="yfinance_") as transport_cache:
        yf.set_tz_cache_location(transport_cache)
        prices = yf.download(
            tickers, start=str(start.date()), end=str((end + pd.Timedelta(days=1)).date()),
            auto_adjust=True, actions=False, group_by="column", multi_level_index=True,
            progress=False, timeout=15,
        )
    if prices.empty:
        raise RuntimeError("Yahoo Finance returned no prices.")
    columns = [(field, ticker) for field in ("Open", "Close") for ticker in tickers]
    prices = prices.reindex(columns=pd.MultiIndex.from_tuples(columns))
    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no usable prices for {missing}.")
    prices.columns = [f"{field}:{ticker}" for field, ticker in columns]
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    return prices.sort_index()


def fred_yield(series, start, end):
    query = urlencode({"id": series, "cosd": start.date(), "coed": end.date()})
    with urlopen(f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}", timeout=20) as response:
        data = pd.read_csv(response, index_col=0, parse_dates=True, na_values=".")
    data = data[[series]].dropna().sort_index()
    if data.empty or data.index[-1] < end - pd.Timedelta(days=7):
        raise RuntimeError(f"FRED {series} does not cover the requested end date.")
    return data


def expanding_zscore(series, min_observations):
    history = series.expanding(min_periods=min_observations)
    scale = history.std(ddof=0)
    return (series - history.mean()) / scale.where(scale > 0)


def growth_signal(closes, treasury, universe, signal, execution):
    def zscore(series):
        return expanding_zscore(series, signal["zscore_min_observations"])

    def softplus(series):
        temperature = signal["softplus_temperature"]
        with np.errstate(invalid="ignore"):  # Preserve missing observations during warmup.
            return temperature * np.logaddexp(0.0, series / temperature)

    rate = treasury.reindex(closes.index, method="ffill")
    rate = rate.shift(execution["treasury_publication_lag_sessions"])
    vix = closes[universe["volatility"]].shift(execution["vix_publication_lag_sessions"])
    spy = closes[universe["market"]]
    drawdown = spy / spy.cummax() - 1.0
    vix_percentile = vix.rolling(signal["vix_percentile_sessions"]).rank(pct=True)

    daily = closes.pct_change(fill_method=None)
    growth = daily[universe["growth"]].mean(axis=1, skipna=False)
    defensive = daily[universe["defensive"]].mean(axis=1, skipna=False)
    trailing = np.expm1(np.log1p(growth - defensive).rolling(signal["crowding_sessions"]).sum())

    rate_relief = -zscore(rate.diff(signal["rate_change_sessions"]))
    drawdown_depth = -zscore(drawdown)
    vix_high = zscore(vix_percentile)
    vix_relief = -zscore(vix.diff(signal["vix_change_sessions"]))
    growth_extension = zscore(trailing)

    rate_vix = rate_relief * vix_high
    stress_relief = softplus(vix_high) * softplus(vix_relief)
    crowded = softplus(growth_extension) * softplus(-vix_high)
    crowded_quiet = crowded * np.exp(-0.5 * rate_relief.pow(2))
    core = signal["alpha"] * rate_relief + (1.0 - signal["alpha"]) * drawdown_depth
    stress = (zscore(rate_vix) + zscore(stress_relief)) / 2.0
    crowding = (zscore(crowded) + zscore(crowded_quiet)) / 2.0
    score = zscore(core + signal["stress_weight"] * stress - signal["crowding_weight"] * crowding)
    target = 0.5 + signal["max_tilt"] * np.tanh(score / signal["weight_temperature"])

    # A neutral initial weight, followed by updates only when a score is available.
    smoothed = np.full(len(target), np.nan)
    weight = 0.5
    for i, desired in enumerate(target):
        if np.isfinite(desired):
            weight += signal["ewma_alpha"] * (desired - weight)
            smoothed[i] = weight
    return pd.Series(smoothed, index=closes.index).ffill()


def nav_after_fees(holdings, target, fee_rate):
    """Solve for post-fee NAV as a fraction of pre-trade opening NAV."""
    if fee_rate == 0:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(42):
        nav = (low + high) / 2.0
        gross_traded = np.abs(nav * target - holdings).sum()
        if nav + fee_rate * gross_traded > 1.0:
            high = nav
        else:
            low = nav
    return (low + high) / 2.0


def simulate(targets, opens, closes, fee_rate):
    dates = targets.index
    overnight = (opens / closes.shift(1)).loc[dates].to_numpy()
    intraday = (closes / opens).loc[dates].to_numpy()
    if not np.isfinite(intraday).all() or not np.isfinite(overnight[1:]).all():
        raise RuntimeError("Missing ETF prices in the execution window.")

    returns, turnovers, trades = [], [], []
    close_weights = np.zeros(targets.shape[1])
    for i, target in enumerate(targets.to_numpy()):
        # Start in cash at the first opening. Thereafter old shares earn the overnight move.
        pre_trade = close_weights if i == 0 else close_weights * overnight[i]
        open_nav = 1.0 if i == 0 else pre_trade.sum()
        holdings = pre_trade / open_nav
        post_fee_nav = nav_after_fees(holdings, target, fee_rate)
        new_holdings = post_fee_nav * target
        traded = np.abs(new_holdings - holdings)
        closing_values = new_holdings * intraday[i]
        close_nav = closing_values.sum()
        close_weights = closing_values / close_nav
        returns.append(open_nav * close_nav - 1.0)
        turnovers.append(traded.sum())
        trades.append(np.count_nonzero(traded > 1e-10))

    return pd.DataFrame({"return": returns, "turnover": turnovers, "trades": trades}, index=dates)


def period_metrics(daily_results, risk_free, sessions_per_year):
    returns = daily_results["return"].to_numpy()
    excess = returns - risk_free.loc[daily_results.index].to_numpy()
    wealth = np.r_[1.0, np.cumprod(1.0 + returns)]
    annualizer = np.sqrt(sessions_per_year)
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    return {
        "cagr": float(wealth[-1] ** (sessions_per_year / len(returns)) - 1.0),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() * annualizer / volatility) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() * annualizer / excess_volatility) if excess_volatility > 0 else None,
        "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1.0).min()),
        "total_return": float(wealth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    history_start, backtest_start, is_end, end = (
        pd.Timestamp(config["dates"][name]) for name in ("history_start", "backtest_start", "is_end", "end")
    )
    if not history_start < backtest_start <= is_end < end:
        raise ValueError("Dates must order history_start < backtest_start <= is_end < end.")

    universe = config["universe"]
    growth, defensive = universe["growth"], universe["defensive"]
    assets = growth + defensive
    if not growth or not defensive or len(assets) != len(set(assets)):
        raise ValueError("Growth and Defensive must be nonempty baskets of distinct tickers.")
    tickers = list(dict.fromkeys(assets + [universe[name] for name in ("market", "volatility", "risk_free")]))
    execution = config["execution"]
    for setting in ("signal_lag_sessions", "treasury_publication_lag_sessions", "vix_publication_lag_sessions"):
        if not isinstance(execution[setting], int) or execution[setting] < 1:
            raise ValueError(f"{setting} must be at least one session.")
    costs = [execution["costs_bps"], execution["slippage_bps"]]
    if not all(cost >= 0 for cost in costs) or sum(costs) >= 10_000:
        raise ValueError("Costs and slippage must be nonnegative and total less than 10000 bps.")

    CACHE.mkdir(exist_ok=True)
    prices, yahoo_fetched_at = cached_frame(
        "yahoo", (tuple(tickers), str(history_start.date()), str(end.date()), "adjusted_ohlc"),
        lambda: yahoo_prices(tickers, history_start, end),
    )
    series = universe["treasury_series"]
    fred, fred_fetched_at = cached_frame(
        "fred", (series, str(history_start.date()), str(end.date())),
        lambda: fred_yield(series, history_start, end),
    )
    closes = prices[[f"Close:{ticker}" for ticker in tickers]].set_axis(tickers, axis=1)
    opens = prices[[f"Open:{ticker}" for ticker in tickers]].set_axis(tickers, axis=1)
    closes = closes.loc[history_start:end].dropna(subset=[universe["market"]])
    opens = opens.reindex(closes.index)
    if closes.empty or closes.index[-1] < end - pd.Timedelta(days=7):
        raise RuntimeError("Yahoo Finance does not cover the requested end date.")
    dates = closes.loc[backtest_start:end].index
    is_dates, oos_dates = dates[dates <= is_end], dates[dates > is_end]
    if len(is_dates) < 2 or len(oos_dates) < 2:
        raise RuntimeError("Both performance periods require at least two sessions.")

    weights = growth_signal(closes, fred[series], universe, config["signal"], execution)
    weights = weights.shift(execution["signal_lag_sessions"]).loc[dates]
    if weights.isna().any():
        raise RuntimeError("Signal warmup is incomplete. Move backtest_start later.")
    targets = pd.DataFrame({ticker: weights / len(growth) for ticker in growth})
    for ticker in defensive:
        targets[ticker] = (1.0 - weights) / len(defensive)
    benchmark_targets = pd.DataFrame(
        {ticker: 0.5 / len(group) for group in (growth, defensive) for ticker in group}, index=dates,
    )
    fee_rate = sum(costs) / 10_000
    strategy_ledger = simulate(targets, opens[assets], closes[assets], fee_rate)
    benchmark_ledger = simulate(benchmark_targets, opens[assets], closes[assets], fee_rate)

    bil = universe["risk_free"]
    risk_free = closes[bil].pct_change(fill_method=None).loc[dates].copy()
    risk_free.iloc[0] = closes.loc[dates[0], bil] / opens.loc[dates[0], bil] - 1.0
    if not np.isfinite(risk_free).all():
        raise RuntimeError(f"Missing {bil} returns in the performance window.")
    periods = {"is": is_dates, "oos": oos_dates, "full": dates}
    annualization = config["metrics"]["sessions_per_year"]
    strategy_metrics = {name: period_metrics(strategy_ledger.loc[idx], risk_free, annualization)
                        for name, idx in periods.items()}
    benchmark = {name: period_metrics(benchmark_ledger.loc[idx], risk_free, annualization)
                 for name, idx in periods.items()}
    benchmark["name"] = "50/50 Growth–Defensive"

    print(f"DATA_FETCHED_AT: {max(yahoo_fetched_at, fred_fetched_at)}")
    print(f"YAHOO_FETCHED_AT: {yahoo_fetched_at}")
    print(f"FRED_FETCHED_AT: {fred_fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print("PARAM_COMBOS_TRIED: 1")
    for name in periods:
        print(f"{name.upper()}: {json.dumps(strategy_metrics[name], allow_nan=False)}")
        print(f"BENCHMARK 50/50 {name.upper()}: {json.dumps(benchmark[name], allow_nan=False)}")
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(dates[0].date()), "end": str(dates[-1].date()),
                       "is_end": str(is_dates[-1].date()), "oos_start": str(oos_dates[0].date())},
        **strategy_metrics,
        "benchmark": benchmark,
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

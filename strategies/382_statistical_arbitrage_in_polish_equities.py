# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 382_statistical_arbitrage_in_polish_equities\nuniverse:\n  - ACP.WA\n  - BDX.WA\n  - CDR.WA\n  - CPS.WA\n  - ENA.WA\n  - EUR.WA\n  - GPW.WA\n  - ING.WA\n  - JSW.WA\n  - KGH.WA\n  - KTY.WA\n  - LPP.WA\n  - MIL.WA\n  - NEU.WA\n  - OPL.WA\n  - PEO.WA\n  - PGE.WA\n  - PKN.WA\n  - PKO.WA\n  - PZU.WA\n  - TPE.WA\ndates:\n  warmup_start: 2015-10-01\n  start: 2017-01-01\n  oos_start: 2020-01-01\n  end: 2020-12-31  # Inclusive\nmodel:\n  pca_window: 252\n  explained_variance: 0.55\n  residual_window: 120\n  min_kappa: 4.0\n  entry_z: 1.10\n  exit_cross_z: 0.50  # A long exits above +0.50, a short exits below -0.50.\nexecution:\n  signal_lag_sessions: 1\n  cost_bps: 10.0\n  slippage_bps: 5.0\n  gross_entry_budget: 2.0  # Per-spread gross entry budget is this divided by universe size.\nreporting:\n  sessions_per_year: 252\n  excess_reference: BIL\n'
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
"""PCA spread and Ornstein–Uhlenbeck backtest for Warsaw-listed equities."""

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"


def load_prices(tickers, reference, dates):
    """Keep only Warsaw sessions with real volume and preserve BIL's own calendar."""
    symbols = [*tickers, reference]
    start = pd.Timestamp(dates["warmup_start"])
    end = pd.Timestamp(dates["end"])
    cache_key = f"adjusted-close-v1|{','.join(symbols)}|{start.date()}|{end.date()}"
    path = CACHE / f"{hashlib.sha256(cache_key.encode()).hexdigest()[:20]}.csv"
    if path.exists():
        with path.open() as saved:
            fetched_at = saved.readline().strip()
            datetime.fromisoformat(fetched_at)
            combined = pd.read_csv(saved, index_col=0, parse_dates=True, dtype=str).astype(float)
        return combined[tickers].dropna(how="all"), combined[reference].dropna(), fetched_at

    CACHE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=CACHE) as scratch:
        yf.set_tz_cache_location(scratch)
        raw = yf.download(
            symbols, start=start.date().isoformat(), end=(end + pd.Timedelta(days=1)).date().isoformat(),
            interval="1d", auto_adjust=True, ignore_tz=True, progress=False, threads=6, timeout=10,
            multi_level_index=True,
        )
    if raw.empty or not {"Close", "Volume"}.issubset(raw.columns.get_level_values(0)):
        raise RuntimeError("Yahoo Finance did not return daily prices and volume.")
    close = raw["Close"].reindex(columns=symbols)
    close = close.where(np.isfinite(close) & close.gt(0)).sort_index()
    missing = close.columns[close.isna().all()].tolist()
    if missing:
        raise RuntimeError(f"No Yahoo Finance close history for {', '.join(missing)}.")

    real_bars = raw["Volume"].reindex(index=close.index, columns=tickers).gt(0)
    sessions = real_bars.any(axis=1) & (close.index.day_of_week < 5)
    stocks = close.loc[sessions, tickers].where(real_bars.loc[sessions])
    if stocks.empty or stocks.index[-1] < end - pd.Timedelta(days=7):
        raise RuntimeError("Warsaw price history ends before the configured sample.")
    gaps = stocks.isna().sum()
    if gaps.any():
        raise RuntimeError(f"Missing positive-volume Warsaw bars by ticker: {gaps[gaps > 0].to_dict()}")
    bil = close[reference].dropna()
    if bil.index[-1] < end - pd.Timedelta(days=7):
        raise RuntimeError(f"{reference} price history ends before the configured sample.")

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Write data and provenance together so an interrupted download cannot leave a valid cache entry.
    with tempfile.NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            temp.write(fetched_at + "\n")
            stocks.join(bil, how="outer").to_csv(temp)
            temp.close()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return stocks, bil, fetched_at


def fit_eigenportfolios(history, variance_target):
    vol = history.std(axis=0, ddof=1)
    if np.any(vol <= 0):
        raise RuntimeError("A Warsaw stock has zero volatility in the PCA window.")
    standardized = (history - history.mean(axis=0)) / vol
    correlation = standardized.T @ standardized / (len(history) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    explained = np.cumsum(eigenvalues[::-1]) / eigenvalues.sum()
    count = np.searchsorted(explained, variance_target) + 1
    if count >= history.shape[1]:
        raise RuntimeError("The PCA variance target leaves no residual dimension.")
    return eigenvectors[:, ::-1][:, :count] / vol[:, None]


def fit_spreads(returns, eigenportfolios, sessions_per_year):
    factors = returns @ eigenportfolios
    factors -= factors.mean(axis=0)
    centered_returns = returns - returns.mean(axis=0)
    betas = np.linalg.lstsq(factors, centered_returns, rcond=None)[0]
    residuals = centered_returns - factors @ betas
    # Each row is one stock minus its PCA hedge expressed in the underlying stocks.
    spreads = np.eye(returns.shape[1]) - (eigenportfolios @ betas).T

    spread_level = residuals.cumsum(axis=0)
    centered = spread_level - spread_level.mean(axis=0)
    lagged, current = centered[:-1], centered[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = (lagged * current).sum(axis=0) / (centered ** 2).sum(axis=0)
        phi = np.where((phi > 0) & (phi < 1), phi, np.nan)
        kappa = -np.log(phi) * sessions_per_year
        innovation = current - phi * lagged
        equilibrium_std = innovation.std(axis=0, ddof=1) / np.sqrt(1 - phi ** 2)
        score = centered[-1] / equilibrium_std
    return spreads, score, kappa


def simulate(prices, first_day, model, execution, sessions_per_year):
    values = prices.to_numpy()
    returns = values[1:] / values[:-1] - 1.0
    calendar = prices.index
    universe_size = len(prices.columns)
    lag = execution["signal_lag_sessions"]
    sleeve_budget = execution["gross_entry_budget"] / universe_size
    fee = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    start_index = calendar.searchsorted(first_day)
    pca_window, residual_window = model["pca_window"], model["residual_window"]
    if start_index < max(pca_window + 1, residual_window):
        raise RuntimeError("Warm-up does not cover the configured model windows.")

    direction = np.zeros(universe_size, dtype=int)
    holdings = np.zeros((universe_size, universe_size))
    # Each pending instruction stores a fill session and the hedge observed on the signal session.
    pending = {}
    rows = []
    cash = previous_nav = 1.0
    fitted_year = None

    for i in range(start_index, len(calendar)):
        price, day = values[i], calendar[i]
        pre_trade_nav = cash + np.sum(holdings * price)
        if not np.isfinite(pre_trade_nav) or pre_trade_nav <= 0:
            raise RuntimeError(f"Non-positive or invalid portfolio NAV on {day.date()}.")

        traded = 0.0
        fills = 0
        for stock, (fill_index, new_direction, weights) in list(pending.items()):
            if fill_index != i:
                continue
            new_units = new_direction * sleeve_budget * pre_trade_nav * weights / price
            delta = (new_units - holdings[stock]) * price
            traded += np.abs(delta).sum()
            fills += np.count_nonzero(delta)
            cash -= delta.sum()
            holdings[stock] = new_units
            direction[stock] = new_direction
            del pending[stock]

        cash -= fee * traded
        nav = pre_trade_nav - fee * traded
        if nav <= 0:
            raise RuntimeError(f"Execution costs exhausted NAV on {day.date()}.")
        rows.append((nav / previous_nav - 1, traded / pre_trade_nav, fills))
        previous_nav = nav

        if fitted_year != day.year:
            # returns[i - 1] ends at today's close and is excluded from the annual PCA fit.
            eigenportfolios = fit_eigenportfolios(
                returns[i - 1 - pca_window:i - 1], model["explained_variance"]
            )
            fitted_year = day.year
        spreads, scores, speeds = fit_spreads(
            returns[i - residual_window:i], eigenportfolios, sessions_per_year
        )
        entry, exit_cross = model["entry_z"], model["exit_cross_z"]
        for stock, score in enumerate(scores):
            if stock in pending or not np.isfinite(score):
                continue
            side = direction[stock]
            if side:
                if side * score > exit_cross:
                    pending[stock] = (i + lag, 0, np.zeros(universe_size))
            elif abs(score) > entry and speeds[stock] > model["min_kappa"]:
                weights = spreads[stock]
                gross = np.abs(weights).sum()
                if gross > 1e-12:
                    pending[stock] = (i + lag, -int(np.sign(score)), weights / gross)

    return pd.DataFrame(rows, index=calendar[start_index:], columns=["return", "turnover", "n_trades"])


def buy_and_hold(prices, fee):
    exposure = 1 / (1 + fee)
    nav = exposure * prices.div(prices.iloc[0]).mean(axis=1)
    result = nav.pct_change().to_frame("return").assign(turnover=0.0, n_trades=0)
    result.iloc[0] = (nav.iloc[0] - 1, exposure, prices.shape[1])
    return result


def metrics(period, reference_returns, sessions_per_year):
    returns = period["return"].to_numpy()
    excess = returns - reference_returns.loc[period.index].to_numpy()
    nav = np.r_[1.0, np.cumprod(1 + returns)]
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    scale = np.sqrt(sessions_per_year)
    return {
        "cagr": float(nav[-1] ** (sessions_per_year / len(returns)) - 1),
        "ann_vol": float(volatility * scale),
        "sharpe": float(returns.mean() / volatility * scale) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * scale) if excess_volatility > 0 else None,
        "max_drawdown": float(np.min(nav / np.maximum.accumulate(nav) - 1)),
        "total_return": float(nav[-1] - 1),
        "turnover": float(period["turnover"].mean() * sessions_per_year),
        "n_trades": int(period["n_trades"].sum()),
    }


def main():
    with (ROOT / "config.yaml").open() as source:
        config = yaml.safe_load(source)
    dates, model, execution = (config[key] for key in ("dates", "model", "execution"))
    tickers = config["universe"]
    reference = config["reporting"]["excess_reference"]
    sessions = config["reporting"]["sessions_per_year"]
    lag = execution["signal_lag_sessions"]
    first, split, end = (pd.Timestamp(dates[key]) for key in ("start", "oos_start", "end"))
    if not pd.Timestamp(dates["warmup_start"]) < first < split <= end:
        raise ValueError("Expected warmup_start < start < oos_start <= end.")
    if len(set(tickers)) != len(tickers) or len(tickers) < 2 or any(not t.endswith(".WA") for t in tickers):
        raise ValueError("Use at least two distinct Warsaw stock tickers ending in .WA.")
    if reference != "BIL" or not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError("Use BIL as the excess-return reference and a positive integer session lag.")
    if min(execution["cost_bps"], execution["slippage_bps"]) < 0 or execution["gross_entry_budget"] <= 0:
        raise ValueError("Execution rates must be nonnegative and the gross budget must be positive.")

    prices, bil, fetched_at = load_prices(tickers, reference, dates)
    prices = prices.loc[:end]
    strategy = simulate(prices, first, model, execution, sessions)
    fee = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    benchmark = buy_and_hold(prices.loc[strategy.index], fee)
    # A same-date US close follows the Warsaw close, so use the last earlier US bar.
    reference_mark = bil.reindex(prices.index - pd.Timedelta(days=1), method="ffill")
    reference_mark.index = prices.index
    reference_returns = reference_mark.pct_change().reindex(strategy.index)
    if reference_returns.isna().any():
        raise RuntimeError(f"Missing prior-close {reference} returns within the evaluation period.")

    def summarize(frame):
        periods = {"is": frame.loc[frame.index < split], "oos": frame.loc[frame.index >= split], "full": frame}
        if any(len(period) < 2 for period in periods.values()):
            raise RuntimeError("IS and OOS must each contain at least two Warsaw sessions.")
        return {name: metrics(period, reference_returns, sessions) for name, period in periods.items()}

    report = summarize(strategy)
    benchmark_report = summarize(benchmark)
    data_range = f"{strategy.index[0].date()} to {strategy.index[-1].date()}"
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close")
    print("PARAM_COMBOS_TRIED: 1")
    print(f"DATA_RANGE: {data_range}")
    print(f"EXCESS_REFERENCE: {reference} (USD; last US close before each Warsaw session)")
    for label, group in (("Strategy", report), ("Benchmark (equal-weight Warsaw buy-and-hold)", benchmark_report)):
        for period, stats in group.items():
            print(f"{label} {period.upper()}: " + json.dumps(stats, allow_nan=False))
    result = {
        "strategy": config["strategy"],
        "data_range": data_range,
        **report,
        "benchmark": {"name": "equal_weight_warsaw_buy_and_hold", **benchmark_report},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

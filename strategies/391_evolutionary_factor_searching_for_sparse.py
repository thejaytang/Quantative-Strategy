# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 391_evolutionary_factor_searching_for_sparse\n\n# US50 from arXiv:2507.17211v1, Appendix C, Table 6.\nuniverse:\n  - AAPL\n  - ADI\n  - ADBE\n  - ADP\n  - ADSK\n  - AMD\n  - AMAT\n  - AMGN\n  - AMZN\n  - APP\n  - ASML\n  - AVGO\n  - BA\n  - BKNG\n  - CDNS\n  - CMCSA\n  - COST\n  - CSCO\n  - CSX\n  - CTAS\n  - FTNT\n  - GILD\n  - GOOG\n  - BRK-B\n  - HON\n  - INTC\n  - INTU\n  - ISRG\n  - KLAC\n  - LIN\n  - LRCX\n  - MELI\n  - META\n  - MRVL\n  - MSFT\n  - MU\n  - NFLX\n  - NVDA\n  - PANW\n  - PEP\n  - PLTR\n  - QCOM\n  - ROP\n  - SBUX\n  - SNPS\n  - TMUS\n  - TSLA\n  - TSM\n  - TXN\n  - VRTX\nbenchmark: SPY\nrisk_free: BIL\n\ndates:\n  history_start: 2016-01-01\n  start: 2019-01-01\n  is_end: 2022-12-31\n  end: 2025-12-31\n\nexecution:\n  signal_lag_sessions: 1\n  cost_bps: 5\n  slippage_bps: 5\n\nsearch:\n  training_sessions: 252\n  refit_every_months: 3\n  seed_windows: [63, 126, 189]\n  mutation_windows: [42, 63, 84, 105, 126, 147, 168, 189, 210, 231, 252, 273]\n  generations: 2\n  mutations_per_generation: 4\n  crossover_parent_pool: 8\n  crossovers_per_generation: 6\n  max_factors: 10\n  min_rank_ic: 0.005\n  redundancy_penalty: 0.05\n  portfolio_size: 10\n'
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
"""C040 price-only evolutionary factor search with a frozen out-of-sample model."""

from datetime import UTC, datetime
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
ANNUAL_SESSIONS = 252
INDICATORS = ("momentum", "sma_ratio", "sharpe", "positive_days")


def load_config():
    with (ROOT / "config.yaml").open() as stream:
        config = yaml.safe_load(stream)
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    if not dates["history_start"] < dates["start"] <= dates["is_end"] < dates["end"]:
        raise ValueError("Expected history_start < start <= is_end < end.")
    config["dates"] = dates

    execution = config["execution"]
    lag = execution["signal_lag_sessions"]
    fees = [execution["cost_bps"], execution["slippage_bps"]]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be a positive integer.")
    if any(not np.isfinite(fee) or fee < 0 for fee in fees) or sum(fees) >= 10_000:
        raise ValueError("Costs and slippage must be nonnegative and total under 10000 bps.")

    assets = config["universe"]
    tickers = [*assets, config["benchmark"], config["risk_free"]]
    if len(tickers) != len(set(tickers)) or any(
        not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", symbol) for symbol in tickers
    ):
        raise ValueError("Use distinct U.S.-listed share and ETF tickers.")
    if config["risk_free"] != "BIL":
        raise ValueError("This implementation uses BIL as its risk-free proxy.")
    search = config["search"]
    if not 1 <= search["portfolio_size"] <= len(assets):
        raise ValueError("portfolio_size is outside the configured universe.")
    if not set(search["seed_windows"]) <= set(search["mutation_windows"]):
        raise ValueError("mutation_windows must include all seed_windows.")
    if search["training_sessions"] < 2 or search["refit_every_months"] < 1:
        raise ValueError("Use at least two training sessions and a positive refit interval.")
    return config


def adjusted_prices(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    query = ("adjusted_close", sorted(tickers), str(start.date()), str(end.date()))
    fingerprint = hashlib.sha256(json.dumps(query).encode()).hexdigest()[:20]
    path = CACHE / f"yfinance-{fingerprint}.csv.gz"
    if path.exists():
        with gzip.open(path, "rt") as stream:
            fetched_at = stream.readline().strip()
            prices = pd.read_csv(stream, index_col="Date", parse_dates=["Date"])
        return prices.reindex(columns=tickers), fetched_at

    with TemporaryDirectory(dir=CACHE, prefix=".yfinance-") as transport_cache:
        yf.set_tz_cache_location(transport_cache)
        prices = download_closes(tickers, start, end)
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=CACHE, suffix=".csv.gz", delete=False) as temporary:
        temp_path = Path(temporary.name)
    try:
        with gzip.open(temp_path, "wt") as stream:
            stream.write(fetched_at + "\n")
            prices.to_csv(stream)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return prices, fetched_at


def download_closes(tickers, start, end):
    prices = pd.DataFrame()
    pending = tickers
    for _ in range(3):
        result = yf.download(
            pending,
            start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True,
            progress=False,
            threads=8,
            timeout=12,
        )
        if "Close" in result:
            close = result["Close"].dropna(axis=1, how="all")
            prices = pd.concat([prices, close], axis=1)
        pending = [ticker for ticker in tickers if ticker not in prices]
        if not pending:
            break
    if pending:
        raise RuntimeError(f"Yahoo Finance returned no daily prices for {', '.join(pending)}")

    prices = prices.sort_index().reindex(columns=tickers)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index.name = "Date"
    return prices.loc[start:end]


def centered_unit_rows(frame):
    centered = frame.sub(frame.mean(axis=1), axis=0).fillna(0.0).to_numpy()
    scale = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, scale, out=np.zeros_like(centered), where=scale > 0)


class FactorSearch:
    def __init__(self, prices, lag, settings, is_end_position):
        self.prices = prices
        self.returns = prices.pct_change(fill_method=None)
        self.settings = settings
        self.lag = lag
        max_window = max(settings["mutation_windows"])
        self.eligible = prices.notna().rolling(max_window + 1).sum().eq(max_window + 1)
        fit_prices = prices.iloc[:is_end_position + 1]
        fit_eligible = self.eligible.iloc[:is_end_position + 1]
        holding_returns = fit_prices.shift(-lag - 1).div(fit_prices.shift(-lag)).sub(1.0)
        complete = (~fit_eligible | holding_returns.notna()).all(axis=1)
        self.usable = (complete & fit_eligible.sum(axis=1).ge(3)).to_numpy()
        label_ranks = holding_returns.where(fit_eligible).rank(axis=1)
        self.labels = centered_unit_rows(label_ranks)
        self.ranks = {}
        self.trials = 0

    def rank_scores(self, factor):
        if factor in self.ranks:
            return self.ranks[factor]
        kind, parameter = factor
        if kind == "blend":
            left, right = parameter
            values = (self.rank_scores(left) + self.rank_scores(right)) / 2
        elif kind == "momentum":
            values = self.prices.pct_change(periods=parameter, fill_method=None)
        elif kind == "sma_ratio":
            values = self.prices.div(self.prices.rolling(parameter).mean()).sub(1)
        elif kind == "sharpe":
            rolling = self.returns.rolling(parameter)
            values = rolling.mean().div(rolling.std().replace(0, np.nan)).fillna(0.0)
        else:
            values = self.returns.gt(0).rolling(parameter).mean()
        ranks = values.where(self.eligible).rank(axis=1)
        counts = ranks.count(axis=1)
        denominator = counts.sub(1).clip(lower=1)
        self.ranks[factor] = ranks.sub(1).div(denominator, axis=0)
        return self.ranks[factor]

    def fit(self, close_position):
        settings = self.settings
        # A score from t has a training outcome only after the close at t + lag + 1.
        end = close_position - self.lag
        start = end - settings["training_sessions"]
        if start < 0:
            raise ValueError("Insufficient pre-sample price history for first search.")
        rows = np.flatnonzero(self.usable[start:end]) + start
        if len(rows) * 2 < settings["training_sessions"]:
            raise ValueError("Too few complete cross sections in the training window.")
        labels = self.labels[rows]
        quality = {}
        standardized = {}

        def evaluate(factors):
            for factor in factors:
                ranks = self.rank_scores(factor).iloc[rows]
                unit = centered_unit_rows(ranks)
                standardized[factor] = unit
                quality[factor] = float(np.einsum("ij,ij->i", unit, labels).mean())
                self.trials += 1

        evaluate((kind, window) for kind in INDICATORS for window in settings["seed_windows"])
        for _ in range(settings["generations"]):
            ordered = sorted(quality, key=quality.__getitem__, reverse=True)
            atomic = [factor for factor in ordered if factor[0] != "blend"]
            mutations = []
            for kind, window in atomic[:settings["mutations_per_generation"]]:
                available = [
                    candidate for candidate in settings["mutation_windows"]
                    if (kind, candidate) not in quality and (kind, candidate) not in mutations
                ]
                if available:
                    nearest = min(available, key=lambda value: (abs(value - window), value))
                    mutations.append((kind, nearest))

            parents = ordered[:settings["crossover_parent_pool"]]
            crosses = [
                ("blend", tuple(sorted(pair)))
                for pair in combinations(parents, 2)
            ]
            novel = [candidate for candidate in crosses if candidate not in quality]
            evaluate(mutations)
            evaluate(novel[:settings["crossovers_per_generation"]])

        ordered = sorted(quality, key=quality.__getitem__, reverse=True)
        selected = [factor for factor in ordered if quality[factor] >= settings["min_rank_ic"]]
        selected = selected[:settings["max_factors"]] or ordered[:1]
        ranks = np.stack([standardized[factor].ravel() for factor in selected])
        correlation = (ranks @ ranks.T) / len(rows)
        np.fill_diagonal(correlation, 1.0)
        correlation = denoise(correlation, len(rows))
        rank_ic = np.array([quality[factor] for factor in selected])
        weights = factor_weights(rank_ic, correlation, settings["redundancy_penalty"])
        return selected, weights


def denoise(correlation, observations):
    values, vectors = np.linalg.eigh(correlation)
    noise = values < (1 + np.sqrt(len(values) / observations)) ** 2
    if not noise.any() or noise.all():
        return correlation
    values[noise] = max(values[noise].mean(), 0.0)
    cleaned = (vectors * values) @ vectors.T
    diagonal = np.sqrt(np.diag(cleaned))
    return cleaned / np.outer(diagonal, diagonal)


def factor_weights(rank_ic, correlation, penalty):
    if len(rank_ic) == 1:
        return np.ones(1)
    initial = np.full(len(rank_ic), 1 / len(rank_ic))
    result = minimize(
        lambda w: penalty * (w @ correlation @ w) - rank_ic @ w,
        initial,
        jac=lambda w: 2 * penalty * (correlation @ w) - rank_ic,
        bounds=[(0.0, 1.0)] * len(initial),
        constraints={
            "type": "eq",
            "fun": lambda w: w.sum() - 1,
            "jac": lambda w: np.ones_like(w),
        },
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 300},
    )
    if not result.success:
        raise RuntimeError(f"Factor-weight optimization failed: {result.message}")
    weights = result.x.clip(0, 1)
    return weights / weights.sum()


def portfolio_signals(search, start_position, is_end_position, lag, dates):
    index = search.prices.index
    first_signal = start_position - lag
    if first_signal < 0 or is_end_position < first_signal:
        raise ValueError("The signal and IS cutoff must fit inside the price history.")
    last_signal = len(index) - lag
    if last_signal <= is_end_position:
        raise ValueError("Too few OOS sessions for the configured execution lag.")
    settings = search.settings
    interval = settings["refit_every_months"]
    first_month = dates["start"].to_period("M") + (interval - 1)
    first_month_end = first_month.to_timestamp(how="end").normalize()
    refit_dates = pd.date_range(
        first_month_end, dates["is_end"], freq=pd.offsets.MonthEnd(interval)
    )
    scheduled = index.searchsorted(refit_dates, side="right") - 1
    fit_positions = sorted({first_signal, is_end_position, *scheduled})
    fit_positions = [p for p in fit_positions if first_signal <= p <= is_end_position]
    targets = pd.DataFrame(0.0, index=index, columns=search.prices.columns)
    for fit_position, stop in zip(fit_positions, [*fit_positions[1:], last_signal]):
        factors, weights = search.fit(fit_position)
        composite = sum(
            weight * search.rank_scores(factor).iloc[fit_position:stop]
            for factor, weight in zip(factors, weights)
        )
        in_portfolio = composite.rank(axis=1, ascending=False, method="first").le(
            settings["portfolio_size"]
        )
        count = in_portfolio.sum(axis=1).clip(lower=1)
        targets.iloc[fit_position:stop] = in_portfolio.div(count, axis=0).to_numpy()
    return targets.shift(lag)


def simulate(prices, days, targets, cost_rate):
    relatives = prices.div(prices.shift(1)).loc[days].to_numpy()
    desired_weights = targets.loc[days].to_numpy()
    weights = np.zeros(prices.shape[1])
    returns, turnover, trades = [], [], []
    for day, relatives_today, target in zip(days, relatives, desired_weights):
        held = weights > 0
        if not np.isfinite(relatives_today[held]).all():
            raise RuntimeError(f"Missing close-to-close data for a held asset on {day.date()}")
        asset_values = weights * np.where(held, relatives_today, 1.0)
        marked_nav = (1 - weights.sum()) + asset_values.sum()
        weights = asset_values / marked_nav
        charge, traded, n_orders = 0.0, 0.0, 0
        if np.isfinite(target).all():
            if not prices.loc[day, target > 0].notna().all():
                raise RuntimeError(f"Missing fill close for a selected asset on {day.date()}")
            # Target allocations are fractions of post-cost NAV.
            def trade_sizes(fee):
                return np.abs((1 - fee) * target - weights)

            if cost_rate > 0 and trade_sizes(0).sum() > 0:
                charge = brentq(lambda fee: fee - cost_rate * trade_sizes(fee).sum(), 0, 1)
            fills = trade_sizes(charge)
            traded = float(fills.sum())
            n_orders = int((fills > 1e-10).sum())
            weights = target
        returns.append(marked_nav * (1 - charge) - 1)
        turnover.append(traded)
        trades.append(n_orders)
    return pd.DataFrame({"return": returns, "turnover": turnover, "trades": trades}, index=days)


def metrics(path, risk_free):
    returns = path["return"]
    wealth = (1 + returns).cumprod()
    total = float(wealth.iloc[-1])
    drawdown = wealth.div(wealth.cummax().clip(lower=1)).sub(1)
    excess = returns - risk_free.reindex(path.index)

    def sharpe(series):
        volatility = series.std(ddof=1)
        return float(np.sqrt(ANNUAL_SESSIONS) * series.mean() / volatility) if volatility > 0 else None

    return {
        "cagr": total ** (ANNUAL_SESSIONS / len(path)) - 1,
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(ANNUAL_SESSIONS)) if len(path) > 1 else None,
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(drawdown.min()),
        "total_return": total - 1,
        "turnover": float(path["turnover"].sum() * ANNUAL_SESSIONS / len(path)),
        "n_trades": int(path["trades"].sum()),
    }


def main():
    config = load_config()
    dates = config["dates"]
    universe, benchmark, risk_free = config["universe"], config["benchmark"], config["risk_free"]
    prices, fetched_at = adjusted_prices(
        [*universe, benchmark, risk_free], dates["history_start"], dates["end"]
    )
    prices = prices.where(np.isfinite(prices) & prices.gt(0))
    if prices[benchmark].isna().any():
        raise RuntimeError(f"Missing {benchmark} close on a date with a Yahoo Finance quote.")
    if prices.empty or prices.index[-1] < dates["end"] - pd.Timedelta(days=7):
        raise RuntimeError("Yahoo prices do not reach the configured end of the sample.")
    days = prices.loc[dates["start"]:dates["end"]].index
    if days.empty or days[0] > dates["start"] + pd.Timedelta(days=7):
        raise RuntimeError("Yahoo prices do not cover the configured start of the sample.")
    split = days <= dates["is_end"]
    if not split.any() or split.all():
        raise ValueError("Both IS and OOS need at least one trading session.")
    rf = prices[risk_free].pct_change(fill_method=None)
    if rf.loc[days].isna().any():
        raise RuntimeError(f"Missing daily adjusted {risk_free} returns in the reporting period.")

    execution = config["execution"]
    lag = execution["signal_lag_sessions"]
    rate = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    start_position = prices.index.get_loc(days[0])
    is_end_position = prices.index.get_loc(days[split][-1])
    search = FactorSearch(prices[universe], lag, config["search"], is_end_position)
    targets = portfolio_signals(search, start_position, is_end_position, lag, dates)
    portfolio = simulate(prices[universe], days, targets, rate)

    buy_and_hold = pd.DataFrame(np.nan, index=days, columns=[benchmark])
    buy_and_hold.iloc[0, 0] = 1.0
    benchmark_path = simulate(prices[[benchmark]], days, buy_and_hold, rate)
    results = {
        "strategy": config["strategy"],
        "data_range": {"start": str(days[0].date()), "end": str(days[-1].date())},
        "is": metrics(portfolio.loc[split], rf),
        "oos": metrics(portfolio.loc[~split], rf),
        "full": metrics(portfolio, rf),
        "benchmark": metrics(benchmark_path, rf),
    }

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return close")
    print(f"PARAM_COMBOS_TRIED: {search.trials}")
    for key, label in (("is", "IS"), ("oos", "OOS"), ("full", "FULL"), ("benchmark", benchmark)):
        print(f"{label}: {json.dumps(results[key], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

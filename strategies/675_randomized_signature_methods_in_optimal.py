"""675_randomized_signature_methods_in_optimal
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/675_randomized_signature_methods_in_optimal.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:00 Europe/Oslo, exit 0, output sha256 a584488ae692b16880dbeb243268cad2dd8770169b174ba41a3d797e3b826c63
Data source: Yahoo Finance via yfinance. Daily ETF open, close, dividends, stock splits and capital-gains distributions for SPY, QQQ, IWM, EFA, EEM, IEF, TLT, GLD and VNQ, plus BIL for the excess Sharpe calculation. The downloader substitutes zero if the capital-gains field is absent and fills missing values in that field with zero.
Data fetched: 2026-11-27T19:00:27.497197+00:00
Sample period: Requested data window 2008-01-02 through 2025-12-31 inclusive, with data before 2011-01-03 used for warm-up. The logged backtest runs from 2011-01-03 through 2025-12-31.   data_range (from output): ['2011-01-03', '2025-12-31']
IS/OOS split: IS 2011-01-03 through 2018-12-31. OOS 2019-01-02 through 2025-12-31. The expanding-window scaler and Ridge fit use labels whose ending open is at or before the signal date and the IS cutoff. The fit freezes at the cutoff, while path features and covariance continue to update. Holdings carry across the split in one continuous backtest.
Costs and slippage: Default 3 bps fee plus 2 bps modeled slippage per side. The opening trade solver deducts 5 bps times the sum of absolute buy and sell notionals from available equity. Slippage enters as a proportional cash charge with open-price valuation. The IS-only higher-friction variant uses 9 bps fee plus 6 bps slippage per side.
Signal lag: The default target uses the last trading session of each month through its close and fills at the next session's open, which is the first trading session of the new month. The schedule requires at least one session of lag.
Price adjustment: yfinance split-normalized OHLC, ex-date cash distributions reinvested at the open
Survivorship bias: The backtest uses a fixed ETF list chosen retrospectively from surviving funds. The README explicitly identifies selection and survivorship bias.
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.06615780777633051, ann_vol=0.11457713316812666, sharpe=0.6167292002370182, sharpe_excess=0.5926668064477615, max_drawdown=-0.20034746361374567, total_return=0.6677475940063597, turnover=8.367222230032711, n_trades=680
    oos: cagr=0.07552513484857726, ann_vol=0.15104036720846492, sharpe=0.5583175514116829, sharpe_excess=0.39093209107701826, max_drawdown=-0.32242915654991966, total_return=0.662807734396474, turnover=9.676799424979636, n_trades=621
    full: cagr=0.07051836876057083, ann_vol=0.13282455369685375, sharpe=0.5800079387551403, sharpe_excess=0.48019663665379675, max_drawdown=-0.32242915654991966, total_return=1.7731435983349066, turnover=8.978265672001584, n_trades=1301
    benchmark: cagr=0.08474114662565668, ann_vol=0.11718214299588675, sharpe=0.7529757031455931, sharpe_excess=0.6397375210075215, max_drawdown=-0.2613248439913919, total_return=2.378860402762118, turnover=0.37030555303687496, n_trades=1620
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 675_randomized_signature_methods_in_optimal\nuniverse: [SPY, QQQ, IWM, EFA, EEM, IEF, TLT, GLD, VNQ]\ncash_proxy: BIL\ndates:\n  data_start: '2008-01-02'\n  backtest_start: '2011-01-03'\n  is_end: '2018-12-31'\n  end: '2025-12-31'\nexecution:\n  lag_sessions: 1\n  cost_bps: 3\n  slippage_bps: 2\nmodel:\n  path_sessions: 22\n  horizon_sessions: 21\n  covariance_sessions: 252\n  min_training_rows: 500\n  reservoir_dimension: 50\n  reservoir_matrix_variance: 0.03\n  return_scale: 20.0\n  seed: 675\n  ridge_alpha: 0.001\n  risk_aversion: 6.0\n  max_weight: 0.20\nrobustness:\n  - name: path_20\n    model: {path_sessions: 20}\n  - name: path_24\n    model: {path_sessions: 24}\n  - name: ridge_0.0005\n    model: {ridge_alpha: 0.0005}\n  - name: ridge_0.002\n    model: {ridge_alpha: 0.002}\n  - name: alternate_seed\n    model: {seed: 676}\n  - name: higher_friction\n    execution: {cost_bps: 9, slippage_bps: 6}\n"
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
"""Research backtest for Akyildirim et al. (2023), arXiv:2312.16448.

Paper: https://arxiv.org/abs/2312.16448
Implementation references (independent Python implementation):
- https://github.com/janka-moeller/Sig-SPT (no license declared)
- https://github.com/ThotDjehuty/optimiz-rs (MIT): https://pypi.org/project/optimiz-rs/
- https://github.com/ThotDjehuty/optimiz-rs/blob/main/LICENSE
Related implementations screened:
- https://github.com/AliAtiah/SigRDE
- https://github.com/jambo6/generalised-signature-method
- https://pypi.org/project/iisignature/
Reservoir computing lecture: https://rss.org.uk/getattachment/Training-Events/Events/Events-2020/Sections/Rough-path-theory-in-machine-learning/Josef-Teichmann.pdf.aspx/?lang=en-GB
"""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


@dataclass(frozen=True)
class Model:
    path_sessions: int
    horizon_sessions: int
    covariance_sessions: int
    min_training_rows: int
    reservoir_dimension: int
    reservoir_matrix_variance: float
    return_scale: float
    seed: int
    ridge_alpha: float
    risk_aversion: float
    max_weight: float


@dataclass(frozen=True)
class Execution:
    lag_sessions: int
    cost_bps: float
    slippage_bps: float

    @property
    def friction(self):
        rates = (self.cost_bps, self.slippage_bps)
        if not all(np.isfinite(rate) and 0 <= rate < 5000 for rate in rates):
            raise ValueError("Invalid fee or slippage rate")
        return sum(rates) / 10_000


@dataclass(frozen=True)
class Market:
    dates: pd.DatetimeIndex
    overnight: np.ndarray
    intraday: np.ndarray
    cash_returns: np.ndarray

    def through(self, end):
        stop = self.dates.searchsorted(end, side="right")
        return Market(self.dates[:stop], self.overnight[:stop], self.intraday[:stop],
                      self.cash_returns[:stop])


def validate_prices(raw, tickers, start, end):
    if not isinstance(raw.index, pd.DatetimeIndex) or not raw.index.is_unique:
        raise ValueError("Invalid daily price index")
    if not raw.index.is_monotonic_increasing:
        raise ValueError("Daily price rows are out of order")
    for field in ("Open", "Close", "Dividends", "Capital Gains", "Stock Splits"):
        values = raw[field].reindex(columns=tickers).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing {field} observations on the common ETF calendar")
        if field in ("Open", "Close") and (values <= 0).any():
            raise ValueError(f"Nonpositive {field} price")
    if len(raw) < 2 or raw.index[0] - start > pd.Timedelta(days=7):
        raise ValueError("Missing price history at the start of the requested window")
    if end - raw.index[-1] > pd.Timedelta(days=7):
        raise ValueError("Missing price history at the end of the requested window")


def yahoo_prices(tickers, start, end):
    start_text, end_text = start.date().isoformat(), end.date().isoformat()
    fingerprint = sha256(json.dumps([tickers, start_text, end_text]).encode())
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"yahoo_ohlc_actions_v1_{fingerprint.hexdigest()[:16]}.csv.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            fetched_at = datetime.fromisoformat(source.readline().strip()).isoformat()
            raw = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        validate_prices(raw, tickers, start, end)
        return raw, fetched_at

    # Keep yfinance's transient session files out of the persisted price cache.
    with TemporaryDirectory(dir=CACHE) as work:
        yf.set_tz_cache_location(work)
        for attempt in range(3):
            try:
                bars = yf.download(
                    tickers, start=start_text, end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    auto_adjust=False, actions=True, repair=False, progress=False,
                    threads=4, timeout=12, multi_level_index=True,
                )
                if bars is None or bars.empty:
                    raise ValueError("Yahoo returned empty history")
                raw = pd.concat({
                    field: bars[field].reindex(columns=tickers)
                    for field in ("Open", "Close", "Dividends", "Stock Splits")
                }, axis=1)
                if "Capital Gains" in bars.columns.get_level_values(0):
                    gains = bars["Capital Gains"].reindex(columns=tickers).fillna(0.0)
                else:
                    gains = pd.DataFrame(0.0, index=bars.index, columns=tickers)
                raw = pd.concat([raw, pd.concat({"Capital Gains": gains}, axis=1)], axis=1)
                raw = raw.loc[start:end]
                validate_prices(raw, tickers, start, end)
                break
            except (ValueError, KeyError, OSError) as exc:
                if attempt == 2:
                    raise RuntimeError("Could not retrieve complete public Yahoo ETF bars") from exc
                time.sleep(2 ** attempt)

        fetched_at = datetime.now(timezone.utc).isoformat()
        staged = Path(work) / path.name
        with gzip.open(staged, "wt", encoding="utf-8") as destination:
            destination.write(fetched_at + "\n")
            raw.to_csv(destination)
        staged.replace(path)
    return raw, fetched_at


def market_returns(raw, universe, cash_proxy):
    opening = raw["Open"]
    closing = raw["Close"]
    distributions = raw["Dividends"] + raw["Capital Gains"]
    # Yahoo's OHLC and distributions are in a common split-normalized basis.
    # At the ex-date open, previous positions earn the distribution; new buys do not.
    overnight = ((opening + distributions) / closing.shift(1)).fillna(1.0)
    intraday = closing / opening
    cash_returns = (overnight[cash_proxy] * intraday[cash_proxy] - 1).to_numpy()
    return Market(raw.index, overnight[universe].to_numpy(), intraday[universe].to_numpy(),
                  cash_returns)


def randomized_signatures(log_returns, model):
    """Euler terminal state of the time-augmented cumulative return path."""
    width = model.path_sessions
    n_assets = log_returns.shape[1]
    dimension = model.reservoir_dimension
    random = np.random.default_rng(model.seed)
    operators = random.normal(0, np.sqrt(model.reservoir_matrix_variance),
                              (dimension, (n_assets + 1) * dimension))
    biases = random.standard_normal((n_assets + 1, dimension))
    starts = np.arange(1, len(log_returns) - width + 1)
    state = np.broadcast_to(random.standard_normal(dimension), (len(starts), dimension)).copy()
    for step in range(width):
        directions = (state @ operators).reshape(-1, n_assets + 1, dimension)
        directions += biases
        np.tanh(directions, out=directions)
        increments = log_returns[starts + step] * model.return_scale
        state += directions[:, 0] / width + np.einsum("ijk,ij->ik", directions[:, 1:], increments)
    return state


def monthly_schedule(dates, start, execution):
    if execution.lag_sessions < 1:
        raise ValueError("Execution must lag the close signal by at least one session")
    months = dates.to_period("M")
    first_sessions = np.flatnonzero(months[1:] != months[:-1]) + 1
    return [(int(fill - execution.lag_sessions), int(fill)) for fill in first_sessions
            if dates[fill] >= start and fill >= execution.lag_sessions]


def mean_variance_weights(prediction, covariance, model):
    n_assets = len(prediction)
    initial = np.full(n_assets, 1.0 / n_assets)

    def objective(weights):
        return 0.5 * model.risk_aversion * (weights @ covariance @ weights) - prediction @ weights

    def gradient(weights):
        return model.risk_aversion * (covariance @ weights) - prediction

    result = minimize(
        objective, initial, jac=gradient, method="SLSQP",
        bounds=[(0, model.max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-11, "maxiter": 100},
    )
    weights = np.clip(result.x, 0, model.max_weight)
    if not result.success or not np.isclose(weights.sum(), 1, atol=1e-8, rtol=0):
        raise RuntimeError(f"Portfolio optimization failed: {result.message}")
    return weights / weights.sum()


def target_weights(market, schedule, fit_until, model, execution):
    n_assets = market.overnight.shape[1]
    if not (1 / n_assets <= model.max_weight <= 1 and model.risk_aversion > 0):
        raise ValueError("Infeasible weights or nonpositive risk aversion")
    if min(model.path_sessions, model.horizon_sessions, model.covariance_sessions) < 2:
        raise ValueError("Path, target horizon and covariance windows must be at least two sessions")
    close_returns = market.overnight * market.intraday - 1
    features = randomized_signatures(np.log1p(close_returns), model)
    lag = execution.lag_sessions
    horizon = model.horizon_sessions
    # An open-to-open label excludes distributions at the entry open and
    # includes distributions at the exit open (earned by the prior positions).
    open_path = np.zeros_like(close_returns)
    open_returns = np.log(market.intraday[:-1] * market.overnight[1:])
    open_path[1:] = np.cumsum(open_returns, axis=0)
    feature_dates = np.arange(model.path_sessions, len(market.dates) - lag - horizon)
    label_ends = feature_dates + lag + horizon
    labels = np.expm1(open_path[label_ends] - open_path[feature_dates + lag])
    final_fit_date = market.dates.searchsorted(fit_until, side="right") - 1

    targets = {}
    fitted_rows = None
    for signal, fill in schedule:
        rows = label_ends.searchsorted(min(signal, final_fit_date), side="right")
        if rows < model.min_training_rows or signal < model.covariance_sessions:
            raise ValueError(f"Insufficient completed training or risk history on {market.dates[signal]}")
        if fitted_rows != rows:
            training = features[:rows]
            scaler = StandardScaler().fit(training)
            # Alpha is defined per training observation in the mean squared error objective.
            readout = Ridge(alpha=rows * model.ridge_alpha).fit(scaler.transform(training), labels[:rows])
            fitted_rows = rows
        current = features[signal - model.path_sessions:signal - model.path_sessions + 1]
        predicted = readout.predict(scaler.transform(current))[0] * (SESSIONS_PER_YEAR / horizon)
        covariance_window = close_returns[signal - model.covariance_sessions + 1:signal + 1]
        covariance = LedoitWolf().fit(covariance_window).covariance_ * SESSIONS_PER_YEAR
        targets[fill] = mean_variance_weights(predicted, covariance, model)
    return targets


def trade_at_open(positions, cash, weights, friction):
    open_value = float(positions.sum() + cash)

    def cash_after(invested):
        gross = np.abs(invested * weights - positions).sum()
        return open_value - invested - friction * gross

    invested = brentq(cash_after, 0.0, open_value, xtol=1e-14)
    new_positions = invested * weights
    changes = np.abs(new_positions - positions)
    remaining = cash_after(invested)
    if remaining < -1e-12 * open_value:
        raise RuntimeError("Opening trades exceed available cash")
    return new_positions, max(0.0, remaining), changes.sum() / open_value, int(
        np.count_nonzero(changes > 1e-10 * open_value)
    )


def backtest(market, targets, execution):
    if not targets:
        raise ValueError("No scheduled portfolio fills")
    start = min(targets)
    friction = execution.friction
    positions = np.zeros(market.overnight.shape[1])
    cash = previous_value = 1.0
    daily_results = []
    for day in range(start, len(market.dates)):
        positions = positions * market.overnight[day]
        turnover, n_trades = 0.0, 0
        if day in targets:
            positions, cash, turnover, n_trades = trade_at_open(positions, cash, targets[day], friction)
        positions *= market.intraday[day]
        close_value = positions.sum() + cash
        daily_results.append((close_value / previous_value - 1, turnover, n_trades,
                              market.cash_returns[day]))
        previous_value = close_value
    return pd.DataFrame(daily_results, index=market.dates[start:],
                        columns=["return", "turnover", "n_trades", "cash_return"])


def metrics(daily_results):
    returns = daily_results["return"]
    if len(returns) < 2:
        raise ValueError("Not enough daily observations for evaluation")
    excess = returns - daily_results["cash_return"]
    growth = (1 + returns).cumprod()
    ann_vol = returns.std() * np.sqrt(SESSIONS_PER_YEAR)
    excess_vol = excess.std() * np.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth.iloc[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(ann_vol),
        "sharpe": float(returns.mean() * SESSIONS_PER_YEAR / ann_vol) if ann_vol > 0 else None,
        "sharpe_excess": float(excess.mean() * SESSIONS_PER_YEAR / excess_vol) if excess_vol > 0 else None,
        "max_drawdown": float((growth / growth.cummax().clip(lower=1) - 1).min()),
        "total_return": float(growth.iloc[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    universe, cash_proxy = config["universe"], config["cash_proxy"]
    tickers = [*universe, cash_proxy]
    if len(set(tickers)) != len(tickers) or len(universe) < 2:
        raise ValueError("Use distinct assets and a separate cash proxy")
    if not dates["data_start"] < dates["backtest_start"] < dates["is_end"] < dates["end"]:
        raise ValueError("Warm-up, evaluation and IS cutoff must be strictly chronological")
    model, execution = Model(**config["model"]), Execution(**config["execution"])
    raw, fetched_at = yahoo_prices(tickers, dates["data_start"], dates["end"])
    market = market_returns(raw, universe, cash_proxy)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance split-normalized OHLC, ex-date cash distributions reinvested at the open",
          flush=True)
    schedule = monthly_schedule(market.dates, dates["backtest_start"], execution)
    in_sample_market = market.through(dates["is_end"])

    with threadpool_limits(limits=1):
        official_targets = target_weights(market, schedule, dates["is_end"], model, execution)
        daily_results = backtest(market, official_targets, execution)
        in_sample = daily_results.index <= dates["is_end"]
        official_is = metrics(daily_results.loc[in_sample])
        # The comparison portfolio uses the same timestamps and per-side friction.
        equal_weight = np.full(len(universe), 1 / len(universe))
        reference = backtest(market, {fill: equal_weight for _, fill in schedule}, execution)

        print("ROBUSTNESS (IS only)", flush=True)
        print("default:", json.dumps(official_is, allow_nan=False), flush=True)
        trial_count = 1
        for case in config["robustness"]:
            variant_model = replace(model, **case.get("model", {}))
            variant_execution = replace(execution, **case.get("execution", {}))
            variant_schedule = monthly_schedule(in_sample_market.dates, dates["backtest_start"],
                                                variant_execution)
            targets = target_weights(in_sample_market, variant_schedule, dates["is_end"],
                                     variant_model, variant_execution)
            results = backtest(in_sample_market, targets, variant_execution)
            trial_count += 1
            print(f"{case['name']}:", json.dumps(metrics(results), allow_nan=False), flush=True)

    summary = {
        "strategy": config["strategy"],
        "data_range": [day.date().isoformat() for day in daily_results.index[[0, -1]]],
        "is": official_is,
        "oos": metrics(daily_results.loc[~in_sample]),
        "full": metrics(daily_results),
        "benchmark": metrics(reference),
    }
    if min(summary[period]["n_trades"] for period in ("is", "oos", "benchmark")) <= 0:
        raise RuntimeError("Expected simulated executions in both periods and in the benchmark")
    print(f"PARAM_COMBOS_TRIED: {trial_count}")
    print("IS:", json.dumps(summary["is"], allow_nan=False))
    print("OOS:", json.dumps(summary["oos"], allow_nan=False))
    print("FULL:", json.dumps(summary["full"], allow_nan=False))
    print("BENCHMARK (monthly equal weight):", json.dumps(summary["benchmark"], allow_nan=False))
    print("已回测 | 研究用途 | 待审查")
    print("RESULTS_JSON: " + json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()

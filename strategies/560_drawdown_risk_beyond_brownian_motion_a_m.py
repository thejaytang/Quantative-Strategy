# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "560_drawdown_risk_beyond_brownian_motion_a_m"\nuniverse:\n  risk_asset: SPY\n  defensive_asset: BIL\nbenchmark: SPY\ndata_start: "2007-06-01"\nbacktest_start: "2008-09-02"\nis_end: "2016-12-31"\nbacktest_end: "2025-12-31"\n\nexecution_lag_sessions: 1\ncost_bps: 1.0\nslippage_bps: 2.0\nannualization_sessions: 252\n\nsignal:\n  lookback_sessions: 252\n  minimum_sharpe: 0.0\n  drawdown_sessions: 63\n  drawdown_quantile: 0.90\n  hurst_weight: 0.5\n  hurst_bounds: [0.4, 0.6]\n  hurst_scales: [1, 2, 4, 8, 16, 32]\n  simulation_paths: 2048\n  simulation_seed: 560\n\nrobustness:\n  lookback_sessions: [210, 294]\n  drawdown_quantile: [0.85, 0.95]\n  square_root_hurst_weight: 0.0\n  higher_cost_bps: 5.0\n  higher_slippage_bps: 5.0\n'
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
import signal
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = "yfinance auto_adjust=True total-return adjusted Open and Close"


@dataclass(frozen=True)
class SignalParameters:
    lookback_sessions: int
    minimum_sharpe: float
    drawdown_sessions: int
    drawdown_quantile: float
    hurst_weight: float
    hurst_bounds: list[float]
    hurst_scales: list[int]
    simulation_paths: int
    simulation_seed: int

    def __post_init__(self):
        if len(self.hurst_scales) < 2 or self.hurst_scales != sorted(set(self.hurst_scales)):
            raise ValueError("Hurst scales must be distinct and increasing")
        if self.hurst_scales[0] < 1 or 2 * self.hurst_scales[-1] > self.lookback_sessions:
            raise ValueError("The lookback must cover at least twice the longest Hurst scale")
        if not 1 <= self.drawdown_sessions <= self.lookback_sessions:
            raise ValueError("The drawdown horizon must fit inside the lookback")
        if not 0 < self.drawdown_quantile < 1 or self.simulation_paths < 1:
            raise ValueError("Specify a drawdown quantile in (0, 1) and at least one simulation path")
        if not np.isfinite(self.minimum_sharpe):
            raise ValueError("The minimum Sharpe must be finite")
        lower, upper = self.hurst_bounds
        if not 0 < lower <= 0.5 <= upper < 1 or not 0 <= self.hurst_weight <= 1:
            raise ValueError("Hurst bounds must bracket 0.5 and the weight must be in [0, 1]")


def validate_prices(prices, ticker, start, end):
    start_tolerance = start + pd.Timedelta(days=6)
    end_tolerance = end - pd.Timedelta(days=6)
    if (
        prices.empty
        or not prices.index.is_monotonic_increasing
        or prices.index.has_duplicates
        or not start <= prices.index[0] <= start_tolerance
        or not end_tolerance <= prices.index[-1] <= end
    ):
        raise ValueError(f"Incomplete or unordered {ticker} data for {start.date()} to {end.date()}")
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError(f"Missing or invalid adjusted prices for {ticker}")


def load_prices(ticker, start, end):
    path = CACHE / f"{ticker}_{start.date()}_{end.date()}_adjusted.csv"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            fetched_at = datetime.fromisoformat(source.readline().strip().removeprefix("# fetched_at="))
            prices = pd.read_csv(source, index_col="date", parse_dates=["date"])
        validate_prices(prices, ticker, start, end)
        return prices, fetched_at

    raw = yf.Ticker(ticker).history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        actions=False,
        keepna=True,
        timeout=15,
    )
    prices = raw.loc[:, ["Open", "Close"]].copy()
    prices.index = prices.index.tz_localize(None).normalize().as_unit("us").rename("date")
    validate_prices(prices, ticker, start, end)
    fetched_at = datetime.now(timezone.utc).replace(microsecond=0)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE, delete=False) as output:
        temporary_path = Path(output.name)
        try:
            output.write(f"# fetched_at={fetched_at.isoformat()}\n")
            prices.to_csv(output, date_format="%Y-%m-%d")
            output.flush()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return prices, fetched_at


def read_market_data(universe, start, end):
    CACHE.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=CACHE) as transport_cache:
        yf.set_tz_cache_location(transport_cache)
        yf.config.debug.hide_exceptions = False
        risk_prices, risk_fetched = load_prices(universe["risk_asset"], start, end)
        defensive_prices, defensive_fetched = load_prices(universe["defensive_asset"], start, end)
    if not risk_prices.index.equals(defensive_prices.index):
        raise ValueError("The two ETF series have different session dates")
    return risk_prices, defensive_prices, max(risk_fetched, defensive_fetched)


def hurst_exponent(daily_log_returns, params):
    if params.hurst_weight == 0:
        return 0.5
    cumulative = np.r_[0.0, np.cumsum(daily_log_returns)]
    scales = params.hurst_scales
    deviations = np.array([(cumulative[lag:] - cumulative[:-lag]).std(ddof=1) for lag in scales])
    if (deviations <= 0).any():
        return 0.5
    raw_hurst = np.polyfit(np.log(scales), np.log(deviations), 1)[0]
    shrunk_hurst = 0.5 + params.hurst_weight * (raw_hurst - 0.5)
    return float(np.clip(shrunk_hurst, *params.hurst_bounds))


def drawdown_threshold(daily_log_returns, daily_mean, params, draws):
    # Keep empirical skew and tails. Scale the shocks only, leaving the daily drift intact.
    horizon_scale = params.drawdown_sessions ** (hurst_exponent(daily_log_returns, params) - 0.5)
    paths = (daily_log_returns[draws] - daily_mean) * horizon_scale + daily_mean
    np.cumsum(paths, axis=1, out=paths)
    peaks = np.maximum.accumulate(np.maximum(paths, 0.0), axis=1)
    deepest = np.max(peaks - paths, axis=1)
    return float(np.quantile(deepest, params.drawdown_quantile))


def opening_targets(closes, params, execution_lag, annualization):
    log_prices = np.log(closes.to_numpy())
    log_returns = np.diff(log_prices)
    weeks = closes.index.to_period("W-FRI")
    new_week = np.r_[True, weeks[1:] != weeks[:-1]]
    weekly_targets = pd.Series(np.nan, index=closes.index, dtype=float)
    weekly_eligible = np.flatnonzero(new_week & (np.arange(len(closes)) >= params.lookback_sessions))
    rng = np.random.default_rng(params.simulation_seed)
    uniforms = rng.random((params.simulation_paths, params.drawdown_sessions))
    draws = (uniforms * params.lookback_sessions).astype(np.intp)
    annualizer = np.sqrt(annualization)
    for row in weekly_eligible:
        window = log_returns[row - params.lookback_sessions:row]
        mean = window.mean()
        daily_vol = window.std(ddof=1)
        sharpe = mean * annualizer / daily_vol if daily_vol > 0 else 0.0
        desired = 0.0
        if daily_vol > 0 and sharpe > params.minimum_sharpe:
            trailing = log_prices[row - params.drawdown_sessions:row + 1]
            drawdown = trailing.max() - trailing[-1]
            threshold = drawdown_threshold(window, mean, params, draws)
            desired = float(drawdown <= threshold)
        weekly_targets.iat[row] = desired
    return weekly_targets.shift(execution_lag).ffill()


def simulate(risk_prices, defensive_prices, targets, start, cost_bps, slippage_bps):
    if not all(np.isfinite(bps) and 0 <= bps for bps in (cost_bps, slippage_bps)):
        raise ValueError("Costs and slippage must be finite and nonnegative")
    fee_rate = (cost_bps + slippage_bps) / 10_000
    if fee_rate >= 1:
        raise ValueError("The total fee per ETF leg must be less than 10,000 bps")
    targets = targets.loc[start:]
    if targets.empty or targets.isna().any():
        raise ValueError("The evaluation start needs an opening target from the warm-up period")

    overnight_risk = (risk_prices["Open"] / risk_prices["Close"].shift()).loc[targets.index].to_numpy()
    overnight_defensive = (defensive_prices["Open"] / defensive_prices["Close"].shift()).loc[targets.index].to_numpy()
    intraday_risk = (risk_prices["Close"] / risk_prices["Open"]).loc[targets.index].to_numpy()
    intraday_defensive = (defensive_prices["Close"] / defensive_prices["Open"]).loc[targets.index].to_numpy()
    in_risk = targets.to_numpy(dtype=bool)
    switched = in_risk[1:] != in_risk[:-1]
    buying = np.r_[True, switched]
    selling = np.r_[False, switched]

    # The old position earns the overnight move. Fees reduce the funds available for the new one.
    overnight = np.r_[1.0, np.where(in_risk[:-1], overnight_risk[1:], overnight_defensive[1:])]
    intraday = np.where(in_risk, intraday_risk, intraday_defensive)
    after_fees = (1 - fee_rate * selling) / (1 + fee_rate * buying)
    return pd.DataFrame(
        {
            "return": overnight * after_fees * intraday - 1,
            "risk_free": np.r_[1.0, overnight_defensive[1:]] * intraday_defensive - 1,
            "turnover": selling + after_fees * buying,
            "n_trades": selling.astype(int) + buying,
        },
        index=targets.index,
    )


def metrics(daily_results, annualization):
    returns = daily_results["return"].to_numpy()
    if len(returns) < 2:
        raise ValueError("Each evaluation period needs at least two sessions")
    excess = returns - daily_results["risk_free"].to_numpy()
    volatility = returns.std(ddof=1)
    excess_volatility = excess.std(ddof=1)
    growth = np.r_[1.0, np.cumprod(1 + returns)]
    max_drawdown = (growth / np.maximum.accumulate(growth) - 1).min()
    annualizer = np.sqrt(annualization)
    return {
        "cagr": float(growth[-1] ** (annualization / len(returns)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(returns.mean() / volatility * annualizer) if volatility > 0 else None,
        "sharpe_excess": float(excess.mean() / excess_volatility * annualizer) if excess_volatility > 0 else None,
        "max_drawdown": float(max_drawdown),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily_results["turnover"].sum()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def period_metrics(daily_results, is_end, annualization):
    in_sample = daily_results.index <= is_end
    return {
        "is": metrics(daily_results.loc[in_sample], annualization),
        "oos": metrics(daily_results.loc[~in_sample], annualization),
        "full": metrics(daily_results, annualization),
    }


def print_metrics(label, values):
    print(f"{label}: {json.dumps(values, allow_nan=False)}")


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    universe = config["universe"]
    if universe != {"risk_asset": "SPY", "defensive_asset": "BIL"} or config["benchmark"] != universe["risk_asset"]:
        raise ValueError("This specification uses SPY as the risk asset and benchmark with BIL as the defensive asset")
    data_start, start, is_end, end = (
        pd.Timestamp(config[key]) for key in ("data_start", "backtest_start", "is_end", "backtest_end")
    )
    if not data_start < start < is_end < end:
        raise ValueError("The data start, evaluation start, IS end and evaluation end must be ordered")
    execution_lag = config["execution_lag_sessions"]
    annualization = config["annualization_sessions"]
    if type(execution_lag) is not int or execution_lag < 1 or not np.isfinite(annualization) or annualization <= 0:
        raise ValueError("Open execution needs at least one session of lag and positive annualization")
    cost, slippage = config["cost_bps"], config["slippage_bps"]
    params = SignalParameters(**config["signal"])
    variations = config["robustness"]
    cases = [
        (f"lookback_sessions={value}", replace(params, lookback_sessions=value), cost, slippage)
        for value in variations["lookback_sessions"]
    ] + [
        (f"drawdown_quantile={value}", replace(params, drawdown_quantile=value), cost, slippage)
        for value in variations["drawdown_quantile"]
    ] + [
        ("square_root_scaling", replace(params, hurst_weight=variations["square_root_hurst_weight"]), cost, slippage),
        ("higher_friction", params, variations["higher_cost_bps"], variations["higher_slippage_bps"]),
    ]

    risk_prices, defensive_prices, fetched_at = read_market_data(universe, data_start, end)
    print(f"DATA_FETCHED_AT: {fetched_at.isoformat(timespec='seconds')}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")

    # Alternative parameters see only the prefix through the last IS session.
    risk_is, defensive_is = risk_prices.loc[:is_end], defensive_prices.loc[:is_end]
    print("ROBUSTNESS (IS only)")
    for label, case, case_cost, case_slippage in cases:
        targets_is = opening_targets(risk_is["Close"], case, execution_lag, annualization)
        daily_is = simulate(risk_is, defensive_is, targets_is, start, case_cost, case_slippage)
        print_metrics(label, metrics(daily_is, annualization))

    targets = opening_targets(risk_prices["Close"], params, execution_lag, annualization)
    daily_results = simulate(risk_prices, defensive_prices, targets, start, cost, slippage)
    benchmark_targets = pd.Series(1.0, index=risk_prices.index)
    benchmark_daily = simulate(risk_prices, defensive_prices, benchmark_targets, start, cost, slippage)
    official = period_metrics(daily_results, is_end, annualization)
    benchmark = period_metrics(benchmark_daily, is_end, annualization)
    for period in official:
        print_metrics(period.upper(), official[period])
        print_metrics(f"{period.upper()} {config['benchmark']} buy and hold", benchmark[period])

    print(f"PARAM_COMBOS_TRIED: {1 + len(cases)}")
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": str(daily_results.index[0].date()), "end": str(daily_results.index[-1].date())},
        **official,
        "benchmark": {"ticker": config["benchmark"], **benchmark},
    }
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


def stop_on_timeout(_signum, _frame):
    raise SystemExit("Run exceeded the 14-minute limit")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_on_timeout)
    signal.alarm(14 * 60)
    try:
        main()
    except (ValueError, OSError) as error:
        sys.exit(str(error))

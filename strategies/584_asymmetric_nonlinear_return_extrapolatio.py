"""584_asymmetric_nonlinear_return_extrapolatio
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/584_asymmetric_nonlinear_return_extrapolatio.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:08 Europe/Oslo, exit 0, output sha256 a1798b2e943ddba9e302aa1c0faa9a5185ad5f5208c69a0a4b96eafe6deb081d
Data source: Yahoo Finance daily ETF prices and corporate actions via yfinance. SPY and IEF are the portfolio assets. BIL is the excess-Sharpe reference. The logged data fetch timestamp is 2026-11-27T17:20:12+00:00.
Data fetched: 2026-11-27T18:09:00+00:00
Sample period: The recorded backtest covers trading sessions from 2008-01-02 through 2025-12-31 inclusive. The configured history request starts on 2006-01-01 for warmup, and the configured sample window is 2008-01-01 through 2025-12-31.   data_range (from output): {'start': '2008-01-02', 'end': '2025-12-31', 'is_end': '2017-12-29', 'oos_start': '2018-01-02'}
IS/OOS split: IS is 2008-01-02 through 2017-12-29 and OOS is 2018-01-02 through 2025-12-31, with a configured cutoff of 2018-01-01. The default strategy runs continuously across the boundary, carrying positions and rolling history into OOS. Default parameters are fixed in config and the eight robustness variants run on IS only.
Costs and slippage: Slippage is modelled as a fixed 3 bps surcharge alongside 2 bps of costs. The simulator charges 5 bps of absolute traded notional on each buy or sell in the open rebalance's cash accounting and reserves the charge when solving target notionals. The IS stress variant doubles the combined charge to 10 bps. Market impact and taxes are omitted.
Signal lag: Signals are computed from daily closes. At the default one-session lag, the signal from the final trading session of each month sets the rebalance at the first trading session's open of the following month. The simulator retains existing positions on other sessions.
Price adjustment: yfinance split-normalized OHLC with forward ex-date dividend/capital-gain total returns
Survivorship bias: SPY and IEF form a fixed ETF universe and both had pre-sample prices. The README discloses hindsight selection of liquid ETFs that remained in existence, with residual instrument-selection and survivorship bias.
Parameter combinations tried: 9
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.11466168306638291, ann_vol=0.11836586540518544, sharpe=0.9765080051279798, sharpe_excess=0.9576824583244605, max_drawdown=-0.18890382979305143, total_return=1.95839780811664, turnover=3.3073195719148933, n_trades=65
    oos: cagr=0.09255905218761895, ann_vol=0.12195880164174289, sharpe=0.7871176113353515, sharpe_excess=0.5883981336672843, max_drawdown=-0.22961696890971806, total_return=1.0267331037422407, turnover=4.818306781464229, n_trades=70
    full: cagr=0.10479284868683147, ann_vol=0.11996273899710058, sharpe=0.8910019303691099, sharpe_excess=0.7909902406484756, max_drawdown=-0.22961696890971528, total_return=4.995882771748526, turnover=3.978239262443424, n_trades=135
    benchmark  full: cagr=0.08166394318976833, ann_vol=0.11187994491212537, sharpe=0.7576976493665273, sharpe_excess=0.6466653982854295, max_drawdown=-0.31773455508351034, total_return=3.0993532391909406, turnover=0.2863843496986784, n_trades=432
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 584_asymmetric_nonlinear_return_extrapolatio\nuniverse:\n  equity: SPY\n  treasury: IEF\n  excess_return_reference: BIL\ndates:\n  history_start: 2006-01-01\n  start: 2008-01-01\n  oos_start: 2018-01-01\n  end: 2025-12-31\nexecution:\n  signal_lag_sessions: 1\n  cost_bps: 2.0\n  slippage_bps: 3.0\nsignal:\n  lookback_sessions: 60\n  decay_per_year: 1.0\n  saturation: 3.0\n  asymmetry: 0.5\n  extrapolation_strength: 0.15\nrisk:\n  lookback_sessions: 63\n  risk_aversion: 3.0\n  drift_prior: 0.08\n  safe_rate_prior: 0.02\n  min_spread_vol: 0.10\nbenchmark_equity_weight: 0.60\nsessions_per_year: 252\nrobustness:\n  signal_lookbacks: [45, 75]\n  saturation_values: [2.0, 5.0]\n  asymmetry_values: [0.25, 0.75]\n  linear_same_local_slope: true\n  cost_multiplier: 2.0\n'
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
"""ETF implementation of C185, strategy 584, for historical research.

Asymmetric Nonlinear Return Extrapolation and Optimal Portfolio Choice under
Stochastic Volatility, Dong Yan, Wenrui Ye, Zhiyue Zong, Wenting Chen (2026).
Withdrawal notice: https://arxiv.org/abs/2606.10805
Original text: https://arxiv.org/html/2606.10805v1
Related implementations consulted for ideas; independently written code:
https://raw.githubusercontent.com/ArturSepp/TrendFollowingSystems/main/src/trendfollowing/systems/european.py
https://pypi.org/project/trendfollowing/
https://raw.githubusercontent.com/ArturSepp/TrendFollowingSystems/main/LICENSE (MIT)
https://www.quantconnect.com/terminal/quantpediaBacktestResult/embedded_backtest_c88c9a72c12bd50f952a6c38be0fd68d.html
https://raw.githubusercontent.com/milansevic22/tsmom-volatility-management/main/src/tsmom/strategies.py
https://github.com/milansevic22/tsmom-volatility-management/blob/main/LICENSE (MIT for source)
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
PRICE_ADJUSTMENT = (
    "yfinance split-normalized OHLC with forward ex-date dividend/capital-gain total returns"
)
FIELDS = ("Open", "Close", "Dividends", "Capital Gains", "Stock Splits", "Adj Close")


def load_bars(tickers, dates):
    CACHE.mkdir(exist_ok=True)
    start = dates["history_start"].date()
    end = dates["end"].date()
    path = CACHE / f"yfinance_actions_{'-'.join(tickers)}_{start}_{end}.csv"
    if path.is_file():
        with path.open() as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            bars = pd.read_csv(source, header=[0, 1], index_col=0, parse_dates=True)
        return bars, fetched_at, path

    with TemporaryDirectory(dir=CACHE, prefix="transport_") as transport:
        yf.set_tz_cache_location(transport)
        bars = yf.download(
            list(tickers), start=str(start), end=str(end + pd.Timedelta(days=1)),
            auto_adjust=False, actions=True, repair=False, keepna=True, ignore_tz=True,
            progress=False, timeout=15,
        )
    if bars is None or bars.empty:
        raise RuntimeError("Yahoo Finance returned no ETF price history.")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    bars = bars.reindex(columns=pd.MultiIndex.from_product([FIELDS, tickers])).sort_index()
    return bars, fetched_at, path


def save_bars(path, bars, fetched_at):
    with NamedTemporaryFile(mode="w", dir=CACHE, delete=False) as temp:
        staged = Path(temp.name)
        try:
            temp.write(fetched_at + "\n")
            bars.to_csv(temp, index_label="date")
            temp.close()
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


def market_factors(bars, tickers, dates):
    index = bars.index
    if (
        bars.empty or not index.is_unique or not index.is_monotonic_increasing
        or not isinstance(index, pd.DatetimeIndex)
        or not dates["history_start"] <= index[0] <= dates["history_start"] + pd.Timedelta(days=6)
        or index[-1] > dates["end"]
        or index[-1] < dates["end"] - pd.Timedelta(days=6)
        or index.to_series().diff().max() > pd.Timedelta(days=7)
    ):
        raise ValueError("Yahoo Finance returned incomplete or invalid dates.")

    for ticker in tickers:
        ticker_bars = bars.xs(ticker, axis=1, level=1)
        first = ticker_bars["Close"].first_valid_index()
        if first is None or first >= dates["start"]:
            raise ValueError(f"{ticker} is missing the pre-sample close.")
        if ticker != tickers[-1] and first != index[0]:
            raise ValueError(f"{ticker} is missing warmup bars.")
        active = ticker_bars.loc[first:]
        prices = active.loc[:, ["Open", "Close", "Adj Close"]]
        if not np.isfinite(active).all().all() or (prices <= 0).any().any():
            raise ValueError(f"{ticker} has missing or invalid price/action fields.")
        if (active.loc[:, ["Dividends", "Capital Gains", "Stock Splits"]] < 0).any().any():
            raise ValueError(f"{ticker} has negative corporate-action fields.")

    close = bars["Close"]
    ex_distribution = bars["Dividends"] + bars["Capital Gains"]
    # Yahoo prices and distribution amounts already share the same split basis.
    prior_reference = close.shift(1) - ex_distribution
    if (prior_reference <= 0).any().any():
        raise ValueError("Ex-date distribution exceeds the prior reference price.")
    overnight = bars["Open"] / prior_reference
    intraday = close / bars["Open"]
    close_growth = overnight * intraday

    # Adjustment audit only. Yahoo's adjusted closes never enter sizing or signals.
    yahoo_close_growth = bars["Adj Close"] / bars["Adj Close"].shift(1)
    residual = (close_growth - yahoo_close_growth).abs().max().max()
    if not np.isfinite(residual) or residual > 0.0002:
        raise ValueError(f"Yahoo total-return reconciliation failed: {residual}")
    return overnight, intraday, close_growth - 1, float(residual)


def equity_targets(returns, signal, risk, annualization, linear=False):
    equity, treasury = returns.columns
    log_equity = np.log1p(returns[equity])
    lookback = signal["lookback_sessions"]
    ages = np.arange(lookback - 1, -1, -1)
    decay = np.exp(-signal["decay_per_year"] * ages / annualization)
    kernel = decay * (annualization / decay.sum())
    sentiment = log_equity.rolling(lookback).apply(lambda window: window @ kernel, raw=True)
    scaled = signal["saturation"] * sentiment
    if linear:
        extrapolation = (1 + signal["asymmetry"]) * scaled
    else:
        bounded = np.tanh(scaled)
        extrapolation = bounded + signal["asymmetry"] * bounded * (1 - bounded)

    spread = returns[equity] - returns[treasury]
    rolling = spread.rolling(risk["lookback_sessions"])
    spread_variance = (rolling.var() * annualization).clip(lower=risk["min_spread_vol"] ** 2)
    bond_covariance = rolling.cov(returns[treasury]) * annualization
    gamma = risk["risk_aversion"]
    relative_drift = (
        risk["drift_prior"] - risk["safe_rate_prior"]
        + signal["extrapolation_strength"] * extrapolation
    )
    # Optimize a long-only equity weight with the remainder in the bond ETF.
    return ((relative_drift - gamma * bond_covariance) / (gamma * spread_variance)).clip(0, 1)


def trade_at_open(position_values, cash, desired_weights, fee_rate):
    open_value = float(cash + position_values.sum())
    low, high = 0.0, open_value
    # Solve for target notional after both buy and sell charges, from below.
    for _ in range(48):
        target_value = (low + high) / 2
        traded = np.abs(target_value * desired_weights - position_values).sum()
        if target_value + fee_rate * traded <= open_value:
            low = target_value
        else:
            high = target_value

    updated = low * desired_weights
    traded = np.abs(updated - position_values)
    total_traded = float(traded.sum())
    cash = open_value - (float(updated.sum()) + fee_rate * total_traded)
    if cash < -open_value * 1e-12:
        raise RuntimeError("Rebalance exceeded available funds.")
    trades = int((traded > open_value * 1e-12).sum())
    return updated, max(cash, 0.0), total_traded / open_value, trades


def simulate(overnight, intraday, weights_at_open, fee_rate):
    position_values = np.zeros(2)
    cash = prior_value = 1.0
    rows = []
    for gap, session, equity_weight in zip(
        overnight.to_numpy(), intraday.to_numpy(), weights_at_open.to_numpy(), strict=True
    ):
        position_values *= gap
        turnover, trades = 0.0, 0
        if not np.isnan(equity_weight):
            weights = np.array([equity_weight, 1 - equity_weight])
            position_values, cash, turnover, trades = trade_at_open(
                position_values, cash, weights, fee_rate
            )
        position_values *= session
        close_value = cash + position_values.sum()
        rows.append((close_value / prior_value - 1, turnover, trades))
        prior_value = close_value
    return pd.DataFrame(rows, index=overnight.index, columns=["return", "turnover", "n_trades"])


def metrics(daily_results, bil_returns, annualization):
    returns = daily_results["return"].to_numpy()
    values = np.r_[1.0, np.cumprod(1 + returns)]
    excess = returns - bil_returns.loc[daily_results.index].to_numpy()
    daily_vol = returns.std(ddof=1)
    excess_vol = excess.std(ddof=1)
    scale = np.sqrt(annualization)
    return {
        "cagr": float(values[-1] ** (annualization / len(returns)) - 1),
        "ann_vol": float(daily_vol * scale),
        "sharpe": float(scale * returns.mean() / daily_vol),
        "sharpe_excess": float(scale * excess.mean() / excess_vol),
        "max_drawdown": float((values / np.maximum.accumulate(values) - 1).min()),
        "total_return": float(values[-1] - 1),
        "turnover": float(daily_results["turnover"].mean() * annualization),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def print_metrics(label, summary):
    print(f"{label}: {json.dumps(summary, allow_nan=False)}")


def run_robustness(is_returns, run, config, fee_rate):
    signal = config["signal"]
    settings = config["robustness"]
    trials = []
    for key, grid in (
        ("lookback_sessions", "signal_lookbacks"),
        ("saturation", "saturation_values"),
        ("asymmetry", "asymmetry_values"),
    ):
        for value in settings[grid]:
            trials.append((f"{key}={value}", {**signal, key: value}, fee_rate, False))
    if settings["linear_same_local_slope"]:
        trials.append(("linear_same_local_slope", signal, fee_rate, True))
    multiplier = settings["cost_multiplier"]
    trials.append((f"cost_and_slippage_x{multiplier:g}", signal, fee_rate * multiplier, False))

    print("ROBUSTNESS (IS only)")
    for label, model, fee, linear in trials:
        print_metrics(f"  {label}", run(is_returns, model, fee, linear))
    return len(trials)


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = {key: pd.Timestamp(value) for key, value in config["dates"].items()}
    if not dates["history_start"] < dates["start"] < dates["oos_start"] <= dates["end"]:
        raise ValueError("Invalid history, sample or OOS boundaries.")
    execution = config["execution"]
    lag = execution["signal_lag_sessions"]
    if not isinstance(lag, int) or lag < 1:
        raise ValueError("Signal lag must be at least one trading session.")
    if min(execution["cost_bps"], execution["slippage_bps"]) < 0:
        raise ValueError("Costs and slippage must be nonnegative.")
    fee_rate = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    if not 0 <= fee_rate * max(1.0, config["robustness"]["cost_multiplier"]) < 1:
        raise ValueError("Invalid combined fee rate.")

    universe = config["universe"]
    assets = [universe["equity"], universe["treasury"]]
    bil = universe["excess_return_reference"]
    tickers = [*assets, bil]
    bars, fetched_at, cache_path = load_bars(tickers, dates)
    overnight, intraday, returns, adjustment_error = market_factors(bars, tickers, dates)
    if not cache_path.is_file():
        save_bars(cache_path, bars, fetched_at)

    annualization = config["sessions_per_year"]
    all_dates = returns.index
    month_opens = all_dates[~all_dates.to_period("M").duplicated()]
    sample = all_dates[all_dates >= dates["start"]]
    is_dates = sample[sample < dates["oos_start"]]
    oos_dates = sample[sample >= dates["oos_start"]]
    if min(len(is_dates), len(oos_dates)) < 2:
        raise ValueError("IS and OOS must contain at least two sessions each.")

    def targets(history, signal, linear=False):
        equity_weight = equity_targets(history, signal, config["risk"], annualization, linear)
        delayed = equity_weight.shift(lag)
        return delayed.where(~delayed.index.to_period("M").duplicated())

    def evaluate(index, weights, fee):
        schedule = month_opens.intersection(index)
        if not np.isfinite(weights.loc[schedule]).all():
            raise ValueError("Incomplete history for a scheduled signal.")
        return simulate(
            overnight.loc[index, assets], intraday.loc[index, assets], weights.loc[index], fee
        )

    def run_is(history, signal, fee, linear):
        weights = targets(history, signal, linear)
        return metrics(evaluate(is_dates, weights, fee), returns.loc[is_dates, bil], annualization)

    signal = config["signal"]
    is_history = returns.loc[:is_dates[-1], assets]
    weights = targets(returns[assets], signal)
    pd.testing.assert_series_equal(weights.loc[is_history.index], targets(is_history, signal))
    official = evaluate(sample, weights, fee_rate)
    benchmark_equity = config["benchmark_equity_weight"]
    if not 0 <= benchmark_equity <= 1:
        raise ValueError("Benchmark equity weight must be in [0, 1].")
    benchmark_weights = pd.Series(benchmark_equity, index=month_opens).reindex(all_dates)
    benchmark = evaluate(sample, benchmark_weights, fee_rate)
    if int(official["n_trades"].sum()) <= 0:
        raise RuntimeError("Default run contains zero fills.")

    print(f"DATA_FETCHED_AT: {fetched_at}")
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"LOOKAHEAD_CHECK signal_lag: {lag} session(s), close signal to first monthly open")
    print("LOOKAHEAD_CHECK data_availability: ETF closes and effective ex-date actions; BIL metrics only")
    print(f"LOOKAHEAD_CHECK splits_and_adjustment: max Yahoo close-return difference {adjustment_error:.8g}")
    print("LOOKAHEAD_CHECK is_oos: fixed defaults; IS-only replay reproduces all IS targets")
    tried = 1 + run_robustness(is_history, run_is, config, fee_rate)
    print(f"PARAM_COMBOS_TRIED: {tried}")

    periods = {"is": is_dates, "oos": oos_dates, "full": sample}
    benchmark_summary = {"name": f"{benchmark_equity:.0%} {assets[0]} / {1 - benchmark_equity:.0%} {assets[1]} monthly"}
    result = {
        "strategy": config["strategy"],
        "data_range": {
            "start": str(sample[0].date()), "end": str(sample[-1].date()),
            "is_end": str(is_dates[-1].date()), "oos_start": str(oos_dates[0].date()),
        },
    }
    for period, index in periods.items():
        reference = returns.loc[index, bil]
        result[period] = metrics(official.loc[index], reference, annualization)
        benchmark_summary[period] = metrics(benchmark.loc[index], reference, annualization)
        print_metrics(period.upper(), result[period])
        print_metrics(f"BENCHMARK {period.upper()}", benchmark_summary[period])
    result["benchmark"] = benchmark_summary
    print_metrics("RESULTS_JSON", result)


if __name__ == "__main__":
    main()

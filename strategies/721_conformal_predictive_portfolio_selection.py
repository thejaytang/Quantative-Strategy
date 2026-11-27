"""721_conformal_predictive_portfolio_selection
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/721_conformal_predictive_portfolio_selection.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:03 Europe/Oslo, exit 0, output sha256 bd840592ec39e1d1e90bd4f330cc2a1d8571f43feb1310cc4c52a5123806f546
Data source: Yahoo Finance daily ETF opens, closes and corporate actions via yfinance. The basket universe is SPY, EFA, EEM, IEF, TLT, LQD, GLD and VNQ. SPY is also the buy-and-hold benchmark. BIL is a risk-free proxy used in reporting. The run recorded a snapshot fetch time of 2026-11-27T18:23:20+00:00.
Data fetched: 2026-11-27T19:03:22+00:00
Sample period: The requested price history starts on 2008-01-02 for warm-up. The configured backtest starts on 2014-01-01. The recorded simulation runs from the first fill on 2014-01-03 through 2025-12-31 inclusive.   data_range (from output): {'start': '2014-01-03', 'end': '2025-12-31'}
IS/OOS split: IS covers 2014-01-03 through 2019-12-31. OOS covers observed sessions on and after the 2020-01-01 cutoff through 2025-12-31. The default run carries positions and cash across the cutoff and keeps rolling the fit and calibration under the same settings. The five robustness variants receive only pre-cutoff prices.
Costs and slippage: The default models slippage as a fixed 5 bps cash charge plus a 5 bps fee, for 10 bps of the sum of absolute buy and sell notionals at each rebalance. The sizing solver budgets for these charges and fills at the opening prices. The double_friction IS variant uses 10 bps for each charge, totaling 20 bps. The SPY benchmark pays the default friction on entry.
Signal lag: The signal uses the close of the first observed trading session of each month. The default lag is one session, so the target is filled at the next observed session's open. A target is scheduled only if its fill row is within the sample.
Price adjustment: yfinance split-normalized OHLC, forward ex-date open reinvestment
Survivorship bias: The fixed ETF pool consists of surviving funds selected for the availability of complete history. The README identifies survivorship and universe-selection bias.
Parameter combinations tried: 6
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.0845896387847973, ann_vol=0.0830316413117198, sharpe=1.0196195281839184, sharpe_excess=0.931916410623184, max_drawdown=-0.1352469093096088, total_return=0.6261959708946252, turnover=11.649621383463625, n_trades=293
    oos: cagr=0.09641488956642585, ann_vol=0.14299398745077552, sharpe=0.7149163690225259, sharpe_excess=0.5319899550902181, max_drawdown=-0.24397005744446176, total_return=0.7346626427938119, turnover=14.341874008314509, n_trades=331
    full: cagr=0.09048427538311232, ann_vol=0.11689436957965266, sharpe=0.7993695628297584, sharpe_excess=0.6565128431143387, max_drawdown=-0.24397005744446276, total_return=1.820901400572685, turnover=12.995301515473944, n_trades=624
    benchmark: cagr=0.13521474551613943, ann_vol=0.17336969559936038, sharpe=0.818591530380703, sharpe_excess=0.7222612906232151, max_drawdown=-0.33722737548690873, total_return=3.564637967800002, turnover=0.08344323889567418, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 721_conformal_predictive_portfolio_selection\nuniverse: [SPY, EFA, EEM, IEF, TLT, LQD, GLD, VNQ]\nbenchmark: SPY\nrisk_free: BIL\ndownload_start: '2008-01-02'\nstart: '2014-01-01'\noos_start: '2020-01-01'\nend: '2025-12-31'\nsignal_lag_sessions: 1\ncost_bps: 5\nslippage_bps: 5\nmodel:\n  ar_order: 3\n  train_months: 36\n  calibration_months: 24\n  alpha: 0.20\n  shortlist: 7\n  basket_sizes: [2, 3, 4]\nrobustness:\n  - name: alpha_15\n    model: {alpha: 0.15}\n  - name: alpha_25\n    model: {alpha: 0.25}\n  - name: shortlist_5\n    model: {shortlist: 5}\n  - name: shortlist_9\n    model: {shortlist: 9}\n  - name: double_friction\n    cost_bps: 10\n    slippage_bps: 10\n"
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
"""Portfolio-level conformal prediction and monthly HR-LR selection.

Paper: Masahiro Kato (2024), https://arxiv.org/abs/2410.16333
Research references (ideas only, independently implemented):
  MAPIE (BSD-3-Clause), https://github.com/scikit-learn-contrib/MAPIE
  https://pypi.org/project/MAPIE/
  https://github.com/scikit-learn-contrib/MAPIE/blob/master/examples/regression/1-quickstart/plot_prefit.py
  https://github.com/scikit-learn-contrib/MAPIE/blob/master/mapie/regression/time_series_regression.py
  https://mapie.readthedocs.io/en/stable/generated/regression/1-quickstart/plot_ts-tutorial/
  Conformal Finance Pipeline (no license identified),
  https://github.com/lfmfreitas/Conformal-Finance-Pipeline
  https://raw.githubusercontent.com/lfmfreitas/Conformal-Finance-Pipeline/main/src/conformal.py
"""

import hashlib
import json
import math
import signal
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


HERE = Path(__file__).resolve().parent
FIELDS = ["Open", "Close", "Dividends", "Capital Gains", "Stock Splits"]
SESSIONS_PER_YEAR = 252


def check_config(config):
    dates = [pd.Timestamp(config[key]) for key in ("download_start", "start", "oos_start", "end")]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Expected download_start < start < oos_start < end.")
    universe = config["universe"]
    if len(universe) != len(set(universe)) or config["risk_free"] in universe:
        raise ValueError("Use distinct portfolio symbols and a separate risk-free proxy.")
    if config["signal_lag_sessions"] < 1:
        raise ValueError("Execution must follow the signal by at least one session.")


def check_prices(frame, symbols, start, end):
    fields = pd.MultiIndex.from_product([FIELDS, symbols])
    frame = frame.reindex(columns=fields)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Yahoo price dates must be unique and ordered.")
    if frame.empty or not (pd.Timestamp(start) <= frame.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7)):
        raise ValueError("Yahoo did not return the start of the requested history.")
    if not (end - pd.Timedelta(days=7) <= frame.index[-1] < end):
        raise ValueError("Yahoo did not return the end of the requested history.")
    values = frame.to_numpy()
    if not np.isfinite(values).all() or (frame[["Open", "Close"]] <= 0).any().any():
        raise ValueError("Yahoo history has missing, nonfinite or nonpositive quotes.")
    if (frame[["Dividends", "Capital Gains", "Stock Splits"]] < 0).any().any():
        raise ValueError("Yahoo history contains a negative corporate action.")
    return frame


def load_prices(config):
    symbols = sorted(set(config["universe"] + [config["benchmark"], config["risk_free"]]))
    start = config["download_start"]
    end = pd.Timestamp(config["end"]) + pd.Timedelta(days=1)
    cache = HERE / "data_cache"
    cache.mkdir(exist_ok=True)
    specification = json.dumps(["yfinance-actions-v1", symbols, str(start), str(end)])
    key = hashlib.sha256(specification.encode()).hexdigest()[:16]
    path = cache / f"yfinance_{key}.npz"

    if path.exists():
        with np.load(path, allow_pickle=False) as snapshot:
            fetched_at = snapshot["fetched_at"].item()
            frame = pd.DataFrame(snapshot["values"], index=pd.DatetimeIndex(snapshot["dates"]),
                                 columns=pd.MultiIndex.from_product([FIELDS, symbols]))
        return check_prices(frame, symbols, start, end), fetched_at

    pieces = []
    with TemporaryDirectory(dir=cache) as session_cache:
        yf.set_tz_cache_location(session_cache)
        for symbol in symbols:
            quotes = yf.Ticker(symbol).history(start=start, end=end.date().isoformat(), interval="1d",
                                               auto_adjust=False, actions=True, timeout=15)
            if quotes.empty:
                raise RuntimeError(f"Yahoo returned no data for {symbol}.")
            if "Capital Gains" not in quotes:
                quotes["Capital Gains"] = 0.0
            quotes = quotes[FIELDS].copy()
            quotes.index = quotes.index.tz_localize(None).normalize()
            pieces.append(quotes)
    frame = pd.concat(pieces, axis=1, keys=symbols).swaplevel(axis=1).sort_index()
    frame = check_prices(frame, symbols, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    temporary = NamedTemporaryFile(dir=cache, delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            np.savez_compressed(temporary, fetched_at=fetched_at, values=frame.to_numpy(),
                                dates=frame.index.to_numpy(dtype="datetime64[ns]"))
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return frame, fetched_at


def total_return_prices(frame):
    # Yahoo Open/Close and cash amounts use the same split-normalized share basis.
    # Reinvest each reported distribution at the ex-date open in synthetic units.
    opens, closes = frame["Open"], frame["Close"]
    distributions = frame["Dividends"] + frame["Capital Gains"]
    reinvestment = (1 + distributions / opens).cumprod()
    scale = reinvestment / reinvestment.iloc[0] / opens.iloc[0]
    return opens * scale, closes * scale


def select_baskets(opens, closes, config):
    model = config["model"]
    universe = config["universe"]
    order = model["ar_order"]
    train = model["train_months"]
    calibration = model["calibration_months"]
    shortlist = model["shortlist"]
    alpha = model["alpha"]
    sizes = model["basket_sizes"]
    if len(set(sizes)) != len(sizes) or not all(1 <= size <= len(universe) for size in sizes):
        raise ValueError("Basket sizes must be unique and within the universe.")
    if not (0 < alpha < 1 and order >= 1 and train > order + 1 and calibration >= 1):
        raise ValueError("Invalid autoregression or calibration settings.")
    rank = math.ceil((calibration + 1) * (1 - alpha))
    if not 1 <= rank <= calibration:
        raise ValueError("Calibration window is too short for the requested alpha.")

    groups = [group for size in sizes for group in combinations(range(len(universe)), size)]
    if not 1 <= shortlist <= len(groups):
        raise ValueError("Shortlist size must be within the basket count.")
    weights = np.zeros((len(groups), len(universe)))
    for row, group in enumerate(groups):
        weights[row, list(group)] = 1.0 / len(group)

    month = opens.index.to_period("M")
    signal_rows = np.flatnonzero(~month.duplicated())
    fill_rows = signal_rows + config["signal_lag_sessions"]
    scheduled = fill_rows < len(opens)
    signal_rows, fill_rows = signal_rows[scheduled], fill_rows[scheduled]
    signal_closes = closes[universe].to_numpy()[signal_rows]
    fill_opens = opens[universe].to_numpy()[fill_rows]
    observed = (signal_closes[1:] / signal_closes[:-1] - 1) @ weights.T
    realized = (fill_opens[1:] / fill_opens[:-1] - 1) @ weights.T

    # Design row j ends at signal close j. Outcome j ends at fill open j+1.
    features = np.full((len(signal_rows), len(weights), order + 1), np.nan)
    features[order:, :, 0] = 1.0
    for lag in range(order):
        features[order:, :, lag + 1] = observed[order - lag - 1:len(observed) - lag]

    targets = {}
    for j, (signal_row, fill_row) in enumerate(zip(signal_rows, fill_rows)):
        fill_date = opens.index[fill_row]
        if fill_date < pd.Timestamp(config["start"]):
            continue
        # Only outcomes whose exit open was observable at this signal close.
        completed = np.searchsorted(fill_rows, signal_row, side="right") - 1
        split = completed - calibration
        first = split - train
        if first < order:
            raise ValueError(f"Insufficient matured training history for {fill_date.date()}.")

        lower = np.empty(len(weights))
        upper = np.empty_like(lower)
        for basket in range(len(weights)):
            x = features[:, basket]
            y = realized[:, basket]
            coefficients = np.linalg.lstsq(x[first:split], y[first:split], rcond=None)[0]
            scores = np.abs(y[split:completed] - x[split:completed] @ coefficients)
            radius = np.partition(scores, rank - 1)[rank - 1]
            center = x[j] @ coefficients
            lower[basket], upper[basket] = center - radius, center + radius

        best_lower = np.argsort(-lower, kind="stable")[:shortlist]
        chosen = best_lower[np.argmax(upper[best_lower])]
        targets[fill_date] = weights[chosen]
    if not targets:
        raise ValueError("No rebalance dates in the requested backtest period.")
    return targets


def rebalance(positions, cash, opening_prices, weights, friction):
    open_values = positions * opening_prices
    value = open_values.sum() + cash
    low, high = 0.0, value
    # Solve for invested value after costs. Fill sizes only use opening prices.
    for _ in range(48):
        middle = (low + high) / 2
        traded = np.abs(weights * middle - open_values).sum()
        if middle + friction * traded <= value:
            low = middle
        else:
            high = middle
    next_values = weights * max(0.0, low - 1e-14 * value)
    changes = next_values - open_values
    traded = np.abs(changes).sum()
    if traded <= 1e-12 * value:
        return positions, cash, 0.0, 0
    next_cash = cash - changes.sum() - friction * traded
    if next_cash < 0:
        raise ArithmeticError("Rebalance would spend more cash than the proceeds and available cash.")
    trades = np.count_nonzero(np.abs(changes) > 1e-12 * value)
    return next_values / opening_prices, next_cash, traded / value, int(trades)


def simulate(opens, closes, targets, config):
    cost_bps, slippage_bps = config["cost_bps"], config["slippage_bps"]
    friction = (cost_bps + slippage_bps) / 10_000
    if cost_bps < 0 or slippage_bps < 0 or not 0 <= friction < 1:
        raise ValueError("Invalid cost or slippage.")
    dates = opens.loc[min(targets):].index
    positions = np.zeros(len(opens.columns))
    cash, previous_value = 1.0, 1.0
    daily_results = []
    for date, opening_prices, closing_prices in zip(dates, opens.loc[dates].to_numpy(),
                                                    closes.loc[dates].to_numpy()):
        turnover, trades = 0.0, 0
        if date in targets:
            positions, cash, turnover, trades = rebalance(positions, cash, opening_prices,
                                                         targets[date], friction)
        value = positions @ closing_prices + cash
        daily_results.append((value / previous_value - 1, turnover, trades))
        previous_value = value
    return pd.DataFrame(daily_results, index=dates, columns=["return", "turnover", "n_trades"])


def metrics(daily, risk_free):
    if len(daily) < 2:
        raise ValueError("A reporting period must contain at least two sessions.")
    daily_returns = daily["return"]
    growth = np.r_[1.0, (1 + daily_returns).cumprod().to_numpy()]
    volatility = daily_returns.std(ddof=1)
    excess = daily_returns - risk_free.loc[daily.index]
    excess_vol = excess.std(ddof=1)
    annualizer = math.sqrt(SESSIONS_PER_YEAR)
    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(daily)) - 1),
        "ann_vol": float(volatility * annualizer),
        "sharpe": float(daily_returns.mean() * annualizer / volatility) if volatility else None,
        "sharpe_excess": float(excess.mean() * annualizer / excess_vol) if excess_vol else None,
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1)),
        "total_return": float(growth[-1] - 1),
        "turnover": float(daily["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily["n_trades"].sum()),
    }


def emit(label, value):
    print(f"{label}: {json.dumps(value, allow_nan=False)}", flush=True)


def main():
    with (HERE / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    check_config(config)
    frame, fetched_at = load_prices(config)
    opens, closes = total_return_prices(frame)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance split-normalized OHLC, forward ex-date open reinvestment")

    universe = config["universe"]
    split = pd.Timestamp(config["oos_start"])
    proxy = config["risk_free"]
    proxy_returns = closes[proxy].pct_change(fill_method=None)
    is_opens, is_closes = opens.loc[opens.index < split], closes.loc[closes.index < split]
    print("ROBUSTNESS: IS only, fixed alternatives")
    trials = 0
    for alternative in config["robustness"]:
        variant = {**config, **alternative, "model": {**config["model"], **alternative.get("model", {})}}
        targets = select_baskets(is_opens, is_closes, variant)
        daily = simulate(is_opens[universe], is_closes[universe], targets, variant)
        first = daily.index[0]
        is_proxy = proxy_returns.loc[daily.index].copy()
        is_proxy.loc[first] = closes.at[first, proxy] / opens.at[first, proxy] - 1
        emit(alternative["name"], metrics(daily, is_proxy))
        trials += 1

    targets = select_baskets(opens, closes, config)
    daily = simulate(opens[universe], closes[universe], targets, config)
    trials += 1
    first = daily.index[0]
    proxy_returns.loc[first] = closes.at[first, proxy] / opens.at[first, proxy] - 1
    benchmark_symbol = config["benchmark"]
    benchmark_daily = simulate(opens[[benchmark_symbol]], closes[[benchmark_symbol]],
                               {first: np.ones(1)}, config)
    periods = {"is": daily.index < split, "oos": daily.index >= split, "full": slice(None)}
    result = {
        "strategy": config["strategy"],
        "data_range": {"start": first.date().isoformat(), "end": daily.index[-1].date().isoformat()},
    }
    for name, period in periods.items():
        result[name] = metrics(daily.loc[period], proxy_returns)
        emit(name.upper(), result[name])
    if min(result[name]["n_trades"] for name in periods) == 0:
        raise RuntimeError("Backtest requires simulated trades in both periods.")
    result["benchmark"] = {"name": f"{benchmark_symbol} buy and hold"}
    result["benchmark"].update(metrics(benchmark_daily, proxy_returns))
    emit("BENCHMARK", result["benchmark"])
    print(f"PARAM_COMBOS_TRIED: {trials}")
    emit("RESULTS_JSON", result)


if __name__ == "__main__":
    signal.alarm(14 * 60)
    main()

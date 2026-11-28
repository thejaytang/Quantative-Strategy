"""443_revisiting_equity_bond_diversification_u
WARNING: known defects (see DEEP_REVIEW_FINDINGS.md); numbers invalid.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/443_revisiting_equity_bond_diversification_u.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 6df04a4d89c55b7a1b162dbdcff0dafc1e2bd682 on 2026-11-27 08:09 Europe/Oslo, exit 0, output sha256 f775f181d4606ba5f796be0875f0eb5a1c14a956e4850e66a66fadee60a4e935
Data source: Yahoo Finance daily ETF closes via yfinance for AOA, AOR, AOM and AOK, with BIL used only for excess-return metrics. The price cache stores the fetch timestamp. The run reports 2026-11-27T06:34:06+00:00.
Data fetched: 2026-11-27T07:09:45+00:00
Sample period: Requested prices span 2017-12-01 through 2024-12-31 inclusive. Data before 2020-01-02 warm up the 504-session rolling window. The logged backtest spans 2020-01-02 through 2024-12-31.   data_range (from output): 2020-01-02 to 2024-12-31
IS/OOS split: IS runs from 2020-01-02 through 2022-12-30. OOS runs from 2023-01-03 through 2024-12-31. The split is dates before versus on or after 2023-01-01. Objective selection uses IS annualized excess Sharpe versus BIL after charges, with ties within 1e-12 resolved in config order. The selected objective is fixed for OOS, daily weights are still updated, and positions carry over at the boundary.
Costs and slippage: The model charges 5 bps in transaction costs plus 2 bps in fixed slippage (7 bps total) on each buy and sell notional, including initial entry. The simulator marks existing positions to the execution close, solves for post-charge NAV using the sum of absolute dollar changes normalized by pre-trade NAV, and reduces that session's portfolio return by the resulting charge.
Signal lag: The target uses 504 daily simple ETF returns ending at close t after that close is available. It executes at the next trading session's close (t+1). Existing positions earn the t-to-t+1 return, and new weights first earn returns after the t+1 close.
Price adjustment: yfinance auto_adjust total-return
Survivorship bias: The universe is a fixed list of four ETFs drawn from a paper published after the sample. Delisted or liquidated funds were excluded, leaving potential survivorship and universe-selection bias.
Parameter combinations tried: 2
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.0070281709846601315, ann_vol=0.09121537006834894, sharpe=-0.03159323163411792, sharpe_excess=-0.09352058234660152, max_drawdown=-0.18933509588907038, total_return=-0.020936674549634438, turnover=0.16655008160954, n_trades=1
    oos: cagr=0.08729765347244056, ann_vol=0.06206731869926302, sharpe=1.3796519569849102, sharpe_excess=0.5795471716657562, max_drawdown=-0.0637334471376958, total_return=0.18143116182928054, turnover=0.0, n_trades=0
    full: cagr=0.029589114790648408, ann_vol=0.08087229026954204, sharpe=0.4011140749144018, sharpe_excess=0.11417746901534627, max_drawdown=-0.18933509588907038, total_return=0.15669592209126226, turnover=0.10008892026773629, n_trades=1
    benchmark: cagr=0.05172953198510588, ann_vol=0.11215919535177646, sharpe=0.5060258283065188, sharpe_excess=0.29911424920445456, max_drawdown=-0.20955974970686053, total_return=0.28631253666927003, turnover=0.3027443106994026, n_trades=5032
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 443_revisiting_equity_bond_diversification_u\nuniverse: [AOA, AOR, AOM, AOK]\nrisk_free: BIL\n\ndata_start: "2017-12-01"\nis_start: "2020-01-02"\noos_start: "2023-01-01"\nend: "2024-12-31"\n\nlookback_sessions: 504\nobjectives: [minimum_variance, historical_cvar]\ncvar_confidence: 0.95\nsignal_lag_sessions: 1\ncost_bps: 5.0\nslippage_bps: 2.0\n'
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
"""Rolling long-only allocation across balanced ETFs."""

from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, linprog, minimize
from scipy.sparse import csr_matrix, eye, hstack
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


def load_config():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    dates = ["data_start", "is_start", "oos_start", "end"]
    for key in dates:
        cfg[key] = pd.Timestamp(cfg[key])
    if not cfg["data_start"] < cfg["is_start"] < cfg["oos_start"] < cfg["end"]:
        raise ValueError("Expected data_start < is_start < oos_start < end.")
    tickers = [*cfg["universe"], cfg["risk_free"]]
    if len(cfg["universe"]) < 2 or len(set(tickers)) != len(tickers):
        raise ValueError("Use at least two distinct ETFs and a separate BIL series.")
    if cfg["risk_free"] != "BIL":
        raise ValueError("This strategy uses BIL for excess returns.")
    for key, minimum in (("lookback_sessions", 2), ("signal_lag_sessions", 1)):
        if type(cfg[key]) is not int or cfg[key] < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}.")
    if not 0 < cfg["cvar_confidence"] < 1:
        raise ValueError("cvar_confidence must be between zero and one.")
    cost, slippage = cfg["cost_bps"], cfg["slippage_bps"]
    if not (0 <= cost and 0 <= slippage and cost + slippage < 10_000):
        raise ValueError("Cost and slippage must be nonnegative and total less than 10,000 bps.")
    return cfg


def check_prices(prices, tickers, start, end):
    if prices.empty or not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("Yahoo Finance prices are empty or dated inconsistently.")
    if prices.index[0] < start or prices.index[-1] > end:
        raise ValueError("Yahoo Finance price dates fall outside the requested range.")
    if prices.index[0] - start > pd.Timedelta(days=7) or end - prices.index[-1] > pd.Timedelta(days=7):
        raise ValueError("Yahoo Finance prices do not cover the requested endpoints.")
    incomplete = prices.isna() | ~np.isfinite(prices) | (prices <= 0)
    if incomplete.to_numpy().any():
        missing = incomplete.stack()
        examples = list(missing[missing].index[:5])
        raise ValueError(f"Missing or invalid prices for {tickers} (date and ticker): {examples}")


def load_prices(cfg):
    tickers = [*cfg["universe"], cfg["risk_free"]]
    start, end = cfg["data_start"], cfg["end"]
    cache_key = sha256(json.dumps([tickers, str(start.date()), str(end.date())]).encode())
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"adjusted_close_{cache_key.hexdigest()[:20]}.csv"
    if path.exists():
        with path.open(encoding="utf-8") as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            prices = pd.read_csv(source, index_col="Date", parse_dates=["Date"], dtype=str)
            prices = prices.astype(float).reindex(columns=tickers)
        check_prices(prices, tickers, start, end)
        return prices, fetched_at

    with TemporaryDirectory(prefix="yfinance_", dir=CACHE) as transport_cache:
        yf.set_tz_cache_location(transport_cache)
        history = yf.download(
            tickers, start=start.date().isoformat(),
            end=(end + pd.Timedelta(days=1)).date().isoformat(),
            auto_adjust=True, interval="1d", progress=False, threads=False, timeout=12,
        )
    if history is None or history.empty or "Close" not in history:
        raise RuntimeError("Yahoo Finance did not return adjusted closing prices.")
    prices = history["Close"].reindex(columns=tickers).dropna(how="all")
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize().rename("Date")
    check_prices(prices, tickers, start, end)
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(mode="w", dir=CACHE, encoding="utf-8", delete=False) as destination:
        destination.write(f"{fetched_at}\n")
        prices.to_csv(destination)
    Path(destination.name).replace(path)
    return prices, fetched_at


def minimum_variance(window):
    sample_cov = np.cov(window * 100, rowvar=False)
    count = window.shape[1]
    result = minimize(
        lambda w: w @ sample_cov @ w,
        np.full(count, 1 / count),
        jac=lambda w: 2 * sample_cov @ w,
        bounds=[(0, 1)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones_like(w)},
        method="SLSQP", options={"ftol": 1e-12},
    )
    return result.x if result.success else None


def historical_cvar(window, confidence):
    observations, assets = window.shape
    # Variables are ETF weights, a free loss threshold and nonnegative exceedances.
    loss_constraints = hstack(
        [csr_matrix(-100 * window), -np.ones((observations, 1)), -eye(observations)],
        format="csr",
    )
    result = linprog(
        c=np.r_[np.zeros(assets), 1, np.full(observations, 1 / ((1 - confidence) * observations))],
        A_ub=loss_constraints, b_ub=np.zeros(observations),
        A_eq=[np.r_[np.ones(assets), np.zeros(1 + observations)]], b_eq=[1],
        bounds=[(0, 1)] * assets + [(None, None)] + [(0, None)] * observations,
        method="highs-ds",
    )
    return result.x[:assets] if result.success else None


def long_only_weights(candidate, asset_count):
    if (candidate is None or not np.isfinite(candidate).all()
            or np.any(candidate < -1e-8) or abs(candidate.sum() - 1) > 1e-8):
        return np.zeros(asset_count)
    weights = np.where(candidate > 1e-10, candidate, 0)
    return weights / weights.sum()


def simulate(returns, cfg, allocate):
    market = returns.to_numpy()
    lookback = cfg["lookback_sessions"]
    lag = cfg["signal_lag_sessions"]
    first = returns.index.searchsorted(cfg["is_start"])
    if first < lookback + lag - 1:
        raise ValueError("Extend data_start to cover the full lookback and execution lag.")
    fee = (cfg["cost_bps"] + cfg["slippage_bps"]) / 10_000
    count = market.shape[1]
    held = np.zeros(count)
    rows = []

    for i in range(first, len(market)):
        # Mark existing positions to this close before executing an older signal.
        marked = held * (1 + market[i])
        gross = marked.sum() + (1 - held.sum())
        pre_trade = marked / gross
        stop = i - lag + 1
        target = long_only_weights(allocate(market[stop - lookback:stop]), count)

        # Solve for the fraction of pre-trade NAV left after paying both charges.
        def residual(net_fraction):
            return net_fraction + fee * np.abs(net_fraction * target - pre_trade).sum() - 1

        net_fraction = brentq(residual, 0, 1, xtol=1e-14)
        traded = np.abs(net_fraction * target - pre_trade)
        rows.append((gross * net_fraction - 1, traded.sum() / 2,
                     int(np.count_nonzero(traded > 1e-8)), not target.any()))
        held = target
    return pd.DataFrame(rows, index=returns.index[first:],
                        columns=["return", "turnover", "n_trades", "cash"])


def sharpe(returns):
    volatility = returns.std(ddof=1)
    if np.isfinite(volatility) and volatility > 0:
        return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility)
    return None


def metrics(portfolio, bill_returns):
    returns = portfolio["return"]
    if len(returns) < 2:
        raise ValueError("Each reporting period must contain at least two sessions.")
    wealth = np.r_[1.0, np.cumprod(1 + returns.to_numpy())]
    return {
        "cagr": float(wealth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - bill_returns.loc[returns.index]),
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(SESSIONS_PER_YEAR * portfolio["turnover"].mean()),
        "n_trades": int(portfolio["n_trades"].sum()),
    }


def main():
    cfg = load_config()
    methods = {
        "minimum_variance": minimum_variance,
        "historical_cvar": partial(historical_cvar, confidence=cfg["cvar_confidence"]),
    }
    objectives = cfg["objectives"]
    if not objectives or len(set(objectives)) != len(objectives) or set(objectives) - methods.keys():
        raise ValueError(f"Choose distinct objectives from {list(methods)}.")
    prices, fetched_at = load_prices(cfg)
    print(f"DATA_FETCHED_AT: {fetched_at}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return")

    returns = prices.pct_change(fill_method=None).iloc[1:]
    assets = returns[cfg["universe"]]
    bills = returns[cfg["risk_free"]]
    training = assets.loc[assets.index < cfg["oos_start"]]
    scores = {}
    for objective in objectives:
        trial = simulate(training, cfg, methods[objective])
        scores[objective] = sharpe(trial["return"] - bills.loc[trial.index])
    print(f"PARAM_COMBOS_TRIED: {len(scores)}")
    print("IS_OBJECTIVE_SCORES: " + json.dumps(scores, allow_nan=False))
    ranked = {objective: score for objective, score in scores.items() if score is not None}
    if not ranked:
        raise ValueError("No objective produced a defined in-sample excess Sharpe.")
    best_score = max(ranked.values())
    selected = next(objective for objective, score in ranked.items() if best_score - score <= 1e-12)
    print(f"IS_SELECTED_OBJECTIVE: {selected}")

    portfolio = simulate(assets, cfg, methods[selected])
    equal = np.full(len(cfg["universe"]), 1 / len(cfg["universe"]))
    benchmark = simulate(assets, cfg, lambda window: equal)
    print(f"CASH_FALLBACK_DAYS: {int(portfolio['cash'].sum())}")
    date_range = f"{portfolio.index[0].date()} to {portfolio.index[-1].date()}"
    result = {"strategy": cfg["strategy"], "data_range": date_range}
    for period, mask in {
        "is": portfolio.index < cfg["oos_start"],
        "oos": portfolio.index >= cfg["oos_start"],
        "full": np.ones(len(portfolio), dtype=bool),
    }.items():
        result[period] = metrics(portfolio.loc[mask], bills)
        benchmark_metrics = metrics(benchmark.loc[mask], bills)
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
        print(f"BENCHMARK {period.upper()} (equal weight): "
              + json.dumps(benchmark_metrics, allow_nan=False))
        if period == "full":
            result["benchmark"] = benchmark_metrics
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
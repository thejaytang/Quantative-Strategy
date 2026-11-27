"""445_dynamic_scenario_based_portfolio_optimiz
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/445_dynamic_scenario_based_portfolio_optimiz.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:04 Europe/Oslo, exit 0, output sha256 fd3d5af30d6343ec64f9a60220ac5c4aad43de94d12247015f4a068a86db7369
Data source: Yahoo Finance daily adjusted ETF closes via yfinance. Portfolio universe: VT, VTI, ACWI, ACWX, SUSA, ROUS, IYG, BND, FPE, BYLD, NTSX, GAL, MDIV, AOM, AOR, INKM, HNDL, ALTY, IYLD, CVY, PDBC, QAI, HDG, MNA, RAAX, RLY, SRET, YYY, GAA, GMOM. BIL is the risk-free return proxy.
Data fetched: 2026-11-27T11:04:39.642705+00:00
Sample period: Actual price sample 2018-12-19 through 2026-09-30. Backtest evaluation 2022-12-21 through 2026-09-30. Both ranges are inclusive. Prices before evaluation supply the rolling estimation history.   data_range (from output): {'start': '2022-12-21', 'end': '2026-09-30', 'price_start': '2018-12-19', 'price_end': '2026-09-30'}
IS/OOS split: IS runs from 2022-12-21 through 2024-12-31 inclusive. OOS starts on the first session after 2024-12-31 and ends on 2026-09-30 inclusive. The script runs one continuous backtest and slices the resulting daily series, so OOS carries forward positions and continues updating the rolling estimation window.
Costs and slippage: 10 bps commission plus 5 bps modeled slippage per dollar of ETF buys and sells, including initial entry. The 15 bps rate applies to the sum of absolute traded notionals from drifted holdings to targets sized on post-cost NAV. The cost reduces NAV at the simulated fill. The buy-and-hold benchmark incurs the same charges at entry only. Slippage is a flat cost, with no separate spread or market-impact model.
Signal lag: The signal uses ETF returns through the close of session t. The target fills at the next trading session's close, t+1. Pre-fill holdings earn the return through the fill close, and the new holdings first earn the return from that close to the following close.
Price adjustment: yfinance auto_adjust=True adjusted-close returns and simulated fills
Survivorship bias: The 30 ETFs are a fixed list from the paper's Appendix A, chosen with historical data available at publication. The README identifies survivorship and hindsight selection bias. The code uses the configured list throughout the historical sample.
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.09487934263690145, ann_vol=0.07462637032329562, sharpe=1.2521022641311332, sharpe_excess=0.5858927754582265, max_drawdown=-0.0692731249107188, total_return=0.20091868057446738, turnover=1.2601871474307538, n_trades=15013
    oos: cagr=0.11487821556650957, ann_vol=0.07878069970827621, sharpe=1.4199167588853312, sharpe_excess=0.9277637576861222, max_drawdown=-0.08665827721654829, total_return=0.20753119765311578, turnover=0.7856483909566926, n_trades=12155
    full: cagr=0.10407274541360567, ann_vol=0.07653478513731093, sharpe=1.3320725410706618, sharpe_excess=0.7486365813904898, max_drawdown=-0.08665827721654817, total_return=0.4501467726380841, turnover=1.0409763265225458, n_trades=27168
    benchmark  full: cagr=0.1144051190077886, ann_vol=0.08677035922887896, sharpe=1.2919561294177955, sharpe_excess=0.7773305235285656, max_drawdown=-0.10365138744765057, total_return=0.5017522723491639, turnover=0.25474064863086265, n_trades=30
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "445_dynamic_scenario_based_portfolio_optimiz"\nuniverse:\n  [VT, VTI, ACWI, ACWX, SUSA, ROUS, IYG, BND, FPE, BYLD, NTSX, GAL, MDIV,\n   AOM, AOR, INKM, HNDL, ALTY, IYLD, CVY, PDBC, QAI, HDG, MNA, RAAX, RLY,\n   SRET, YYY, GAA, GMOM]\nrisk_free_ticker: BIL\ndates:\n  price_start: 2018-12-19\n  evaluation_start: 2022-12-21\n  is_end: 2024-12-31\n  end: 2026-09-30\nmodel:\n  lookback_sessions: 1008\n  cvar_confidence: 0.95\n  target_turnover_cap: 0.00013227513227513228  # 1 / (30 * 252)\n  execution_lag_sessions: 1\n  cash_buffer: 0.05\ncosts:\n  commission_bps: 10.0\n  slippage_bps: 5.0\nannualization_sessions: 252\n'
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
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
TRADE_TOLERANCE = 1e-10


class NYSECalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay, USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        Holiday("George H. W. Bush day of mourning", year=2018, month=12, day=5),
        Holiday("Jimmy Carter day of mourning", year=2025, month=1, day=9),
    ]


def nyse_sessions(start, end):
    if start.year < 2018 or end.year > 2026:
        raise ValueError("NYSE session validation covers 2018 through 2026")
    return pd.bdate_range(start, end).difference(NYSECalendar().holidays(start, end))


def extract_closes(frame, tickers):
    if frame is None or frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([]), columns=tickers, dtype=float)
    closes = frame["Close"]
    if isinstance(closes, pd.Series) and len(tickers) == 1:
        closes = closes.to_frame(tickers[0])
    if not isinstance(closes, pd.DataFrame):
        raise ValueError("Yahoo Finance close columns could not be matched to the requested symbols")
    return closes.reindex(columns=tickers)


def download_closes(tickers, start, end):
    settings = dict(start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    auto_adjust=True, progress=False, timeout=12,
                    ignore_tz=True, multi_level_index=True)
    closes = extract_closes(yf.download(tickers, threads=8, **settings), tickers)
    if closes.empty or not closes.notna().any().any():
        raise RuntimeError("Yahoo Finance returned no adjusted close data")
    for ticker in tickers:
        if closes[ticker].notna().any():
            continue
        retry = extract_closes(yf.download(ticker, threads=False, **settings), [ticker])
        closes = closes.combine_first(retry)
    return closes


def check_closes(closes, tickers, start, end):
    closes = closes.reindex(columns=tickers).sort_index().rename_axis(index="Date", columns=None)
    closes.index = pd.DatetimeIndex(closes.index, dtype="datetime64[ns]")
    if closes.index.has_duplicates:
        raise ValueError("Yahoo Finance returned duplicate session dates")
    closes = closes.loc[start:end]
    sessions = nyse_sessions(start, end)
    missing_sessions = sessions.difference(closes.index)
    extra_dates = closes.index.difference(sessions)
    if sessions.empty or len(missing_sessions) or len(extra_dates):
        raise ValueError(f"Yahoo Finance dates do not match NYSE sessions "
                         f"({len(sessions)} expected): {len(missing_sessions)} missing "
                         f"{missing_sessions[:10].strftime('%Y-%m-%d').tolist()}, "
                         f"{len(extra_dates)} unexpected {extra_dates[:10].strftime('%Y-%m-%d').tolist()}")
    invalid = ~np.isfinite(closes) | (closes <= 0)
    if invalid.any().any():
        missing = invalid.sum()
        raise ValueError(f"Incomplete adjusted close panel (rows by ticker): "
                         f"{missing[missing > 0].to_dict()}")
    return closes


def load_closes(tickers, start, end):
    CACHE.mkdir(exist_ok=True)
    key = hashlib.sha256(f"auto_adjust_close|{tickers}|{start}|{end}".encode()).hexdigest()[:16]
    path = CACHE / f"yahoo_{key}.csv"
    if not path.exists():
        with TemporaryDirectory(dir=CACHE, prefix=".yfinance_") as transient_cache:
            yf.set_tz_cache_location(transient_cache)
            closes = download_closes(tickers, start, end)
        closes = check_closes(closes, tickers, start, end)
        fetched_at = datetime.now(timezone.utc).isoformat()
        with NamedTemporaryFile(mode="w", dir=CACHE, encoding="utf-8", delete=False) as handle:
            temporary = Path(handle.name)
            try:
                handle.write(fetched_at + "\n")
                closes.to_csv(handle, index_label="Date")
                handle.close()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)

    snapshot = path.read_bytes()
    with BytesIO(snapshot) as handle:
        fetched_at = datetime.fromisoformat(handle.readline().decode().strip()).isoformat()
        closes = pd.read_csv(handle, index_col="Date", parse_dates=["Date"])
    return check_closes(closes, tickers, start, end), fetched_at, hashlib.sha256(snapshot).hexdigest()


class HistoricalCVaR:
    def __init__(self, n_assets, lookback, confidence, target_cap):
        self.n_assets = n_assets
        self.target_cap = target_cap
        # LP variables are weights, absolute target changes, VaR, and scenario excess losses.
        self.objective = np.r_[np.zeros(2 * n_assets), 1,
                               np.full(lookback, 1 / ((1 - confidence) * lookback))]
        identity = eye(n_assets, format="csr")
        zeros = csr_matrix((n_assets, 1 + lookback))
        self.change_constraints = vstack([
            hstack([identity, -identity, zeros]),
            hstack([-identity, -identity, zeros]),
            csr_matrix(np.r_[np.zeros(n_assets), np.ones(n_assets), np.zeros(1 + lookback)])
        ], format="csr")
        self.scenario_slack = hstack([csr_matrix((lookback, n_assets)),
                                      csr_matrix(-np.ones((lookback, 1))),
                                      -eye(lookback, format="csr")], format="csr")
        self.budget = csr_matrix(np.r_[np.ones(n_assets), np.zeros(n_assets + 1 + lookback)])
        self.bounds = ([(0, 1)] * n_assets + [(0, None)] * n_assets
                       + [(None, None)] + [(0, None)] * lookback)
        self.scenario_rhs = np.zeros(lookback)

    def target(self, scenarios, previous):
        # Percent units keep portfolio loss and VaR away from solver precision limits.
        loss_constraints = hstack([csr_matrix(-100 * scenarios), self.scenario_slack])
        result = linprog(
            self.objective,
            A_ub=vstack([self.change_constraints, loss_constraints], format="csr"),
            b_ub=np.r_[previous, -previous, 2 * self.target_cap, self.scenario_rhs],
            A_eq=self.budget, b_eq=[1.0], bounds=self.bounds, method="highs-ds",
        )
        if not result.success:
            raise RuntimeError(f"CVaR optimization failed: {result.message}")
        weights = np.maximum(result.x[:self.n_assets], 0)
        weights /= weights.sum()
        change = 0.5 * np.abs(weights - previous).sum()
        if change > self.target_cap + 1e-7:
            raise RuntimeError("CVaR target exceeds the adjustment cap")
        if change > self.target_cap:
            weights = previous + (weights - previous) * (self.target_cap / change)
        return weights


def size_rebalance(positions, cash, sizing_prices, target, cash_buffer, fee_rate):
    value = cash + positions @ sizing_prices
    # At sizing prices, total buys plus sells cannot exceed twice the portfolio value.
    target_value = value * (1 - cash_buffer - 2 * fee_rate)
    orders = target_value * target / sizing_prices - positions
    return np.where(np.abs(orders) * sizing_prices > TRADE_TOLERANCE * value, orders, 0.0)


def fill_at_close(positions, cash, orders, fill_prices, commission_rate, slippage_rate, label):
    fill_values = orders * fill_prices
    buys = float(fill_values[fill_values > 0].sum())
    sells = float(-fill_values[fill_values < 0].sum())
    traded = buys + sells
    commission = commission_rate * traded
    slippage = slippage_rate * traded
    remaining_cash = cash + sells - buys - commission - slippage
    if remaining_cash < 0:
        raise RuntimeError(f"{label}: fixed share quantities would leave negative cash ({remaining_cash:.8g})")

    n_trades = int(np.count_nonzero(orders))
    details = {
        "turnover": traded / (cash + positions @ fill_prices),
        "n_trades": n_trades,
        "buy_notional": buys,
        "sell_notional": sells,
        "commission": commission,
        "slippage": slippage,
        "min_trade_notional": float(np.abs(fill_values[orders != 0]).min()) if n_trades else None,
        "cash": remaining_cash,
    }
    return positions + orders, remaining_cash, details


def backtest(asset_closes, evaluation_start, model, commission_rate, slippage_rate):
    lookback = model["lookback_sessions"]
    lag = model["execution_lag_sessions"]
    cash_buffer = model["cash_buffer"]
    fee_rate = commission_rate + slippage_rate
    dates = asset_closes.index
    first_fill = dates.searchsorted(evaluation_start)
    if first_fill < lookback + lag or first_fill >= len(dates):
        raise ValueError("The evaluation start needs a full estimation window before the first fill")

    n_assets = asset_closes.shape[1]
    optimizer = HistoricalCVaR(n_assets, lookback, model["cvar_confidence"],
                               model["target_turnover_cap"])
    equal = np.full(n_assets, 1 / n_assets)
    previous_target = equal.copy()
    positions = {name: np.zeros(n_assets) for name in ("strategy", "benchmark")}
    cash = dict.fromkeys(positions, 1.0)
    last_values = cash.copy()
    targets = {}
    pending_orders = {}
    no_orders = np.zeros(n_assets)
    prices = asset_closes.to_numpy()
    asset_returns = asset_closes.pct_change(fill_method=None).to_numpy()
    daily_results = {name: [] for name in positions}

    for i in range(first_fill - lag, len(dates)):
        close = prices[i]
        if i >= first_fill:
            for name in positions:
                orders = pending_orders.pop(name, no_orders)
                positions[name], cash[name], details = fill_at_close(
                    positions[name], cash[name], orders, close, commission_rate, slippage_rate,
                    f"{name} {dates[i].date()}")
                value = cash[name] + positions[name] @ close
                details["return"] = value / last_values[name] - 1
                details["cash_weight"] = cash[name] / value
                if name == "strategy" or i == first_fill:
                    details["signal_date"] = dates[i - lag] if name == "strategy" else pd.NaT
                    details["sizing_date"] = dates[i - 1]
                daily_results[name].append(details)
                last_values[name] = value

        if i + lag < len(dates):
            scenarios = asset_returns[i - lookback + 1:i + 1]
            previous_target = optimizer.target(scenarios, previous_target)
            targets[i + lag] = previous_target

        next_session = i + 1
        if first_fill <= next_session < len(dates):
            # Share quantities are fixed at this close for the next session's close.
            pending_orders["strategy"] = size_rebalance(
                positions["strategy"], cash["strategy"], close, targets.pop(next_session), cash_buffer, fee_rate)
            if next_session == first_fill:
                pending_orders["benchmark"] = size_rebalance(
                    positions["benchmark"], cash["benchmark"], close, equal, cash_buffer, fee_rate)

    fill_dates = dates[first_fill:].rename("fill_date")
    return pd.concat({name: pd.DataFrame(rows, index=fill_dates)
                      for name, rows in daily_results.items()}, axis=1)


def metrics(frame, risk_free, sessions_per_year):
    daily = frame["return"].to_numpy()
    excess = daily - risk_free.reindex(frame.index).to_numpy()
    years = len(daily) / sessions_per_year
    wealth = np.r_[1.0, np.cumprod(1 + daily)]
    vol = float(daily.std(ddof=1))
    excess_vol = float(excess.std(ddof=1))
    annualizer = math.sqrt(sessions_per_year)
    return {
        "cagr": float(wealth[-1] ** (1 / years) - 1),
        "ann_vol": vol * annualizer,
        "sharpe": float(annualizer * daily.mean() / vol) if vol else None,
        "sharpe_excess": float(annualizer * excess.mean() / excess_vol) if excess_vol else None,
        "max_drawdown": float(np.min(wealth / np.maximum.accumulate(wealth) - 1)),
        "total_return": float(wealth[-1] - 1),
        "turnover": float(frame["turnover"].sum() / years),
        "n_trades": int(frame["n_trades"].sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Backtest the historical-scenario ETF portfolio")
    parser.add_argument("--show-fills", action="store_true", help="Print daily execution totals as CSV")
    args = parser.parse_args()
    with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    universe = config["universe"]
    risk_free_ticker = config["risk_free_ticker"]
    model = config["model"]
    start, evaluation_start, is_end, end = (
        pd.Timestamp(config["dates"][key]) for key in ("price_start", "evaluation_start", "is_end", "end"))
    commission, slippage = (config["costs"][key] for key in ("commission_bps", "slippage_bps"))
    sessions_per_year = config["annualization_sessions"]
    if not start < evaluation_start <= is_end < end:
        raise ValueError("Expected price_start < evaluation_start <= is_end < end")
    if len(set(universe)) != len(universe) or not universe or risk_free_ticker != "BIL":
        raise ValueError("Expected a unique ETF universe and BIL for the risk-free proxy")
    if (model["lookback_sessions"] < 2 or model["execution_lag_sessions"] < 1
            or not 0 < model["cvar_confidence"] < 1 or not 0 <= model["target_turnover_cap"] <= 1):
        raise ValueError("Invalid rolling window, execution lag, confidence level, or adjustment cap")
    if not (commission >= 0 and slippage >= 0 and commission + slippage < 5000
            and sessions_per_year > 0):
        raise ValueError("Invalid costs or annualization convention")
    commission_rate, slippage_rate = commission / 10_000, slippage / 10_000
    if not 0 < model["cash_buffer"] < 1 - 2 * (commission_rate + slippage_rate):
        raise ValueError("Cash and fee reserves must leave positive capital for investment")

    tickers = list(dict.fromkeys([*universe, risk_free_ticker]))
    closes, fetched_at, data_sha256 = load_closes(tickers, start, end)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"DATA_SHA256: {data_sha256}", flush=True)
    print("PRICE_ADJUSTMENT: yfinance auto_adjust=True adjusted-close returns and simulated fills", flush=True)
    print("PARAM_COMBOS_TRIED: 1", flush=True)
    daily = backtest(closes[universe], evaluation_start, model, commission_rate, slippage_rate)
    periods = {"is": daily.loc[daily.index <= is_end], "oos": daily.loc[daily.index > is_end], "full": daily}
    if min(len(period) for period in periods.values()) < 2:
        raise ValueError("IS and OOS each need at least two daily observations")

    risk_free = closes[risk_free_ticker].pct_change(fill_method=None)
    strategy_stats = {name: metrics(period["strategy"], risk_free, sessions_per_year)
                      for name, period in periods.items()}
    benchmark_stats = {name: metrics(period["benchmark"], risk_free, sessions_per_year)
                       for name, period in periods.items()}
    for name in periods:
        print(f"{name.upper()}: {json.dumps(strategy_stats[name], allow_nan=False)}")
        print(f"BENCHMARK {name.upper()} (equal-weight buy and hold): "
              f"{json.dumps(benchmark_stats[name], allow_nan=False)}")

    min_cash_weights = {name: float(daily[name]["cash_weight"].min())
                        for name in ("strategy", "benchmark")}
    print(f"MIN_POST_FILL_CASH_WEIGHT: {json.dumps(min_cash_weights, allow_nan=False)}")
    if args.show_fills:
        for name in ("strategy", "benchmark"):
            print(f"{name.upper()}_DAILY_FILLS_CSV:")
            print(daily[name].to_csv(), end="")

    results = {
        "strategy": config["strategy"],
        "data_range": {"start": daily.index[0].date().isoformat(),
                       "end": daily.index[-1].date().isoformat(),
                       "price_start": closes.index[0].date().isoformat(),
                       "price_end": closes.index[-1].date().isoformat()},
        **strategy_stats,
        "benchmark": {"name": "equal-weight buy and hold (same ETF universe)", **benchmark_stats},
    }
    print("RESULTS_JSON: " + json.dumps(results, allow_nan=False))


if __name__ == "__main__":
    main()

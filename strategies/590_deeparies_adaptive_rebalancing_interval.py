"""590_deeparies_adaptive_rebalancing_interval
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/590_deeparies_adaptive_rebalancing_interval.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 25a9179813de3addbbaee65a8c08579eb6e27d08 on 2026-11-27 19:09 Europe/Oslo, exit 0, output sha256 2d08302e1d795e5d6b311e19b0f93a83fbba12d4f8e3bf86ff145f28f33075a8
Data source: Free Yahoo Finance ETF daily open/close prices and corporate actions via yfinance. Strategy universe: SPY, QQQ, IWM, EFA, EEM, IEF, GLD. SPY is also the buy-and-hold benchmark. BIL daily total returns are used as the risk-free proxy for excess-return metrics only. The run log records the cache download timestamp as 2026-11-27T17:22:13+00:00.
Data fetched: 2026-11-27T18:09:19+00:00
Sample period: Price history requested from 2008-01-01 through 2025-12-31 inclusive, with 2008 used for warm-up. The configured report start is 2009-01-01 and the realized report spans trading sessions from 2009-01-02 through 2025-12-31 inclusive.   data_range (from output): ['2009-01-02', '2025-12-31']
IS/OOS split: Yes. The retrospective chronological split is IS 2009-01-02 through 2020-12-31 and OOS 2021-01-04 through 2025-12-31. Robustness variants run on the IS slice only. The full default simulation carries positions, cash and pending signal state across the split. Its IS rows must match a separate IS-only replay exactly.
Costs and slippage: 2 bps transaction cost plus 3 bps modeled slippage, for 5 bps per side on each dollar of gross buy and sell notional. The open rebalance deducts both as a proportional cash charge and sizes purchases within the post-friction budget. The benchmark pays the same combined rate on its first opening purchase.
Signal lag: Targets and the rebalance decision use information through close t. A scheduled rebalance fills at the next trading session's open with the configured one-session lag. The initial entry uses the prior session's target.
Price adjustment: yfinance auto_adjust=False, split-normalized OHLC with ex-date distributions reinvested at close (total-return proxy)
Survivorship bias: The asset universe is a fixed list of ETFs selected with knowledge of which funds survived, as disclosed in the README. This introduces asset-selection and survivorship bias. Historical fund returns reflect constituents through the ETF itself.
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.08087902135123692, ann_vol=0.0654040548088224, sharpe=1.2220250007634437, sharpe_excess=1.1579096604518724, max_drawdown=-0.10943133691610096, total_return=1.5405219109863268, turnover=2.023549797934384, n_trades=5250
    oos: cagr=0.0588879390266035, ann_vol=0.09508725654034632, sharpe=0.6492878717262837, sharpe_excess=0.3274389935757925, max_drawdown=-0.21657204056609902, total_return=0.32971005408149257, turnover=1.9824310572049502, n_trades=2457
    full: cagr=0.07437773723343688, ann_vol=0.07532842396132643, sharpe=0.9901668843452893, sharpe_excess=0.8323048157882889, max_drawdown=-0.2165720405660988, total_return=2.3781575276528297, turnover=2.0114815052273123, n_trades=7707
    benchmark SPY full: cagr=0.14724513031882114, ann_vol=0.1789858565713259, sharpe=0.857330477741663, sharpe_excess=0.7907961481293594, max_drawdown=-0.33699813656214483, total_return=9.286365255697001, turnover=0.05890413072229087, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 590_deeparies_adaptive_rebalancing_interval\nuniverse: [SPY, QQQ, IWM, EFA, EEM, IEF, GLD]\nbenchmark: SPY\nrisk_free_proxy: BIL\n\ndata_start: '2008-01-01'\nstart: '2009-01-01'\nis_end: '2020-12-31'\nend: '2025-12-31'\nannualization_sessions: 252\n\nlag_sessions: 1\ncost_bps: 2.0\nslippage_bps: 3.0\n\npolicy:\n  intervals: [1, 5, 20]\n  fast_vol_sessions: 20\n  slow_vol_sessions: 126\n  calm_vol_ratio: 0.85\n  shock_vol_ratio: 1.25\n  medium_weight_drift: 0.03\n  large_weight_drift: 0.06\n  covariance_sessions: 63\n  covariance_shrinkage: 0.25\n\nrobustness:\n  shock_vol_ratios: [1.15, 1.35]\n  large_weight_drifts: [0.05, 0.07]\n  fixed_interval: 5\n  friction_multiplier: 2.0\n"
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
#!/usr/bin/env python3
"""C194 adaptation of DeepAries (Kim et al., 2025).

Paper: https://arxiv.org/abs/2510.14985
Authors' implementation (no declared license): https://github.com/dmis-lab/DeepAries
Risk-budget objective reference (Apache-2.0):
https://github.com/QuantConnect/Lean/blob/master/Algorithm.Framework/Portfolio/RiskParityPortfolioOptimizer.py
The strategy draws on the published methods and uses independent code.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
FIELDS = ["Open", "Close", "Dividends", "Capital Gains", "Stock Splits"]
PRICE_ADJUSTMENT = (
    "yfinance auto_adjust=False, split-normalized OHLC with ex-date "
    "distributions reinvested at close (total-return proxy)"
)


def validate_prices(prices, tickers, start, end):
    expected = pd.MultiIndex.from_product([tickers, FIELDS])
    if (
        prices.empty
        or not prices.columns.equals(expected)
        or not prices.index.is_unique
        or not prices.index.is_monotonic_increasing
        or not start <= prices.index[0] <= start + pd.Timedelta(days=7)
        or not end - pd.Timedelta(days=7) <= prices.index[-1] <= end
        or not np.isfinite(prices.to_numpy()).all()
    ):
        raise ValueError("Yahoo snapshot has incomplete fields, dates, or quotes")

    fields = prices.columns.get_level_values(1)
    if (prices.loc[:, fields.isin(["Open", "Close"])] <= 0).any().any():
        raise ValueError("Nonpositive Yahoo price")
    if (prices.loc[:, ~fields.isin(["Open", "Close"])] < 0).any().any():
        raise ValueError("Negative Yahoo corporate-action value")


def load_prices(config):
    start = pd.Timestamp(config["data_start"])
    end = pd.Timestamp(config["end"])
    tickers = list(dict.fromkeys([*config["universe"], config["benchmark"],
                                 config["risk_free_proxy"]]))
    cache = ROOT / "data_cache"
    cache.mkdir(exist_ok=True)
    if not all(ticker.isalnum() for ticker in tickers):
        raise ValueError("This strategy expects plain ETF symbols")
    key = "_".join(tickers)
    snapshot = cache / f"yf_split_bars_{key}_{start.date()}_{end.date()}.csv"

    if not snapshot.exists():
        histories = {}
        with TemporaryDirectory(prefix="yf_", dir=cache) as temp_cache:
            yf.set_tz_cache_location(temp_cache)
            for ticker in tickers:
                history = yf.Ticker(ticker).history(
                    start=start.date().isoformat(),
                    end=(end + pd.Timedelta(days=1)).date().isoformat(),
                    interval="1d", auto_adjust=False, actions=True,
                    timeout=15,
                )
                if history.empty:
                    raise RuntimeError(f"yfinance returned no daily history for {ticker}")
                history.index = history.index.tz_localize(None).rename("date")
                if "Capital Gains" not in history:
                    history["Capital Gains"] = 0.0
                histories[ticker] = history[FIELDS]

        reference = histories[config["benchmark"]].index
        for ticker, frame in histories.items():
            if not frame.index.equals(reference):
                raise ValueError(f"Missing or extra {ticker} dates on the benchmark calendar")
        prices = pd.concat(histories, axis=1)
        validate_prices(prices, tickers, start, end)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache,
                                suffix=".tmp", delete=False) as pending_file:
            pending_path = Path(pending_file.name)
            try:
                pending_file.write(f"# fetched_at={fetched_at}\n")
                prices.to_csv(pending_file)
                pending_file.close()
                pending_path.replace(snapshot)
            finally:
                pending_path.unlink(missing_ok=True)

    with snapshot.open(encoding="utf-8") as cached:
        header = cached.readline().strip()
        if not header.startswith("# fetched_at="):
            raise ValueError(f"Missing download time in {snapshot.name}")
        fetched_at = header.removeprefix("# fetched_at=")
        if datetime.fromisoformat(fetched_at).utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("Download time must be in UTC")
        prices = pd.read_csv(cached, header=[0, 1], index_col=0, parse_dates=True)
    validate_prices(prices, tickers, start, end)
    return prices, fetched_at


@dataclass(frozen=True)
class Market:
    overnight: pd.DataFrame
    intraday: pd.DataFrame
    distributions: pd.DataFrame

    @property
    def returns(self):
        return self.overnight * self.intraday + self.distributions - 1.0


def market_from_quotes(prices):
    opens = prices.xs("Open", axis=1, level=1)
    closes = prices.xs("Close", axis=1, level=1)
    dividends = prices.xs("Dividends", axis=1, level=1)
    capital_gains = prices.xs("Capital Gains", axis=1, level=1)
    previous_close = closes.shift(1)
    # Yahoo OHLC and payouts share a split-normalized unit. Ratios cancel that unit.
    return Market(opens / previous_close, closes / opens,
                  (dividends + capital_gains) / previous_close)


def risk_parity_weights(returns, shrinkage):
    covariance = np.cov(returns, rowvar=False)
    diagonal = np.diag(covariance).copy()
    if not (diagonal > 0).all():
        raise ValueError("Risk parity requires positive asset variances")
    covariance = (1.0 - shrinkage) * covariance + shrinkage * np.diag(diagonal)
    covariance /= diagonal.mean()
    diagonal = np.diag(covariance)
    size = len(diagonal)
    budget = 1.0 / size
    allocation = np.sqrt(budget / diagonal)

    # Minimize 1/2 x' Cov x - sum(budget * log(x)) with positive coordinate updates.
    for _ in range(100):
        for asset in range(size):
            curvature = diagonal[asset]
            slope = covariance[asset] @ allocation - curvature * allocation[asset]
            allocation[asset] = (np.hypot(slope, 2.0 * np.sqrt(curvature * budget))
                                 - slope) / (2.0 * curvature)
        contribution = allocation * (covariance @ allocation)
        if np.max(np.abs(contribution - budget)) < 1e-10:
            return allocation / allocation.sum()
    raise ArithmeticError("Equal-risk contribution optimizer failed to converge")


def build_signals(returns, config):
    policy = config["policy"]
    fast = policy["fast_vol_sessions"]
    slow = policy["slow_vol_sessions"]
    span = policy["covariance_sessions"]
    lag = config["lag_sessions"]
    first_signal = returns.index.searchsorted(config["start"]) - lag
    if first_signal < max(slow, span):
        raise ValueError("Insufficient history for the first close signal")

    basket = returns.mean(axis=1, skipna=False)
    ratio = (basket.rolling(fast).std() / basket.rolling(slow).std()).to_numpy()
    if not np.isfinite(ratio[first_signal:]).all():
        raise ValueError("Nonfinite volatility ratio during the decision window")

    values = returns.to_numpy()
    targets = np.full(values.shape, np.nan)
    for row in range(first_signal, len(values) - lag):
        window = values[row - span + 1:row + 1]
        targets[row] = risk_parity_weights(window, policy["covariance_shrinkage"])
    return ratio, targets


def friction_rate(config):
    cost = float(config["cost_bps"])
    slippage = float(config["slippage_bps"])
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Costs and slippage must be nonnegative and total less than 10,000 bps")
    return (cost + slippage) / 10_000


def rebalance(open_positions, cash, target, rate):
    """Solve for the post-fee value so sells and available cash fund every purchase."""
    opening_value = cash + open_positions.sum()
    low, high = 0.0, opening_value
    for _ in range(48):
        budget = (low + high) / 2.0
        gross_traded = np.abs(target * budget - open_positions).sum()
        if budget + rate * gross_traded <= opening_value:
            low = budget
        else:
            high = budget

    new_positions = target * low
    trade_amounts = np.abs(new_positions - open_positions)
    gross_traded = trade_amounts.sum()
    cash = opening_value - new_positions.sum() - rate * gross_traded
    if cash < -1e-12 * opening_value or (new_positions < 0).any():
        raise ArithmeticError("The rebalance would exceed available cash or sell short")
    n_trades = np.count_nonzero(trade_amounts > 1e-10 * opening_value)
    return new_positions, max(0.0, cash), gross_traded / opening_value, n_trades


def choose_interval(vol_ratio, drift, policy):
    short, medium, long = policy["intervals"]
    if vol_ratio >= policy["shock_vol_ratio"] or drift >= policy["large_weight_drift"]:
        return short
    if vol_ratio <= policy["calm_vol_ratio"] and drift < policy["medium_weight_drift"]:
        return long
    return medium


def simulate(market, config, signals, *, fixed_interval=None):
    universe = config["universe"]
    dates = market.overnight.index
    opening_moves = market.overnight[universe].to_numpy()
    intraday_moves = market.intraday[universe].to_numpy()
    payouts = market.distributions[universe].to_numpy()
    ratios, targets = signals
    lag = config["lag_sessions"]
    first = dates.searchsorted(config["start"])
    rate = friction_rate(config)
    positions = np.zeros(len(universe))
    cash, previous_value = 1.0, 1.0
    last_fill = None
    pending_signal = first - lag
    pending_interval = 0
    rows = []

    for row in range(first, len(dates)):
        # Prior-close units establish payout eligibility. Reinvest only at the close.
        distribution = positions * payouts[row]
        positions *= opening_moves[row]
        turnover, n_trades, signal_row, interval = 0.0, 0, -1, 0

        if pending_signal is not None and row == pending_signal + lag:
            positions, cash, turnover, n_trades = rebalance(
                positions, cash, targets[pending_signal], rate,
            )
            signal_row, interval = pending_signal, pending_interval
            last_fill, pending_signal = row, None

        positions *= intraday_moves[row]
        positions += distribution
        value = cash + positions.sum()
        rows.append((value / previous_value - 1.0, turnover, n_trades,
                     row, signal_row, interval, cash))
        previous_value = value

        if pending_signal is None and row + lag < len(dates):
            drift = 0.5 * (np.abs(positions / value - targets[row]).sum() + cash / value)
            interval = fixed_interval or choose_interval(ratios[row], drift, config["policy"])
            if row + lag - last_fill >= interval:
                pending_signal, pending_interval = row, interval

    return pd.DataFrame(rows, index=dates[first:], columns=[
        "return", "turnover", "n_trades", "row", "signal_row", "interval", "cash",
    ])


def buy_and_hold(market, daily_results, config):
    ticker = config["benchmark"]
    dates = daily_results.index
    benchmark = pd.DataFrame({"return": market.returns.loc[dates, ticker],
                              "turnover": 0.0, "n_trades": 0})
    first = dates[0]
    invested = 1.0 / (1.0 + friction_rate(config))
    benchmark.loc[first, "return"] = invested * market.intraday.loc[first, ticker] - 1.0
    benchmark.loc[first, "turnover"] = invested
    benchmark.loc[first, "n_trades"] = 1
    return benchmark


def metrics(daily_results, reference, year):
    returns = daily_results["return"]
    excess = returns - reference.loc[returns.index]
    growth = np.r_[1.0, (1.0 + returns).cumprod().to_numpy()]
    ann_vol = returns.std() * np.sqrt(year)
    excess_vol = excess.std() * np.sqrt(year)

    def sharpe(series, volatility):
        return float(series.mean() * year / volatility) if volatility > 0 else None

    return {
        "cagr": float(growth[-1] ** (year / len(returns)) - 1.0),
        "ann_vol": float(ann_vol),
        "sharpe": sharpe(returns, ann_vol),
        "sharpe_excess": sharpe(excess, excess_vol),
        "max_drawdown": float(np.min(growth / np.maximum.accumulate(growth) - 1.0)),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(year * daily_results["turnover"].mean()),
        "n_trades": int(daily_results["n_trades"].sum()),
    }


def report_robustness(market, config, signals, baseline, reference):
    robustness = config["robustness"]
    year = config["annualization_sessions"]
    print("ROBUSTNESS (IS only, default retained)")
    print("default:", json.dumps(metrics(baseline, reference, year), allow_nan=False))
    n_trials = 1
    for parameter, values in [("shock_vol_ratio", robustness["shock_vol_ratios"]),
                              ("large_weight_drift", robustness["large_weight_drifts"])]:
        for value in values:
            variant = {**config, "policy": {**config["policy"], parameter: value}}
            result = simulate(market, variant, signals)
            n_trials += 1
            print(f"{parameter}={value}:", json.dumps(metrics(result, reference, year),
                                                        allow_nan=False))

    interval = robustness["fixed_interval"]
    fixed = simulate(market, config, signals, fixed_interval=interval)
    n_trials += 1
    print(f"fixed_interval={interval}:", json.dumps(metrics(fixed, reference, year),
                                                    allow_nan=False))

    multiplier = robustness["friction_multiplier"]
    high_friction = {**config, "cost_bps": config["cost_bps"] * multiplier,
                     "slippage_bps": config["slippage_bps"] * multiplier}
    result = simulate(market, high_friction, signals)
    n_trials += 1
    print(f"friction_multiplier={multiplier}:", json.dumps(metrics(result, reference, year),
                                                             allow_nan=False))
    return n_trials


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if not (config["data_start"] < config["start"] < config["is_end"] < config["end"]):
        raise ValueError("Expected ascending data start, report start, IS end and report end")
    if not isinstance(config["lag_sessions"], int) or config["lag_sessions"] < 1:
        raise ValueError("Execution lag must be at least one session")
    policy = config["policy"]
    short, medium, long = policy["intervals"]
    if not (1 <= short < medium < long and 0 < policy["covariance_shrinkage"] <= 1
            and 2 <= policy["fast_vol_sessions"] < policy["slow_vol_sessions"]
            and 2 <= policy["covariance_sessions"]
            and 0 < policy["calm_vol_ratio"] < policy["shock_vol_ratio"]
            and 0 < policy["medium_weight_drift"] < policy["large_weight_drift"] < 1
            and len(set(config["universe"])) == len(config["universe"]) >= 2):
        raise ValueError("Invalid adaptive policy settings")

    prices, fetched_at = load_prices(config)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}")
    print(f"Universe: {', '.join(config['universe'])}")
    print(f"Execution: {config['lag_sessions']} session lag, open, "
          f"{config['cost_bps']} cost bps + {config['slippage_bps']} slippage bps per side")

    is_end = config["is_end"]
    is_market = market_from_quotes(prices.loc[:is_end])
    is_returns = is_market.returns
    is_signals = build_signals(is_returns[config["universe"]], config)
    is_baseline = simulate(is_market, config, is_signals)
    risk_free = config["risk_free_proxy"]
    n_trials = report_robustness(is_market, config, is_signals, is_baseline, is_returns[risk_free])

    market = market_from_quotes(prices)
    signals = build_signals(market.returns[config["universe"]], config)
    daily_results = simulate(market, config, signals)
    reference = market.returns[risk_free]
    benchmark = buy_and_hold(market, daily_results, config)
    is_mask = daily_results.index <= pd.Timestamp(is_end)
    if is_mask.sum() < 2 or (~is_mask).sum() < 2:
        raise ValueError("Both reporting partitions need multiple sessions")

    fills = daily_results.loc[daily_results["signal_row"] >= 0]
    if (not (fills["row"] - fills["signal_row"] == config["lag_sessions"]).all()
            or (daily_results["cash"] < 0).any()
            or daily_results.loc[~is_mask, "n_trades"].sum() == 0):
        raise ArithmeticError("Execution lag, funding, or real-trading check failed")
    pd.testing.assert_frame_equal(daily_results.loc[is_mask], is_baseline, check_exact=True)

    first, last = daily_results.index[[0, -1]].date
    print(f"Date Range: {first} to {last}")
    print(f"Split: IS through {is_end}, OOS from {daily_results.index[~is_mask][0].date()}")
    print(f"PARAM_COMBOS_TRIED: {n_trials}")
    print("LOOK_AHEAD_SELF_CHECK: close signal -> lagged open sizing, close data and "
          "ex-date payouts after execution, split-normalized price ratios, IS prefix "
          "replayed without OOS quotes, default policy frozen")
    print("Scheduled fills by selected interval:",
          fills.loc[fills["interval"] > 0, "interval"].value_counts().sort_index().to_dict())
    year = config["annualization_sessions"]
    periods = {"is": is_mask, "oos": ~is_mask, "full": slice(None)}
    results = {period: metrics(daily_results.loc[mask], reference, year)
               for period, mask in periods.items()}
    benchmark_results = {period: metrics(benchmark.loc[mask], reference, year)
                         for period, mask in periods.items()}
    for period in periods:
        print(f"{period.upper()} {config['strategy']}:",
              json.dumps(results[period], allow_nan=False))
        print(f"{period.upper()} {config['benchmark']} benchmark:",
              json.dumps(benchmark_results[period], allow_nan=False))
    print("已回测 | pending review")
    print("RESULTS_JSON: " + json.dumps({
        "strategy": config["strategy"], "data_range": [str(first), str(last)], **results,
        "benchmark": {"ticker": config["benchmark"], **benchmark_results},
    }, allow_nan=False))


if __name__ == "__main__":
    main()

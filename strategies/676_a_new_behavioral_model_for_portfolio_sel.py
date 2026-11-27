"""676_a_new_behavioral_model_for_portfolio_sel
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Status: PENDING — not reviewed by Gatekeeper, not accepted
Run (one command): python strategies/676_a_new_behavioral_model_for_portfolio_sel.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 9532cd9f815cc2d43ceb143b2f1416acf34e9864 on 2026-11-27 20:00 Europe/Oslo, exit 0, output sha256 8afc515921d150cb9fedf8fb5e76ee7383b0e52efa042d498de7a960c4c27cd8
Data source: yfinance 公开日线，使用 Open、Close、Dividends 和 Capital Gains。资产池为 SPY、EFA、EEM、IEF、TLT、LQD、GLD、DBC。BIL 仅用于计算超额 Sharpe。
Data fetched: 2026-11-27T19:00:33+00:00
Sample period: 数据请求从 2008-01-01 开始用于热身，正式回测为 2011-01-03 至 2025-12-31。   data_range (from output): {'start': '2011-01-03', 'end': '2025-12-31'}
IS/OOS split: IS 为 2011-01-03 至 2018-12-31，OOS 为 2019-01-02 至 2025-12-31。配置分界为 2019-01-01，按首个达到该日期的交易日切分。六个稳健性变体仅在 IS 运行，默认规格连续模拟并分段报告，OOS 承接原有持仓及调仓节奏。
Costs and slippage: 默认单边手续费 2 bps，滑点 3 bps。滑点作为与手续费合计的比例成本建模，从开盘净卖出名义额扣除 5 bps，向开盘净买入名义额加收 5 bps，并纳入自融资预算。IS 的高摩擦变体为单边手续费 6 bps 加滑点 9 bps。
Signal lag: 调仓权重使用截至信号日 t 收盘的收益和状态，默认在下一交易日开盘模拟成交，每 20 个交易日调仓。
Price adjustment: yfinance split-adjusted Open/Close; ex-date distributions reinvested at open
Survivorship bias: 固定 ETF 池仅覆盖当时已上市且当前仍有历史记录的基金，存在事后选池及幸存者偏差。
Parameter combinations tried: 7
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.04262606553083215, ann_vol=0.14245333943523128, sharpe=0.3643221080128922, sharpe_excess=0.3451525013594587, max_drawdown=-0.2954978989095177, total_return=0.3955354371403277, turnover=6.966002509712782, n_trades=118
    oos: cagr=0.17722008511425003, ann_vol=0.1823968721143738, sharpe=0.9863484423927005, sharpe_excess=0.8477541929504713, max_drawdown=-0.2734015340911773, total_return=2.125207929503023, turnover=5.616381032770965, n_trades=111
    full: cagr=0.1033968929384459, ann_vol=0.16234629022971017, sharpe=0.6875851836848658, sharpe_excess=0.6059588142935847, max_drawdown=-0.2954978989095177, total_return=3.3613384140533897, turnover=6.336274567131233, n_trades=229
    benchmark  full: cagr=0.05442920199848689, ann_vol=0.08626916821701472, sharpe=0.6576063978309661, sharpe_excess=0.5039886970065198, max_drawdown=-0.19621266780256663, total_return=1.2107043384892018, turnover=0.35649454510416134, n_trades=1512
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: "676_ a_new_behavioral_model_for_portfolio_sel"\nuniverse: [SPY, EFA, EEM, IEF, TLT, LQD, GLD, DBC]\nbenchmark: equal_weight_cross_asset\nrisk_free_proxy: BIL\n\ndates:\n  data_start: 2008-01-01\n  start: 2011-01-03\n  oos_start: 2019-01-01\n  end: 2025-12-31\n\nmodel:\n  lookback_sessions: 500\n  lambda_gain: 0.30\n  lambda_loss: 0.69\n  equity_state_proxy: SPY\n  bond_state_proxy: IEF\n  state_window_sessions: 63\n  kernel_bandwidth: 1.0\n  unconditional_share: 0.50\n\nexecution:\n  rebalance_sessions: 20\n  lag_sessions: 1\n  cost_bps: 2\n  slippage_bps: 3\n\n# Each case changes the default specification for the IS run only.\nrobustness:\n  - {name: lower_gain_lambda, model: {lambda_gain: 0.28}}\n  - {name: higher_gain_lambda, model: {lambda_gain: 0.32}}\n  - {name: lower_loss_lambda, model: {lambda_loss: 0.67}}\n  - {name: higher_loss_lambda, model: {lambda_loss: 0.71}}\n  - {name: uniform_scenarios, model: {unconditional_share: 1.0}}\n  - {name: higher_friction, execution: {cost_bps: 6, slippage_bps: 9}}\n'
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
"""C249 HF/HE utility with market-conditioned cross-asset scenarios.

Paper: https://arxiv.org/abs/2312.10749
Full text: https://arxiv.org/html/2312.10749
Research references (ideas only, independently written implementation):
https://www.francescocesarone.com/papers
https://www.francescocesarone.com/data-sets
https://doi.org/10.1016/j.ejor.2025.08.040
https://github.com/R-Finance/Meucci/blob/master/demo/FullFlexProbs.R
https://github.com/R-Finance/Meucci/blob/master/DESCRIPTION (GPL)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2312126
https://github.com/dcajasn/Riskfolio-Lib/blob/master/riskfolio/src/RiskFunctions.py
https://github.com/dcajasn/Riskfolio-Lib/blob/master/LICENSE.txt (BSD-3-Clause)
https://pypi.org/project/riskfolio-lib/
https://www.quantconnect.com/research/15928/sortino-portfolio-optimization-with-alpha-streams-algorithms/
https://www.quantconnect.com/forum/discussion/11373/alpha-streams-portfolio-optimization-notebook/p1
"""

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import signal
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import softmax
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252
PRICE_ADJUSTMENT = "yfinance split-adjusted Open/Close; ex-date distributions reinvested at open"


def validate_bars(bars, symbols, start, end):
    required = pd.MultiIndex.from_product(
        [["Open", "Close", "Dividends", "Capital Gains"], symbols]
    )
    if bars.empty or not required.isin(bars.columns).all():
        raise ValueError("Incomplete public daily bars")
    if not bars.index.is_monotonic_increasing or not bars.index.is_unique:
        raise ValueError("Daily bars must have strictly increasing dates")
    tolerance = pd.Timedelta(days=7)
    if not (start <= bars.index[0] <= start + tolerance
            and end - tolerance <= bars.index[-1] <= end):
        raise ValueError(f"Incomplete date range {bars.index[0]} to {bars.index[-1]}")
    snapshot = bars.loc[:, required]
    if not np.isfinite(snapshot.to_numpy()).all():
        bad = snapshot.loc[snapshot.isna().any(axis=1)]
        raise ValueError(f"Missing or invalid ETF bars. First missing sessions {list(bad.index[:5])}")
    if (snapshot[["Open", "Close"]] <= 0).to_numpy().any():
        raise ValueError("Nonpositive ETF price")
    return snapshot


def load_bars(symbols, start, end):
    CACHE_DIR.mkdir(exist_ok=True)
    cache_key = json.dumps(["yf-split-ohlc-distributions-v1", *symbols, str(start), str(end)])
    digest = hashlib.sha256(cache_key.encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"daily_{digest}.csv.gz"
    if cache_file.exists():
        with gzip.open(cache_file, "rt") as source:
            fetched_at = source.readline().strip()
            datetime.fromisoformat(fetched_at)
            bars = pd.read_csv(
                source, header=[0, 1], index_col=0, parse_dates=True,
                float_precision="round_trip",
            )
        return validate_bars(bars, symbols, start, end), fetched_at

    # Only the public price snapshot persists. Keep the client's anonymous session state temporary.
    with TemporaryDirectory(dir=CACHE_DIR) as session_dir:
        yf.set_tz_cache_location(session_dir)
        for attempt in range(2):
            try:
                bars = yf.download(
                    symbols, start=start.date(), end=(end + pd.Timedelta(days=1)).date(),
                    interval="1d", auto_adjust=False, actions=True,
                    threads=False, ignore_tz=True, progress=False, timeout=12,
                )
                # Yahoo can omit Capital Gains for a ticker without such distributions.
                for symbol in symbols:
                    if ("Capital Gains", symbol) not in bars.columns:
                        bars["Capital Gains", symbol] = 0.0
                bars = validate_bars(bars, symbols, start, end)
                break
            except (ValueError, OSError) as error:
                if attempt == 1:
                    raise RuntimeError("Could not retrieve complete yfinance data") from error

    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with NamedTemporaryFile(dir=CACHE_DIR, suffix=".csv.gz", delete=False) as temporary:
        staging_file = Path(temporary.name)
    try:
        with gzip.open(staging_file, "wt") as destination:
            destination.write(fetched_at + "\n")
            bars.to_csv(destination)
        staging_file.replace(cache_file)
    finally:
        staging_file.unlink(missing_ok=True)
    return bars, fetched_at


def return_factors(bars):
    opens, closes = bars["Open"], bars["Close"]
    distributions = bars["Dividends"] + bars["Capital Gains"]
    overnight = (opens + distributions) / closes.shift(1)
    intraday = closes / opens
    return overnight, intraday, overnight * intraday - 1.0


def market_states(returns, model):
    window = model["state_window_sessions"]
    equity_vol = returns[model["equity_state_proxy"]].rolling(window).std(ddof=0)
    bond_log_return = np.log1p(returns[model["bond_state_proxy"]])
    bond_return = np.expm1(bond_log_return.rolling(window).sum())
    return np.column_stack((equity_vol, bond_return))


def scenario_probabilities(past_states, current_state, model):
    count = len(past_states)
    uniform_share = model["unconditional_share"]
    if uniform_share == 1.0:
        return np.full(count, 1.0 / count)
    scale = past_states.std(axis=0)
    distance = (past_states - current_state) / np.where(scale > 1e-12, scale, 1.0)
    squared_distance = np.square(distance / model["kernel_bandwidth"]).sum(axis=1)
    return (1.0 - uniform_share) * softmax(-0.5 * squared_distance) + uniform_share / count


def hfhe_objective(weights, scenarios, probabilities, coefficients):
    """Negative weighted HF/HE utility and its derivative away from the kinks."""
    returns = scenarios @ weights
    gradient_by_return = probabilities.copy()
    utility = probabilities @ returns
    for coefficient, side in zip(coefficients, (1, -1), strict=True):
        active = side * returns > 0
        part = np.where(active, returns, 0.0)
        centered = part - probabilities @ part
        signs = np.sign(centered)
        utility += coefficient * (probabilities @ np.abs(centered))
        gradient_by_return += coefficient * active * probabilities * (signs - probabilities @ signs)
    return -utility, -(scenarios.T @ gradient_by_return)


def optimize_weights(scenarios, probabilities, model):
    asset_count = scenarios.shape[1]
    equal = np.full(asset_count, 1.0 / asset_count)
    starts = [equal, *np.eye(asset_count)]
    coefficients = (2 * model["lambda_gain"] - 1, 2 * model["lambda_loss"] - 1)
    args = (100 * scenarios, probabilities, coefficients)
    budget = {
        "type": "eq",
        "fun": lambda weights: weights.sum() - 1,
        "jac": lambda weights: np.ones_like(weights),
    }
    best = min(starts, key=lambda start: hfhe_objective(start, *args)[0])
    best_score = hfhe_objective(best, *args)[0]
    for start in starts:
        result = minimize(
            hfhe_objective, start, args=args, jac=True, method="SLSQP",
            bounds=[(0.0, 1.0)] * asset_count, constraints=budget,
            options={"maxiter": 75, "ftol": 1e-10},
        )
        candidate = result.x
        if not np.isfinite(candidate).all() or abs(candidate.sum() - 1.0) > 1e-7:
            continue
        if candidate.min() < -1e-7 or candidate.max() > 1.0 + 1e-7:
            continue
        candidate = np.maximum(candidate, 0.0)
        candidate[candidate < 1e-12] = 0.0
        candidate /= candidate.sum()
        score = hfhe_objective(candidate, *args)[0]
        if score < best_score:
            best, best_score = candidate, score
    return best


def target_weights(returns, first_fill, model, execution):
    lag = execution["lag_sessions"]
    lookback = model["lookback_sessions"]
    states = market_states(returns, model)
    daily = returns.to_numpy()
    for fill in range(first_fill, len(daily), execution["rebalance_sessions"]):
        signal_day = fill - lag
        first_scenario = signal_day - lookback + 1
        if first_scenario < 1:
            raise ValueError("Insufficient warm-up sessions for lagged market states")
        scenarios = daily[first_scenario:signal_day + 1]
        labels = states[first_scenario - 1:signal_day]
        current_state = states[signal_day]
        if not all(np.isfinite(values).all() for values in (scenarios, labels, current_state)):
            raise ValueError("Incomplete warm-up or invalid returns at a signal date")
        probabilities = scenario_probabilities(labels, current_state, model)
        yield fill, optimize_weights(scenarios, probabilities, model)


def rebalance(open_values, cash, weights, friction):
    """Fund all net trades from opening values after proportional costs."""
    open_total = cash + open_values.sum()
    open_fractions = open_values / open_total

    def budget(invested_fraction):
        turnover = np.abs(weights * invested_fraction - open_fractions).sum()
        return invested_fraction + friction * turnover - 1.0

    invested_fraction = brentq(budget, 0.0, 1.0, xtol=1e-15)
    desired = invested_fraction * weights * open_total
    changes = desired - open_values
    sells = np.maximum(-changes, 0.0)
    buys = np.maximum(changes, 0.0)
    cash_available = cash + (1.0 - friction) * sells.sum()
    buy_cost = (1.0 + friction) * buys.sum()
    if buy_cost > cash_available:
        # A root rounded upward can leave a residual at floating-point precision.
        buys *= cash_available / buy_cost
        buy_cost = min(cash_available, (1.0 + friction) * buys.sum())
    gross = sells + buys
    return (
        open_values - sells + buys,
        cash_available - buy_cost,
        gross.sum() / open_total,
        int(np.count_nonzero(gross > open_total * 1e-10)),
    )


def simulate(overnight, intraday, first_fill, targets, execution):
    friction = (execution["cost_bps"] + execution["slippage_bps"]) / 10_000
    gaps, intraday_marks = overnight.to_numpy(), intraday.to_numpy()
    positions = np.zeros(overnight.shape[1])
    cash, previous_value = 1.0, 1.0
    rows = []
    for day in range(first_fill, len(overnight)):
        positions = positions * gaps[day]
        turnover, trades = 0.0, 0
        if day in targets:
            positions, cash, turnover, trades = rebalance(positions, cash, targets[day], friction)
        if cash < 0 or (positions < 0).any():
            raise RuntimeError("Invalid cash or positions after open")
        positions = positions * intraday_marks[day]
        close_value = cash + positions.sum()
        rows.append((close_value / previous_value - 1.0, turnover, trades))
        previous_value = close_value
    return pd.DataFrame(rows, index=overnight.index[first_fill:], columns=["return", "turnover", "trades"])


def period_metrics(daily_results, risk_free):
    returns = daily_results["return"]
    excess = returns - risk_free.loc[returns.index]
    growth = np.concatenate(([1.0], (1.0 + returns).cumprod()))
    drawdown = growth / np.maximum.accumulate(growth) - 1.0
    annualizer = np.sqrt(SESSIONS_PER_YEAR)

    def sharpe(series):
        deviation = series.std(ddof=1)
        return float(series.mean() / deviation * annualizer) if deviation > 0 else None

    return {
        "cagr": float(growth[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1.0),
        "ann_vol": float(returns.std(ddof=1) * annualizer),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(excess),
        "max_drawdown": float(drawdown.min()),
        "total_return": float(growth[-1] - 1.0),
        "turnover": float(daily_results["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(daily_results["trades"].sum()),
    }


def validate_settings(universe, model, execution):
    if execution["lag_sessions"] < 1 or execution["rebalance_sessions"] < 1:
        raise ValueError("Use at least one session for both execution lag and cadence")
    cost, slippage = execution["cost_bps"], execution["slippage_bps"]
    if not (cost >= 0 and slippage >= 0 and cost + slippage < 10_000):
        raise ValueError("Invalid cost or slippage")
    for key in ("lambda_gain", "lambda_loss", "unconditional_share"):
        if not 0.0 <= model[key] <= 1.0:
            raise ValueError(f"{key} must be between zero and one")
    if not (model["lookback_sessions"] >= 2 and model["state_window_sessions"] >= 2
            and 0.0 < model["kernel_bandwidth"] < np.inf):
        raise ValueError("Invalid model windows or kernel bandwidth")
    if not {model["equity_state_proxy"], model["bond_state_proxy"]} <= set(universe):
        raise ValueError("State proxies must be members of the price universe")


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":"))


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    universe = config["universe"]
    proxy = config["risk_free_proxy"]
    dates = {name: pd.Timestamp(date) for name, date in config["dates"].items()}
    model, execution = config["model"], config["execution"]
    validate_settings(universe, model, execution)
    if not dates["data_start"] < dates["start"] < dates["oos_start"] <= dates["end"]:
        raise ValueError("Dates must be ordered")
    if len(set(universe)) != len(universe) or proxy in universe or proxy != "BIL":
        raise ValueError("Use distinct assets and BIL as the separate excess-return proxy")
    bars, fetched_at = load_bars([*universe, proxy], dates["data_start"], dates["end"])
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"PRICE_ADJUSTMENT: {PRICE_ADJUSTMENT}", flush=True)
    overnight, intraday, all_returns = return_factors(bars)
    first_fill = bars.index.searchsorted(dates["start"])
    split = bars.index.searchsorted(dates["oos_start"])
    if split - first_fill < 2 or len(bars) - split < 2:
        raise ValueError("Both IS and OOS need at least two sessions")
    risk_free = all_returns[proxy]
    overnight, intraday, returns = overnight[universe], intraday[universe], all_returns[universe]
    if not np.isfinite(all_returns.iloc[1:].to_numpy()).all():
        raise ValueError("Invalid total-return observations")

    tried = 0
    print("ROBUSTNESS (IS only)", flush=True)
    for case in config["robustness"]:
        case_model = model | case.get("model", {})
        case_execution = execution | case.get("execution", {})
        validate_settings(universe, case_model, case_execution)
        targets = dict(target_weights(returns.iloc[:split], first_fill, case_model, case_execution))
        case_results = simulate(
            overnight.iloc[:split], intraday.iloc[:split], first_fill, targets, case_execution,
        )
        print(f"{case['name']} " + encode(period_metrics(case_results, risk_free.iloc[:split])), flush=True)
        tried += 1

    targets = dict(target_weights(returns, first_fill, model, execution))
    official = simulate(overnight, intraday, first_fill, targets, execution)
    tried += 1
    equal = np.full(len(universe), 1.0 / len(universe))
    benchmark = simulate(overnight, intraday, first_fill, dict.fromkeys(targets, equal), execution)

    sections = {
        "is": official.index[official.index < dates["oos_start"]],
        "oos": official.index[official.index >= dates["oos_start"]],
        "full": official.index,
    }
    scores, benchmark_scores = {}, {"name": config["benchmark"]}
    for label, index in sections.items():
        scores[label] = period_metrics(official.loc[index], risk_free)
        benchmark_scores[label] = period_metrics(benchmark.loc[index], risk_free)
        print(f"{label.upper()} " + encode(scores[label]))
        print(f"BENCHMARK {label.upper()} " + encode(benchmark_scores[label]))
    if not scores["is"]["n_trades"] or not scores["oos"]["n_trades"]:
        raise RuntimeError("The default backtest must contain trades in IS and OOS")
    print(f"PARAM_COMBOS_TRIED: {tried}")
    print("RESULTS_JSON: " + encode({
        "strategy": config["strategy"],
        "data_range": {
            "start": official.index[0].date().isoformat(),
            "end": official.index[-1].date().isoformat(),
        },
        **scores,
        "benchmark": benchmark_scores,
    }))


def stop_after_deadline(_signum, _frame):
    raise TimeoutError("The full research run exceeded the 14-minute deadline")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, stop_after_deadline)
    signal.alarm(14 * 60)
    try:
        main()
    finally:
        signal.alarm(0)

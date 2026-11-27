"""400_noxtrader_lstm_based_stock_return_moment
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/400_noxtrader_lstm_based_stock_return_moment.py   (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit da9754ee51e8d74f175ab50e14ef28c47c764ad1 on 2026-11-27 12:04 Europe/Oslo, exit 0, output sha256 49b0caf51d2e45b7365b879529a8ef7733278d5990d4808d09bdfb897d48e912
Data source: 公开 Yahoo Finance（yfinance）日线。股票池使用复权收盘价和成交量：AAPL、MSFT、AMZN、GOOGL、META、NVDA、JPM、BAC、WMT、COST、PG、JNJ、UNH、XOM、CVX、HD、V、MA。SPY 为买入持有基准，BIL 总回报为超额 Sharpe 的现金代理。（config.yaml:2-4；strategy.py:65-97、112-114、339-348、370-375、391-402；README.md:21）
Data fetched: 2026-11-27T11:04:08+00:00
Sample period: 请求的行情区间为 2017-01-03 至 2025-12-31（含首尾），2017-01-03 起的回测前行情用于预热。运行日志报告的回测区间为 2018-10-04 至 2025-12-31（含首尾）。（config.yaml:6-11；strategy.py:65-71、97-107、362-364；README.md:22；logs/run_20261127_071050.log:5）   data_range (from output): 2018-10-04 to 2025-12-31
IS/OOS split: IS 为 2018-10-04 至 2023-10-31，OOS 为 2023-11-01 至 2025-12-31。以 2018-10-04 至 2022-12-30（IS 的子区间）的扣费后 BIL 超额 Sharpe 选择分位数，并以选定分位数回算全期，因此 IS 和全期统计均包含选参区间。IS 与 OOS 来自同一条连续持仓路径。OOS 沿用选定分位数，模型权重仍按每 10 个交易日滚动重拟合。（config.yaml:6-11、22-23、30-31；strategy.py:265-289、362-368、374-401；README.md:22-23；logs/run_20261127_071050.log:5）
Costs and slippage: 每美元成交名义金额扣除 1 bp 手续费与 5 bp 固定滑点，合计 6 bp。先按当日收益标记原持仓，再按扣费后净值对应的目标头寸与漂移后原头寸之差的绝对额之和扣费，涵盖买卖、开空和回补。SPY 首次建仓也扣费。滑点以固定扣费建模，代码没有按成交价格或市场冲击另行模拟滑点。（config.yaml:13-15；strategy.py:307-329、376、391-394；README.md:25、29）
Signal lag: 以 t 日收盘后的日线收盘价及成交量生成信号，在下一交易日 t+1 的收盘模拟调仓。t+1 收盘时先计入原持仓截至该收盘的收益，再调至目标仓位。新仓位首段价格收益为 t+1 收盘至 t+2 收盘。（config.yaml:16；strategy.py:131-151、265-288、307-328、370-375；README.md:24）
Price adjustment: yfinance auto_adjust total-return
Survivorship bias: 固定股票池由 18 只仍有完整公开历史行情的大盘股构成，历史退出、退市或被收购的证券代表不足，存在幸存者偏差和事后选样偏差。代码逐期使用同一配置股票池，要求各配置标的有有效复权收盘价。（config.yaml:2-4；strategy.py:65-67、95-111、361-371；README.md:15、29）
Parameter combinations tried: 3
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=-0.06792541195283086, ann_vol=0.23226956818325828, sharpe=-0.1866137533049973, sharpe_excess=-0.25563594039504434, max_drawdown=-0.42199908478311965, total_return=-0.29984777982377686, turnover=270.6668657428136, n_trades=7705
    oos: cagr=-0.25619618933014765, ann_vol=0.21125650569153331, sharpe=-1.2943265171788179, sharpe_excess=-1.514287007040158, max_drawdown=-0.5099395016868924, total_return=-0.4715262076176311, turnover=267.21819786133017, n_trades=1817
    full: cagr=-0.1286061054021741, ann_vol=0.2262443405625381, sharpe=-0.495006798666675, sharpe_excess=-0.6058265839479685, max_drawdown=-0.7010429479930944, total_return=-0.6299879009585345, turnover=269.63794999575566, n_trades=9522
    benchmark: cagr=0.13780302025433433, ann_vol=0.19024613818297287, sharpe=0.7740818600224869, sharpe_excess=0.6417864116456042, max_drawdown=-0.3210091642697305, total_return=1.5405569074138241, turnover=0.1305104001017193, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = 'strategy: 400_noxtrader_lstm_based_stock_return_moment\nuniverse: [AAPL, MSFT, AMZN, GOOGL, META, NVDA, JPM, BAC, WMT, COST, PG, JNJ, UNH, XOM, CVX, HD, V, MA]\nbenchmark: SPY\nrisk_free: BIL\n\n# Dates are inclusive. Calibration is a subset of IS.\ndata_start: 2017-01-03\nstart: 2018-10-04\ncalibration_end: 2022-12-30\nis_end: 2023-10-31\nend: 2025-12-31\n\n# Bps per dollar of entry, exit, or rebalance turnover.\ncost_bps: 1.0\nslippage_bps: 5.0\nsignal_lag_sessions: 1\n# Reserve fixed at the prior close. The benchmark buys once.\ncash_buffer: 0.30\nbenchmark_cash_buffer: 0.05\n\n# The original cache had no embedded timestamp. Bind its logged date to its bytes.\nlegacy_snapshot:\n  sha256: 3afec23bd0819fe97ad9b041b95bb0330dfb18beffae7cc6214382a56b077c74\n  reported_fetched_at: "2026-11-27T06:08:43+00:00"\n\nmodel:\n  week_sessions: 5\n  month_sessions: 20\n  lookback_sessions: 10\n  training_sessions: 240\n  refit_sessions: 10\n  hidden_units: 32\n  epochs: 3\n  batch_size: 256\n  learning_rate: 0.001\n  seed: 400\n\n# The middle candidate is fixed until calibration ends. Apply the winner prospectively.\npreselection_score_quantile: 0.80\nscore_quantiles: [0.70, 0.80, 0.90]\n'
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
"""Walk-forward NoxTrader research backtest using public daily data."""

from datetime import datetime, timezone
import gzip
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from threadpoolctl import threadpool_limits
import yaml


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data_cache"
SESSIONS_PER_YEAR = 252


class NYSEHolidays(AbstractHolidayCalendar):
    """Full-session closures for the U.S. equity sample. Half-days are sessions."""

    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("Bush day of mourning", year=2018, month=12, day=5),
        Holiday("Carter day of mourning", year=2025, month=1, day=9),
    ]


def load_config():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    dates = [
        pd.Timestamp(cfg[key])
        for key in ("data_start", "start", "calibration_end", "is_end", "end")
    ]
    if any(left >= right for left, right in zip(dates, dates[1:])):
        raise ValueError("Expected data_start < start < calibration_end < is_end < end.")

    tickers = [*cfg["universe"], cfg["benchmark"], cfg["risk_free"]]
    if (
        len(cfg["universe"]) < 2
        or len(tickers) != len(set(tickers))
        or any(not re.fullmatch(r"[A-Z]{1,5}(?:-[A-Z])?", ticker) for ticker in tickers)
    ):
        raise ValueError("Use distinct U.S. equity/ETF symbols for the universe and references.")
    lag = cfg["signal_lag_sessions"]
    if type(lag) is not int or lag < 1:
        raise ValueError("signal_lag_sessions must be a positive integer.")
    costs_bps = [cfg["cost_bps"], cfg["slippage_bps"]]
    if not all(np.isfinite(bps) and bps >= 0 for bps in costs_bps) or sum(costs_bps) >= 1000:
        raise ValueError("Costs and slippage must be nonnegative and sum to less than 1000 bps.")

    for key in ("cash_buffer", "benchmark_cash_buffer"):
        if not 0 < cfg[key] < 1:
            raise ValueError(f"{key} must be a fraction in (0, 1).")

    model = cfg["model"]
    for key in (
        "week_sessions", "month_sessions", "lookback_sessions", "training_sessions",
        "refit_sessions", "hidden_units", "epochs", "batch_size",
    ):
        if type(model[key]) is not int or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer.")
    if not np.isfinite(model["learning_rate"]) or model["learning_rate"] <= 0:
        raise ValueError("model.learning_rate must be positive and finite.")
    quantiles = cfg["score_quantiles"]
    if (
        not quantiles
        or len(quantiles) != len(set(quantiles))
        or any(not 0 < quantile < 1 for quantile in quantiles)
    ):
        raise ValueError("score_quantiles must contain distinct values in (0, 1).")
    if cfg["preselection_score_quantile"] not in quantiles:
        raise ValueError("preselection_score_quantile must be one of score_quantiles.")
    return cfg


def read_quotes(snapshot):
    with gzip.open(BytesIO(snapshot), "rt", encoding="utf-8") as source:
        first_line = source.readline()
        if first_line.startswith("{"):
            fetched_at = json.loads(first_line)["fetched_at"]
        else:
            fetched_at = None
            source.seek(0)
        quotes = pd.read_csv(
            source, header=[0, 1], index_col=0, parse_dates=True,
            float_precision="round_trip" if fetched_at is not None else None,
        )
    return quotes, fetched_at


def validate_quotes(quotes, tickers, universe, start, end):
    if (
        quotes.empty
        or not isinstance(quotes.columns, pd.MultiIndex)
        or not {"Close", "Volume"} <= set(quotes.columns.get_level_values(0))
    ):
        raise RuntimeError("yfinance returned no usable adjusted closes and volume.")

    dates = pd.DatetimeIndex(quotes.index).tz_localize(None).normalize().as_unit("ns")
    quotes = quotes.set_axis(dates).sort_index().loc[start:end]
    if quotes.index.has_duplicates:
        raise RuntimeError("Duplicate exchange dates in yfinance history.")
    sessions = pd.date_range(start, end, freq="B").difference(NYSEHolidays().holidays(start, end))
    if not sessions.equals(quotes.index):
        missing = sessions.difference(quotes.index).strftime("%Y-%m-%d").tolist()
        extra = quotes.index.difference(sessions).strftime("%Y-%m-%d").tolist()
        raise RuntimeError(f"NYSE calendar mismatch. Missing: {missing[:10]}. Unexpected: {extra[:10]}.")

    close = quotes["Close"].reindex(columns=tickers)
    bad_prices = (~np.isfinite(close) | (close <= 0)).any()
    if bad_prices.any():
        missing = ", ".join(bad_prices[bad_prices].index)
        raise RuntimeError(f"Missing or invalid adjusted closes for {missing}.")

    volume = quotes["Volume"].reindex(columns=universe)
    bad_volume = (~np.isfinite(volume) | (volume <= 0)).all()
    if bad_volume.any():
        missing = ", ".join(bad_volume[bad_volume].index)
        raise RuntimeError(f"No usable volume history for {missing}.")
    return close, volume


def price_history(cfg):
    tickers = [*cfg["universe"], cfg["benchmark"], cfg["risk_free"]]
    start = pd.Timestamp(cfg["data_start"])
    end = pd.Timestamp(cfg["end"])
    # yfinance's end is exclusive.
    end_exclusive = (end + pd.Timedelta(days=1)).date().isoformat()
    request = json.dumps([tickers, start.date().isoformat(), end_exclusive, "auto_adjust"])
    key = hashlib.sha256(request.encode()).hexdigest()[:16]
    cache_path = CACHE / f"daily_{key}.csv.gz"

    download = not cache_path.exists()
    if download:
        import yfinance as yf

        CACHE.mkdir(exist_ok=True)
        yf.set_tz_cache_location(str(CACHE / "yfinance"))
        quotes = yf.download(
            tickers,
            start=start.date().isoformat(),
            end=end_exclusive,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            timeout=15,
        )
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    else:
        snapshot = cache_path.read_bytes()
        quotes, fetched_at = read_quotes(snapshot)

    close, volume = validate_quotes(quotes, tickers, cfg["universe"], start, end)
    provenance = "recorded when yfinance download completed"
    if download:
        with NamedTemporaryFile(dir=CACHE, suffix=".csv.gz", delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            with gzip.open(temp_path, "wt", encoding="utf-8", newline="") as destination:
                destination.write(json.dumps({"fetched_at": fetched_at}) + "\n")
                quotes.to_csv(destination)
            snapshot = temp_path.read_bytes()
            temp_path.replace(cache_path)
        finally:
            temp_path.unlink(missing_ok=True)

    digest = hashlib.sha256(snapshot).hexdigest()
    if fetched_at is None:
        legacy = cfg["legacy_snapshot"]
        if digest != legacy["sha256"]:
            raise RuntimeError("The legacy cache lacks a download timestamp and does not match its recorded SHA-256.")
        fetched_at = legacy["reported_fetched_at"]
        provenance = "original run's cache-mtime proxy, bound to the matching snapshot"
    return close, volume, fetched_at, provenance, digest


def make_examples(close, volume, cfg):
    model = cfg["model"]
    returns = close.pct_change(fill_method=None)
    momentum = returns.diff()
    volume = volume.where(volume > 0)
    fields = [
        returns,
        momentum,
        momentum.diff(),
        close.pct_change(model["week_sessions"], fill_method=None),
        close.pct_change(model["month_sessions"], fill_method=None),
        volume.pct_change(fill_method=None),
    ]
    features = np.stack([field.to_numpy(dtype=np.float32) for field in fields], axis=-1)
    lookback = model["lookback_sessions"]
    windows = np.lib.stride_tricks.sliding_window_view(features, lookback, axis=0)
    windows = windows.transpose(0, 1, 3, 2)
    # Signal at t, fill at t+lag, and realize the holding return at t+lag+1.
    # The label is that holding return minus the return of the prior session.
    label_delay = cfg["signal_lag_sessions"] + 1
    labels = momentum.shift(-label_delay).iloc[lookback - 1:].to_numpy(dtype=np.float32)
    return windows, labels, label_delay


class LSTM:
    """Single LSTM layer and linear readout, trained with MSE and Adam."""

    def __init__(self, n_features, hidden_units, rng):
        self.hidden_units = hidden_units
        width = n_features + hidden_units
        self.kernel = rng.standard_normal((width, 4 * hidden_units), dtype=np.float32)
        self.kernel *= 1 / np.sqrt(width)
        self.bias = np.zeros(4 * hidden_units, dtype=np.float32)
        self.bias[hidden_units:2 * hidden_units] = 1
        self.readout = rng.standard_normal((hidden_units, 1), dtype=np.float32)
        self.readout *= 1 / np.sqrt(hidden_units)
        self.intercept = np.zeros(1, dtype=np.float32)
        self.parameters = (self.kernel, self.bias, self.readout, self.intercept)

    def forward(self, x, training=False):
        hidden = np.zeros((len(x), self.hidden_units), dtype=np.float32)
        cell = np.zeros_like(hidden)
        history = []
        for observation in x.transpose(1, 0, 2):
            joined = np.concatenate((observation, hidden), axis=1)
            input_gate, forget, candidate, output_gate = np.split(
                joined @ self.kernel + self.bias, 4, axis=1,
            )
            input_gate = 1 / (1 + np.exp(-np.clip(input_gate, -30, 30)))
            forget = 1 / (1 + np.exp(-np.clip(forget, -30, 30)))
            candidate = np.tanh(candidate)
            output_gate = 1 / (1 + np.exp(-np.clip(output_gate, -30, 30)))
            previous_cell = cell
            cell = forget * previous_cell + input_gate * candidate
            tanh_cell = np.tanh(cell)
            hidden = output_gate * tanh_cell
            if training:
                history.append((joined, previous_cell, input_gate, forget,
                                candidate, output_gate, tanh_cell))
        return hidden @ self.readout + self.intercept, hidden, history

    def gradients(self, x, y):
        prediction, hidden, history = self.forward(x, training=True)
        error_gradient = 2 * (prediction - y) / len(x)
        readout_gradient = hidden.T @ error_gradient
        intercept_gradient = error_gradient.sum(axis=0)
        hidden_gradient = error_gradient @ self.readout.T
        cell_gradient = np.zeros_like(hidden)
        kernel_gradient = np.zeros_like(self.kernel)
        bias_gradient = np.zeros_like(self.bias)
        recurrent_kernel = self.kernel[-self.hidden_units:]

        for joined, previous, input_gate, forget, candidate, output, tanh_cell in reversed(history):
            output_gradient = hidden_gradient * tanh_cell
            cell_gradient += hidden_gradient * output * (1 - tanh_cell ** 2)
            input_gradient = cell_gradient * candidate
            forget_gradient = cell_gradient * previous
            candidate_gradient = cell_gradient * input_gate
            gates_gradient = np.concatenate(
                (
                    input_gradient * input_gate * (1 - input_gate),
                    forget_gradient * forget * (1 - forget),
                    candidate_gradient * (1 - candidate ** 2),
                    output_gradient * output * (1 - output),
                ),
                axis=1,
            )
            kernel_gradient += joined.T @ gates_gradient
            bias_gradient += gates_gradient.sum(axis=0)
            hidden_gradient = gates_gradient @ recurrent_kernel.T
            cell_gradient *= forget
        return kernel_gradient, bias_gradient, readout_gradient, intercept_gradient

    def fit(self, x, y, cfg, rng):
        first_moment = [np.zeros_like(param) for param in self.parameters]
        second_moment = [np.zeros_like(param) for param in self.parameters]
        step = 0
        batch_size = cfg["batch_size"]
        for _ in range(cfg["epochs"]):
            order = rng.permutation(len(x))
            for offset in range(0, len(order), batch_size):
                rows = order[offset:offset + batch_size]
                gradients = self.gradients(x[rows], y[rows])
                step += 1
                for param, gradient, first, second in zip(
                    self.parameters, gradients, first_moment, second_moment,
                ):
                    first *= 0.9
                    first += 0.1 * gradient
                    second *= 0.999
                    second += 0.001 * gradient ** 2
                    mean = first / (1 - 0.9 ** step)
                    variance = second / (1 - 0.999 ** step)
                    param -= cfg["learning_rate"] * mean / (np.sqrt(variance) + 1e-8)


def fit_window(windows, labels, valid, stop, cfg):
    start = stop - cfg["training_sessions"]
    if start < 0:
        raise ValueError("Increase history before the first execution to fit the training window.")
    usable = valid[start:stop] & np.isfinite(labels[start:stop])
    if np.count_nonzero(usable.any(axis=1)) < cfg["training_sessions"]:
        raise RuntimeError("Each training session must have at least one complete labeled sequence.")
    x = windows[start:stop][usable]
    y = labels[start:stop][usable][:, None]
    center = x.mean(axis=(0, 1))
    scale = np.maximum(x.std(axis=(0, 1)), 1e-6)
    y_center, y_scale = y.mean(), max(y.std(), 1e-6)
    rng = np.random.default_rng(cfg["seed"])
    model = LSTM(x.shape[-1], cfg["hidden_units"], rng)
    model.fit((x - center) / scale, (y - y_center) / y_scale, cfg, rng)
    return model, center, scale, y_center, y_scale


def execution_scores(windows, labels, label_delay, first, last, cfg):
    model_cfg = cfg["model"]
    lag = cfg["signal_lag_sessions"]
    # Convert execution dates into signal indices in the sequence array.
    first_signal = first - lag - (model_cfg["lookback_sessions"] - 1)
    last_signal = last - lag - (model_cfg["lookback_sessions"] - 1)
    valid = np.isfinite(windows).all(axis=(2, 3))
    scores = np.full((last - first + 1, windows.shape[1]), np.nan)
    with threadpool_limits(limits=1, user_api="blas"):
        for i, signal in enumerate(range(first_signal, last_signal + 1)):
            if i % model_cfg["refit_sessions"] == 0:
                # The latest usable label resolves at this signal's close.
                stop = signal - label_delay + 1
                model, center, scale, y_center, y_scale = fit_window(
                    windows, labels, valid, stop, model_cfg,
                )
            eligible = valid[signal]
            if eligible.any():
                x = (windows[signal, eligible] - center) / scale
                prediction, _, _ = model.forward(x)
                prediction = prediction[:, 0] * y_scale + y_center
                if not np.isfinite(prediction).all():
                    raise RuntimeError("LSTM generated a nonfinite prediction.")
                scores[i, eligible] = prediction
    return scores


def target_weights(scores, quantile):
    weights = np.zeros_like(scores)
    for i, row in enumerate(scores):
        eligible = np.isfinite(row)
        available = row[eligible]
        if len(available) < 2:
            continue
        cutoff = np.quantile(np.abs(available), quantile)
        shrunk = np.sign(available) * np.maximum(np.abs(available) - cutoff, 0)
        gross = np.abs(shrunk).sum()
        if gross > 0:
            weights[i, eligible] = shrunk / gross
    return weights


def simulate(close, targets, cash_buffer, cost_rate):
    """Use the first close for sizing only, then size each fill at the prior close."""
    positions = np.zeros(close.shape[1])
    cash = previous_value = 1.0
    daily_results = []
    prices = close.to_numpy()
    sessions = zip(close.index[1:], prices[:-1], prices[1:], targets, strict=True)
    for day, prior_prices, close_prices, target in sessions:
        next_positions = positions
        if target is not None:
            order_budget = previous_value * (1 - cash_buffer)
            next_positions = target * order_budget / prior_prices
        share_changes = next_positions - positions

        before_fill = cash + positions @ close_prices
        if before_fill <= 0:
            raise RuntimeError(f"{day.date()}: portfolio value exhausted before the close fill.")
        traded = np.abs(share_changes) @ close_prices
        cash -= share_changes @ close_prices + cost_rate * traded
        marked_positions = next_positions * close_prices
        net_exposure = marked_positions.sum()
        gross_exposure = np.abs(marked_positions).sum()
        value = cash + net_exposure
        # At the close, reserve short proceeds plus matching collateral from cash.
        free_cash = value - gross_exposure
        if cash < 0 or free_cash < 0 or value <= 0:
            raise RuntimeError(
                f"{day.date()}: fixed close fill breaches cash or gross exposure limits "
                f"(cash={cash:.8g}, uncommitted cash={free_cash:.8g}, value={value:.8g})."
            )
        daily_results.append((
            value / previous_value - 1,
            traded / before_fill,
            np.count_nonzero(share_changes),
            gross_exposure / value,
            net_exposure / value,
            np.max(np.abs(marked_positions)) / value,
            np.count_nonzero(next_positions),
            free_cash / value,
        ))
        positions = next_positions
        previous_value = value
    return pd.DataFrame(daily_results, index=close.index[1:], columns=[
        "return", "turnover", "n_trades", "gross_exposure", "net_exposure",
        "largest_weight", "n_positions", "free_cash_fraction",
    ])


def sharpe(returns):
    volatility = returns.std(ddof=1)
    if volatility == 0 or not np.isfinite(volatility):
        return None
    return float(np.sqrt(SESSIONS_PER_YEAR) * returns.mean() / volatility)


def metrics(history, risk_free):
    returns = history["return"]
    if len(returns) < 2:
        raise ValueError("Each metric period needs at least two sessions.")
    value_path = np.r_[1.0, (1 + returns).cumprod().to_numpy()]
    return {
        "cagr": float(value_path[-1] ** (SESSIONS_PER_YEAR / len(returns)) - 1),
        "ann_vol": float(returns.std(ddof=1) * np.sqrt(SESSIONS_PER_YEAR)),
        "sharpe": sharpe(returns),
        "sharpe_excess": sharpe(returns - risk_free.loc[history.index]),
        "max_drawdown": float(np.min(value_path / np.maximum.accumulate(value_path) - 1)),
        "total_return": float(value_path[-1] - 1),
        "turnover": float(history["turnover"].mean() * SESSIONS_PER_YEAR),
        "n_trades": int(history["n_trades"].sum()),
    }


def main():
    cfg = load_config()
    close, volume, fetched_at, provenance, digest = price_history(cfg)
    print(f"DATA_FETCHED_AT: {fetched_at}", flush=True)
    print(f"DATA_FETCHED_AT_SOURCE: {provenance}")
    print(f"DATA_SNAPSHOT_SHA256: {digest}")
    print("PRICE_ADJUSTMENT: yfinance auto_adjust total-return", flush=True)
    universe = cfg["universe"]
    first = close.index.searchsorted(pd.Timestamp(cfg["start"]))
    last = len(close) - 1
    dates = close.index[first:last + 1]
    is_end = pd.Timestamp(cfg["is_end"])
    is_stop = dates.searchsorted(is_end, side="right")
    calibration_stop = dates.searchsorted(pd.Timestamp(cfg["calibration_end"]), side="right")
    if min(calibration_stop, is_stop - calibration_stop, len(dates) - is_stop) < 2:
        raise ValueError("Each of calibration, post-calibration IS, and OOS needs at least two sessions.")
    selection_start = calibration_stop + cfg["signal_lag_sessions"] - 1
    if selection_start >= is_stop:
        raise ValueError("Leave enough post-calibration IS sessions for the signal lag.")

    windows, labels, delay = make_examples(close[universe], volume, cfg)
    scores = execution_scores(windows, labels, delay, first, last, cfg)
    risk_free = close[cfg["risk_free"]].loc[dates].pct_change(fill_method=None)
    risk_free.iloc[0] = 0.0  # The cash proxy starts at the first fill close as well.
    asset_close = close[universe].iloc[first - 1:last + 1]
    cost_rate = (cfg["cost_bps"] + cfg["slippage_bps"]) / 10_000
    cash_buffer = cfg["cash_buffer"]

    calibration_scores = scores[:calibration_stop]
    baseline = cfg["preselection_score_quantile"]
    trials = {}
    for quantile in cfg["score_quantiles"]:
        trial = simulate(
            asset_close.iloc[:calibration_stop + 1],
            target_weights(calibration_scores, quantile), cash_buffer, cost_rate,
        )
        trials[quantile] = metrics(trial, risk_free)
    selected = max(
        (quantile for quantile in trials if trials[quantile]["sharpe_excess"] is not None),
        key=lambda quantile: (trials[quantile]["sharpe_excess"], quantile == baseline),
        default=baseline,
    )
    print(f"PARAM_COMBOS_TRIED: {len(trials)}")
    print(f"PRESELECTION_SCORE_QUANTILE: {baseline:.2f}")
    print(f"IS_CALIBRATION_SELECTED_SCORE_QUANTILE: {selected:.2f}")
    trial_summary = {
        f"{quantile:.2f}": {key: trial[key] for key in ("sharpe_excess", "turnover", "n_trades")}
        for quantile, trial in trials.items()
    }
    print(f"CALIBRATION_TRIALS: {json.dumps(trial_summary, allow_nan=False)}")

    targets = np.concatenate((
        target_weights(scores[:selection_start], baseline),
        target_weights(scores[selection_start:], selected),
    ))
    portfolio = simulate(asset_close, targets, cash_buffer, cost_rate)
    reference = [cfg["benchmark"]]
    benchmark_targets = [np.ones(1)] + [None] * (len(dates) - 1)
    benchmark = simulate(
        close[reference].iloc[first - 1:last + 1], benchmark_targets,
        cfg["benchmark_cash_buffer"], cost_rate,
    )
    benchmark_name = f'{cfg["benchmark"]} buy and hold'
    result = {
        "strategy": cfg["strategy"],
        "data_range": f"{dates[0].date()} to {dates[-1].date()}",
        "is": metrics(portfolio.iloc[:is_stop], risk_free),
        "oos": metrics(portfolio.iloc[is_stop:], risk_free),
        "full": metrics(portfolio, risk_free),
        "benchmark": {"name": benchmark_name, **metrics(benchmark, risk_free)},
    }
    print(f'TEST_DATES: {result["data_range"]} (IS ends {is_end.date()})')
    print(f"CALIBRATION: {json.dumps(trials[baseline], allow_nan=False)}")
    post_calibration = metrics(portfolio.iloc[calibration_stop:is_stop], risk_free)
    print(f"POST_CALIBRATION_IS: {json.dumps(post_calibration, allow_nan=False)}")
    exposure_fields = ["gross_exposure", "net_exposure", "largest_weight", "n_positions", "free_cash_fraction"]
    exposure = portfolio[exposure_fields].agg(["min", "mean", "max"]).to_dict()
    print(f"CLOSE_EXPOSURE: {json.dumps(exposure, allow_nan=False)}")
    for period in ("is", "oos", "full"):
        print(f"{period.upper()}: {json.dumps(result[period], allow_nan=False)}")
    print(f"BENCHMARK: {json.dumps(result['benchmark'], allow_nan=False)}")
    print("RESULTS_JSON: " + json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

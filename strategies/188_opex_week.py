"""188_opex_week
WARNING: known look-ahead; numbers invalid. See DEEP_REVIEW_FINDINGS.md.
研究用途，非投资建议 / Research only, not investment advice. 进入投资系统须另经 Gatekeeper 审查。
Author: Codex; review before entry: Codex self-review (not an independent review); publication gate: Codex.
Run (one command): python strategies/188_opex_week.py (free public data only; config embedded)
Backtest reproduced: run of this file's code at commit 92e75bea82090bd8b6723b610a02dabec59e659b on 2026-11-27 07:29 Europe/Oslo, exit 0, output sha256 fa543a3c5385d2a3f53bb4bc745651c3b872589d1342b792dabbf43d47155c13
Data source: Yahoo Finance 免费公开日线，经 yfinance 获取 SPY（权益 ETF）和 BIL（现金代理 ETF）的复权开盘价及收盘价。优先读取本地 CSV 快照，缓存缺失或校验失败时请求 Yahoo Finance。
Data fetched: 2026-11-27
Sample period: 配置覆盖 2008-01-01 至 2026-10-31，含起止日期。指定运行的实际交易日覆盖 2008-01-02 至 2026-10-30。 data_range (from output): 2008-01-02 to 2026-10-30
IS/OOS split: 按时间划分，样本内 2008-01-01 至 2017-12-31，样本外 2018-01-01 至 2026-10-31，均含起止日期。组合全程模拟一次，再切片报告，跨界持仓连续。样本外区间与原文历史覆盖期重叠。
Costs and slippage: 每条 ETF 买卖腿计固定费用 1 bp 和固定滑点 1 bp，合计 2 bps。日增长因子乘以 (1 - 0.0002)^当日成交腿数。每次换仓计两腿，初始直接买入计一腿，SPY 买入持有基准仅计初始一腿。配对周收益为毛收益。滑点模型为固定基点扣减，未模拟随规模、流动性或到期日变化的滑点。
Signal lag: 信号由交易日历预排，最迟于前一交易日 t 收盘确定。到期周首个交易日开盘由 BIL 换入 SPY，到期周最后一个交易日收盘换回 BIL。对每次切换而言，成交分别采用 t+1 交易日的开盘价或收盘价。
Price adjustment: Yahoo Finance auto_adjust=True (split- and distribution-adjusted Open and Close)
Survivorship bias: 固定使用样本期内已上市的 SPY 和 BIL，无个股成分表或以现今指数成分股回填历史。两只 ETF 为事后选定的存续基金，样本起点受 BIL 上市时间限制。
Parameter combinations tried: 1
Key numbers (copied verbatim from that run's RESULTS_JSON):
    is: cagr=0.015636341477718085, ann_vol=0.09406205509400564, sharpe=0.2123662107619181, sharpe_excess=0.18779511246727484, max_drawdown=-0.24411796724775647, total_return=0.1676934548790161, turnover=48.13820492454329, n_trades=481
    oos: cagr=0.003012210308401153, ann_vol=0.07436339337060842, sharpe=0.07773528499037947, sharpe_excess=-0.26367712265119353, max_drawdown=-0.2530938481420447, total_return=0.02685039892022112, turnover=48.12972972972973, n_trades=424
    full: cagr=0.009701616162450577, ann_vol=0.0853922255461183, sharpe=0.1560391651359337, sharpe_excess=0.0031622959421207055, max_drawdown=-0.2530938481420453, total_return=0.1990464899590505, turnover=48.13423385394681, n_trades=905
    benchmark: cagr=0.11117437644667372, ann_vol=0.19695562343836195, sharpe=0.6338740983048184, sharpe_excess=0.5658568809392597, max_drawdown=-0.5240876392850615, total_return=6.257355038288677, turnover=0.05318699873364289, n_trades=1
"""
# === CODE START ===
# --- single-file shim: the strategy's config.yaml is embedded here ---
import builtins as _b, io as _io, os as _os
_CONFIG_YAML = "strategy: 188_opex_week\ncategory: seasonality\n\nuniverse:\n  equity: SPY\n  cash_proxy: BIL\n\n# Inclusive dates. The portfolio runs continuously across the split.\nin_sample:\n  start: '2008-01-01'\n  end: '2017-12-31'\nout_of_sample:\n  start: '2018-01-01'\n  end: '2026-10-31'\n\nparameters:\n  match_lookback_weeks: 12\n\ncosts:\n  # Charged on every ETF buy or sell, including both legs of each rotation.\n  bps_per_side: 1.0\n  slippage_bps_per_side: 1.0\n"
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
"""Monthly expiration-week rotation and matched-week comparison."""

import json
from datetime import datetime, timezone
from pathlib import Path
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
import yaml
import yfinance as yf
from yfinance.exceptions import YFException


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data_cache'
SESSIONS_PER_YEAR = 252
PRICE_COLUMNS = ['Open', 'Close']


class NyseHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, start_date='2022-06-01', observance=nearest_workday),
        Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
    ]


# Additional full-market closures in the supported history, starting in 2008.
SPECIAL_CLOSURES = pd.to_datetime(['2012-10-29', '2012-10-30', '2018-12-05', '2025-01-09'])


def session_calendar(start, end):
    # Complete the edge weeks so a partial reporting window cannot change an expiration.
    first_monday = start - pd.Timedelta(days=start.weekday())
    last_friday = end.to_period('W-FRI').end_time.normalize()
    weekdays = pd.bdate_range(first_monday, last_friday)
    holidays = NyseHolidays().holidays(first_monday, last_friday)
    return weekdays.difference(holidays.union(SPECIAL_CLOSURES))


def trading_weeks(calendar):
    grouped = pd.Series(calendar, index=calendar).groupby(calendar.to_period('W-FRI'))
    weeks = grouped.agg(first='min', last='max', n_sessions='size')
    friday = weeks.index.end_time.normalize()
    weeks['expiration'] = (friday.day >= 15) & (friday.day <= 21)
    weeks['quarterly'] = friday.month.isin([3, 6, 9, 12])
    return weeks


def checked_prices(prices, sessions, ticker):
    prices = prices[PRICE_COLUMNS].sort_index()
    if not prices.index.is_unique:
        raise ValueError(f'{ticker} has duplicate daily bars')
    missing = sessions.difference(prices.index)
    unexpected = prices.index.difference(sessions)
    if len(missing) or len(unexpected):
        raise ValueError(f'{ticker} calendar mismatch (missing {list(missing[:5].date)}, '
                         f'unexpected {list(unexpected[:5].date)})')
    values = prices.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f'{ticker} has invalid adjusted opens or closes')
    return prices.astype(float)


def load_prices(ticker, start, end, sessions):
    path = CACHE / f'{ticker}_{start.date()}_{end.date()}_adjusted.csv'
    if path.exists():
        try:
            cached = pd.read_csv(path, index_col='Date', parse_dates=['Date'])
            prices = checked_prices(cached, sessions, ticker)
            # The original snapshots have no request timestamp. Report their
            # cache-write date, which is also recorded on fresh downloads.
            fetched_on = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date()
            return prices, fetched_on
        except (KeyError, ValueError) as exc:
            print(f'Refreshing invalid cache for {ticker} ({exc})', flush=True)

    yf.set_tz_cache_location(str(CACHE / 'yfinance'))
    for attempt in range(1, 4):
        try:
            raw = yf.Ticker(ticker).history(
                start=start.date().isoformat(),
                end=(end + pd.Timedelta(days=1)).date().isoformat(),
                interval='1d', auto_adjust=True, actions=False, timeout=15,
            )
            raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
            prices = checked_prices(raw, sessions, ticker)
            fetched_on = datetime.now(timezone.utc).date()
            break
        except (OSError, KeyError, ValueError, YFException) as exc:
            if attempt == 3:
                raise RuntimeError(f'Unable to load complete daily prices for {ticker}') from exc
            print(f'Retrying Yahoo Finance for {ticker} ({exc})', flush=True)

    # Atomic replacement keeps concurrent runs from reading half a download.
    with NamedTemporaryFile(mode='w', dir=CACHE, suffix='.csv', delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            prices.to_csv(temp, index_label='Date')
            temp.flush()
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)
    return prices, fetched_on


def simulate(equity, cash_proxy, weeks, cost):
    sessions = equity.index
    equity_day = pd.Series(False, index=sessions)
    enter = equity_day.copy()
    exit_ = equity_day.copy()
    expiration_weeks = weeks[weeks['expiration'] & weeks['first'].isin(sessions)]
    for week in expiration_weeks.itertuples():
        equity_day.loc[week.first:week.last] = True
        enter.loc[week.first] = True
        if week.last in sessions:
            exit_.loc[week.last] = True

    # Both switches are scheduled by the prior session's close, using the
    # exchange calendar alone. The old holding earns the overnight return.
    equity_night = (equity_day & ~exit_).shift(fill_value=False)
    equity_overnight = equity['Open'] / equity['Close'].shift()
    cash_overnight = cash_proxy['Open'] / cash_proxy['Close'].shift()
    overnight = cash_overnight.where(~equity_night, equity_overnight)
    overnight.iloc[0] = equity_overnight.iloc[0] = cash_overnight.iloc[0] = 1.0
    equity_intraday = equity['Close'] / equity['Open']
    cash_intraday = cash_proxy['Close'] / cash_proxy['Open']
    intraday = cash_intraday.where(~equity_day, equity_intraday)

    legs = 2 * (enter.astype(int) + exit_.astype(int))
    legs.iloc[0] = 1 + 2 * int(exit_.iloc[0])  # First open buys the target directly.
    benchmark_legs = pd.Series(0, index=sessions)
    benchmark_legs.iloc[0] = 1
    strategy_growth = overnight * intraday * (1.0 - cost) ** legs
    benchmark_growth = equity_overnight * equity_intraday * (1.0 - cost) ** benchmark_legs
    cash_growth = cash_overnight * cash_intraday
    return strategy_growth, legs, benchmark_growth, benchmark_legs, cash_growth


def annualized_sharpe(returns, daily_vol):
    if daily_vol > 0 and returns.min() < returns.max():
        return returns.mean() / daily_vol * np.sqrt(SESSIONS_PER_YEAR)
    return None


def metrics(growth, legs, cash_growth):
    nav = np.r_[1.0, growth.cumprod().to_numpy()]
    returns = growth - 1.0
    excess_returns = returns - (cash_growth - 1.0)
    daily_vol = returns.std(ddof=1)
    n_trades = int(legs.sum())
    return {
        'cagr': nav[-1] ** (SESSIONS_PER_YEAR / len(growth)) - 1.0,
        'ann_vol': daily_vol * np.sqrt(SESSIONS_PER_YEAR),
        'sharpe': annualized_sharpe(returns, daily_vol),
        'sharpe_excess': annualized_sharpe(excess_returns, excess_returns.std(ddof=1)),
        'max_drawdown': np.min(nav / np.maximum.accumulate(nav) - 1.0),
        'total_return': nav[-1] - 1.0,
        'turnover': n_trades * SESSIONS_PER_YEAR / len(growth),
        'n_trades': n_trades,
    }


def matched_weeks(equity, weeks, start, end, lookback):
    slice_weeks = weeks[weeks['first'].between(start, end) & weeks['last'].between(start, end)]
    events = slice_weeks[slice_weeks['expiration']]
    ordinary = slice_weeks[~slice_weeks['expiration']]

    def weekly_return(first, last):
        return equity.at[last, 'Close'] / equity.at[first, 'Open'] - 1.0

    rows = []
    for event in events.itertuples():
        controls = ordinary.loc[event.Index - lookback:event.Index - 1]
        controls = controls[controls['n_sessions'] == event.n_sessions]
        if controls.empty:
            continue
        control = controls.iloc[-1]
        rows.append((event.quarterly, control.name,
                     weekly_return(event.first, event.last),
                     weekly_return(control['first'], control['last'])))

    pairs = pd.DataFrame(rows, columns=['quarterly', 'control', 'event_return', 'control_return'])
    quarterly = pairs['quarterly'].eq(True)
    results = {}
    for name, group, event_count in (
        ('all', pairs, len(events)),
        ('quarterly', pairs[quarterly], int(events['quarterly'].sum())),
        ('non_quarterly', pairs[~quarterly], int((~events['quarterly']).sum())),
    ):
        results[name] = {
            'n_events': event_count,
            'n_pairs': len(group),
            'n_distinct_controls': group['control'].nunique(),
            'opex_avg_bps': 10_000 * group['event_return'].mean(),
            'control_avg_bps': 10_000 * group['control_return'].mean(),
            'spread_bps': 10_000 * (group['event_return'] - group['control_return']).mean(),
        }
    return results


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def main():
    with (ROOT / 'config.yaml').open(encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    in_sample, out_of_sample = (
        tuple(pd.Timestamp(config[name][key]) for key in ('start', 'end'))
        for name in ('in_sample', 'out_of_sample')
    )
    start = in_sample[0]
    end = out_of_sample[1]
    if not pd.Timestamp('2008-01-01') <= start <= in_sample[1] < out_of_sample[0] <= end:
        raise ValueError('Use ordered, non-overlapping IS/OOS ranges starting in 2008 or later')

    lookback = config['parameters']['match_lookback_weeks']
    fee_bps = float(config['costs']['bps_per_side'])
    slippage_bps = float(config['costs'].get('slippage_bps_per_side', 0.0))
    cost = (fee_bps + slippage_bps) / 10_000
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError('Lookback must be a positive integer')
    if not (fee_bps >= 0 and slippage_bps >= 0 and cost < 1):
        raise ValueError('Fees and slippage must be non-negative and total less than 10000 bps')

    calendar = session_calendar(start, end)
    sessions = calendar[(calendar >= start) & (calendar <= end)]
    periods = {'is': in_sample, 'oos': out_of_sample, 'full': (start, end)}
    if any(sessions[(sessions >= left) & (sessions <= right)].empty for left, right in periods.values()):
        raise ValueError('Each reporting range must contain at least one trading session')

    CACHE.mkdir(exist_ok=True)
    equity_ticker = config['universe']['equity']
    cash_ticker = config['universe']['cash_proxy']
    equity, equity_fetched = load_prices(equity_ticker, start, end, sessions)
    cash_proxy, cash_fetched = load_prices(cash_ticker, start, end, sessions)
    weeks = trading_weeks(calendar)
    strategy_growth, legs, benchmark_growth, benchmark_legs, cash_growth = simulate(
        equity, cash_proxy, weeks, cost
    )

    strategy = {}
    benchmark = {}
    comparisons = {}
    for label, (left, right) in periods.items():
        period_cash = cash_growth.loc[left:right]
        strategy[label] = metrics(strategy_growth.loc[left:right], legs.loc[left:right], period_cash)
        benchmark[label] = metrics(benchmark_growth.loc[left:right], benchmark_legs.loc[left:right], period_cash)
        comparisons[label] = matched_weeks(equity, weeks, left, right, lookback)

    benchmark_name = f'{equity_ticker} buy and hold'
    result = {
        'strategy': config['strategy'],
        'data_range': f'{sessions[0].date()} to {sessions[-1].date()}',
        **strategy,
        'benchmark': {'name': benchmark_name, **benchmark['full'], **benchmark},
        'matched_weeks': comparisons,
    }

    print(f"{result['strategy']} | {result['data_range']} | "
          f"{fee_bps:g} bps fees + {slippage_bps:g} bps slippage per ETF trade")
    fetched_at = (str(equity_fetched) if equity_fetched == cash_fetched else
                  f'{equity_ticker}={equity_fetched} {cash_ticker}={cash_fetched}')
    print('PARAM_COMBOS_TRIED: 1')
    print(f'DATA_FETCHED_AT: {fetched_at}')
    print('PRICE_ADJUSTMENT: Yahoo Finance auto_adjust=True (split- and distribution-adjusted Open and Close)')
    print(f'Cash reference for excess Sharpe: {cash_ticker} gross total return')
    print(f'IS {in_sample[0].date()} to {in_sample[1].date()} | '
          f'OOS {out_of_sample[0].date()} to {out_of_sample[1].date()}')
    portfolio_table = pd.concat({'Strategy': pd.DataFrame(strategy).T,
                                 benchmark_name: pd.DataFrame(benchmark).T})
    percentage = {
        key: '{:.2%}'.format for key in ('cagr', 'ann_vol', 'max_drawdown', 'total_return', 'turnover')
    }
    print(portfolio_table.to_string(na_rep='N/A', formatters={**percentage, 'n_trades': '{:.0f}'.format},
                                    float_format='%.3f'))
    print(f'\nMatched-week comparison (gross {equity_ticker} open-to-close returns in bps)')
    comparison_table = pd.concat({name: pd.DataFrame(groups).T for name, groups in comparisons.items()})
    count_format = {key: '{:.0f}'.format for key in ('n_events', 'n_pairs', 'n_distinct_controls')}
    print(comparison_table.to_string(na_rep='N/A', formatters=count_format, float_format='%.2f'))
    print('RESULTS_JSON: ' + json.dumps(json_safe(result), allow_nan=False))


if __name__ == '__main__':
    main()
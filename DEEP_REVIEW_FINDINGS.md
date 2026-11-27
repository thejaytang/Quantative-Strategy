# Deep-review findings (already published)

Already-published strategies are not removed from git history. If a later deep review finds a real problem (look-ahead, wrong data, or wrong numbers), the file is kept as last published on origin/main and listed here in a new commit. Fixes, if any, also land in a new commit.

Generated 2026-11-27 10:20 Europe/Oslo.

| File | Latest self-review | Category | Stage note |
|---|---|---|---|
| strategies/323_information_transmission_across_tail.py | 待修 | see review | kept_from_origin: latest self-review is not PASS |
| strategies/179_crude_oil_news_sentiment.py | 待修 | see review | kept_from_origin: latest self-review is not PASS |
| strategies/106_low_ivol_low_max.py | FAIL | 成交数量依赖事后复权开盘价; 固定股票池有幸存者偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/382_statistical_arbitrage_in_polish_equities.py | 待修 | 成交日最终收盘价参与确定开仓股数; 股票池以事后可用数据构造，样本外仍继承筛选偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/113_kalman_garch_pairs.py | FAIL | 成交日收盘价参与定仓; 成本只按固定费率扣除，成交价无滑点 | kept_from_origin: latest self-review is not PASS |
| strategies/315_leveraged_funds_robust_replication.py | 待修 | see review | kept_from_origin: latest self-review is not PASS |
| strategies/107_illiquidity_premium_amihud.py | FAIL | 成交数量规则不通过; 股票池存在幸存者和事后选样偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/309_intangible_value_sixth_factor.py | 待修 | 实际买卖量使用执行日最终收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/130_ml_risk_budget_allocation.py | 待修 | see review | kept_from_origin: latest self-review is not PASS |
| strategies/438_the_mathematics_of_heuristic_portfolio_o.py | 待修 | 成交规模依赖成交日最终收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/463_simple_dynamic_stock_bond_gold_portfolio.py | 待修 | 成交金额依赖成交当日最终收盘 | kept_from_origin: latest self-review is not PASS |

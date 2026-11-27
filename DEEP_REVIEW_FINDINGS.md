# Deep-review findings (already published)

Already-published strategies are not removed from git history. If a later deep review finds a real problem (look-ahead, wrong data, or wrong numbers), the file is kept as last published on origin/main and listed here in a new commit. Fixes, if any, also land in a new commit.

Generated 2026-11-27 10:35 Europe/Oslo.

| File | Latest self-review | Category | Stage note |
|---|---|---|---|
| strategies/501_nested_clustered_optimization_is_one_end.py | FAIL | 成交数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/451_a_declining_cvar_glidepath_framework_for.py | FAIL | 成交量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/323_information_transmission_across_tail.py | 待修 | see review | kept_from_origin: latest self-review is not PASS |
| strategies/452_simple_dynamic_stock_bond_gold_portfolio.py | FAIL | 按成交日收盘反推成交规模; 成交日历依赖行情行 | kept_from_origin: latest self-review is not PASS |
| strategies/484_nonlinear_shrinkage_for_higher_moments_e.py | FAIL | 成交数量前视; 固定股票池的幸存者与事后选样偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/476_conditional_value_at_risk_under_reward_p.py | FAIL | 成交量前视，触发 FAIL | kept_from_origin: latest self-review is not PASS |
| strategies/449_the_mathematics_of_heuristic_portfolio_o.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/483_wasserstein_penalisation_as_an_endogenou.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/431_mitigating_return_drag_and_regime_whipsa.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/534_portfolio_diversification_and_concentrat.py | FAIL | 成交名义金额依赖成交日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/179_crude_oil_news_sentiment.py | 待修 | see review | kept_from_origin: latest self-review is not PASS |
| strategies/450_a_declining_cvar_glidepath_framework_for.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/481_optimal_portfolio_construction_a_reinfor.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/500_dynamic_portfolio_optimization_under_cva.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/106_low_ivol_low_max.py | FAIL | 成交数量依赖事后复权开盘价; 固定股票池有幸存者偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/338_optimal_portfolio_using_factor.py | FAIL | 开盘定量间接读取成交日收盘字段 | kept_from_origin: latest self-review is not PASS |
| strategies/465_semivariance_and_semivolatility_in_portf.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/113_kalman_garch_pairs.py | FAIL | 成交日收盘价参与定仓; 成本只按固定费率扣除，成交价无滑点 | kept_from_origin: latest self-review is not PASS |
| strategies/315_leveraged_funds_robust_replication.py | FAIL | 成交数量使用成交日收盘价; 成交摩擦只按固定费率建模 | kept_from_origin: latest self-review is not PASS |
| strategies/480_topological_risk_parity.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/472_breaking_the_trend_how_to_avoid_cherry_p.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/467_an_entropic_factor_model_for_robust_port.py | FAIL | 成交数量依赖成交日收盘; 股票池带有幸存者偏差和事后筛选偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/478_portfolio_allocation_under_heterogeneous.py | FAIL | 成交量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/443_revisiting_equity_bond_diversification_u.py | FAIL | 成交数量依赖成交日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/109_factor_etf_vol_matched.py | FAIL | 成交数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/490_noise_proofing_universal_portfolio_shrin.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/511_dynamic_portfolio_optimization_under_cva.py | FAIL | 成交日收盘价参与确定同一收盘的交易量 | kept_from_origin: latest self-review is not PASS |
| strategies/400_noxtrader_lstm_based_stock_return_moment.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/492_optimal_portfolio_construction_a_reinfor.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/107_illiquidity_premium_amihud.py | FAIL | 成交数量规则不通过; 股票池存在幸存者和事后选样偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/309_intangible_value_sixth_factor.py | FAIL | 以成交日收盘价倒算成交量; 成本有披露，成交摩擦仅由固定费率近似 | kept_from_origin: latest self-review is not PASS |
| strategies/130_ml_risk_budget_allocation.py | 待修 | see review | kept_from_origin: latest self-review is not PASS |
| strategies/438_the_mathematics_of_heuristic_portfolio_o.py | FAIL | 成交日收盘价决定了实际持仓份额; 成交比说明晚一个交易日; 费率有披露，费用与换手使用前一收盘的估计名义额 | kept_from_origin: latest self-review is not PASS |
| strategies/468_cleaning_correlation_matrices_with_exter.py | FAIL | 成交数量依赖成交日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/324_volatility_conditions_weekend_effect.py | FAIL | 开仓数量隐含使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/455_hierarchical_risk_parity_for_portfolio_a.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/469_two_sides_of_schur_damping_high_dimensio.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/512_nested_clustered_optimization_is_one_end.py | FAIL | 用成交日收盘价反推同一收盘的成交规模; 全市场缺行可能改变信号及成交日期 | kept_from_origin: latest self-review is not PASS |
| strategies/316_hidden_cost_etf_investing.py | FAIL | 收盘建仓和调仓数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/434_dynamic_scenario_based_portfolio_optimiz.py | FAIL | 调仓数量使用成交日收盘价; 固定 ETF 名单由事后公开的论文提供 | kept_from_origin: latest self-review is not PASS |
| strategies/470_entropic_value_at_risk_parity_for_temper.py | FAIL | look-ahead / same-close fill | kept_from_origin: latest self-review is not PASS |
| strategies/463_simple_dynamic_stock_bond_gold_portfolio.py | FAIL | 实际持仓数量依赖成交日收盘价; 实际成交比文档多等一个交易日 | kept_from_origin: latest self-review is not PASS |
| strategies/479_noise_proofing_universal_portfolio_shrin.py | FAIL | 成交日收盘价反推成交数量和名义金额 | kept_from_origin: latest self-review is not PASS |

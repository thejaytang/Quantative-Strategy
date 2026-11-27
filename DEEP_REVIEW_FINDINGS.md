# Deep-review findings (already published)

Already-published strategies are not removed from git history. If a later deep review finds a real problem (look-ahead, wrong data, or wrong numbers), the file is kept as last published on origin/main and listed here in a new commit. Fixes, if any, also land in a new commit.

Generated 2026-11-27 10:52 Europe/Oslo.

| File | Latest self-review | Category | Stage note |
|---|---|---|---|
| strategies/547_generalizing_markowitz_portfolio_optimiz.py | FAIL | 成交数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/544_special_markowitz_thermodynamic_formalis.py | FAIL | 成交量依赖执行日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/501_nested_clustered_optimization_is_one_end.py | FAIL | 成交数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/451_a_declining_cvar_glidepath_framework_for.py | FAIL | 成交量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/323_information_transmission_across_tail.py | FAIL | 成交金额间接依赖成交日收盘字段 | kept_from_origin: latest self-review is not PASS |
| strategies/510_some_general_results_on_risk_budgeting_p.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/452_simple_dynamic_stock_bond_gold_portfolio.py | FAIL | 按成交日收盘反推成交规模; 成交日历依赖行情行 | kept_from_origin: latest self-review is not PASS |
| strategies/494_wasserstein_penalisation_as_an_endogenou.py | FAIL | 成交数量使用成交日收盘信息，构成入库阻断 | kept_from_origin: latest self-review is not PASS |
| strategies/533_special_markowitz_thermodynamic_formalis.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/514_risk_limited_asset_allocation_with_a_bud.py | FAIL | 成交额和目标头寸依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/484_nonlinear_shrinkage_for_higher_moments_e.py | FAIL | 成交数量前视; 固定股票池的幸存者与事后选样偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/476_conditional_value_at_risk_under_reward_p.py | FAIL | 成交量前视，触发 FAIL | kept_from_origin: latest self-review is not PASS |
| strategies/432_revisiting_equity_bond_diversification_u.py | FAIL | 成交日收盘参与反推成交数量 | kept_from_origin: latest self-review is not PASS |
| strategies/505_competitive_optimal_portfolio_selection.py | FAIL | 成交日收盘价参与反算同一收盘的成交量 | kept_from_origin: latest self-review is not PASS |
| strategies/449_the_mathematics_of_heuristic_portfolio_o.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/483_wasserstein_penalisation_as_an_endogenou.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/503_risk_limited_asset_allocation_with_a_bud.py | FAIL | 成交额使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/431_mitigating_return_drag_and_regime_whipsa.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/532_robust_enhanced_index_tracking_portfolio.py | FAIL | 成交规模前视; 股票池存在幸存者偏差与事后选样 | kept_from_origin: latest self-review is not PASS |
| strategies/194_breaking_beta_defensive.py | FAIL | 成交数量依赖成交日收盘字段; 开盘成交与滑点假设 | kept_from_origin: latest self-review is not PASS |
| strategies/534_portfolio_diversification_and_concentrat.py | FAIL | 成交名义金额依赖成交日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/101_correlated_stress_reversal.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/179_crude_oil_news_sentiment.py | PASS | see review | kept_from_origin: latest self-review is not PASS |
| strategies/195_century_reversals_volatility.py | FAIL | 收盘买回 BIL 的数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/509_beyond_de_prado_and_cotton_hierarchical.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/520_beyond_de_prado_and_cotton_hierarchical.py | FAIL | 成交数量依赖成交日收盘价（`strategy.py:149–166`） | kept_from_origin: latest self-review is not PASS |
| strategies/548_entropic_value_at_risk_portfolio_optimiz.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/395_signature_based_optimal_execution_for_st.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/450_a_declining_cvar_glidepath_framework_for.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/481_optimal_portfolio_construction_a_reinfor.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/500_dynamic_portfolio_optimization_under_cva.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/541_entropy_regularization_under_no_shorting.py | FAIL | 成交数量使用执行日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/459_entropic_value_at_risk_parity_for_temper.py | FAIL | look-ahead / same-close fill | kept_from_origin: latest self-review is not PASS |
| strategies/497_interpretable_allocation_and_model_predi.py | FAIL | 开盘成交依赖成交日收盘字段; 候选池事后选定，存在幸存者偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/124_dollar_safe_haven_reversal.py | FAIL | 第五个持有日收盘买回 BIL 的数量依赖当日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/106_low_ivol_low_max.py | FAIL | 成交数量依赖事后复权开盘价; 固定股票池有幸存者偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/442_mitigating_return_drag_and_regime_whipsa.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/338_optimal_portfolio_using_factor.py | FAIL | 开盘定量间接读取成交日收盘字段 | kept_from_origin: latest self-review is not PASS |
| strategies/465_semivariance_and_semivolatility_in_portf.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/113_kalman_garch_pairs.py | FAIL | 成交日收盘价参与定仓; 成本只按固定费率扣除，成交价无滑点 | kept_from_origin: latest self-review is not PASS |
| strategies/315_leveraged_funds_robust_replication.py | FAIL | 成交数量使用成交日收盘价; 成交摩擦只按固定费率建模 | kept_from_origin: latest self-review is not PASS |
| strategies/480_topological_risk_parity.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/341_monitoring_attention_inflation_news.py | FAIL | 成交数量使用了成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/545_portfolio_diversification_and_concentrat.py | FAIL | 成交数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/543_robust_enhanced_index_tracking_portfolio.py | FAIL | 成交日收盘价参与定量; 股票池存在已披露的幸存者和事后选择偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/461_breaking_the_trend_how_to_avoid_cherry_p.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/487_conditional_value_at_risk_under_reward_p.py | FAIL | 成交数量使用成交日收盘价; 股票池存在事后选择和幸存者偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/538_mirror_descent_algorithms_for_risk_budge.py | FAIL | 成交数量使用成交日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/472_breaking_the_trend_how_to_avoid_cherry_p.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/467_an_entropic_factor_model_for_robust_port.py | FAIL | 成交数量依赖成交日收盘; 股票池带有幸存者偏差和事后筛选偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/478_portfolio_allocation_under_heterogeneous.py | FAIL | 成交量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/491_topological_risk_parity.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/188_opex_week.py | FAIL | 收盘轮动的隐含买入数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/443_revisiting_equity_bond_diversification_u.py | FAIL | 成交数量依赖成交日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/159_deep_momentum_market_trend.py | FAIL | 开盘定量输入依赖成交日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/109_factor_etf_vol_matched.py | FAIL | 成交数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/186_federal_holiday_effect.py | FAIL | 事件日收盘买回 BIL 的份额由当日收盘价反推; 标的选择和成交假设限制解释范围 | kept_from_origin: latest self-review is not PASS |
| strategies/490_noise_proofing_universal_portfolio_shrin.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/456_an_entropic_factor_model_for_robust_port.py | FAIL | 成交数量使用执行日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/537_hierarchical_minimum_variance_portfolios.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/511_dynamic_portfolio_optimization_under_cva.py | FAIL | 成交日收盘价参与确定同一收盘的交易量 | kept_from_origin: latest self-review is not PASS |
| strategies/400_noxtrader_lstm_based_stock_return_moment.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/504_exploratory_mean_variance_with_jumps_an.py | FAIL | 成交数量使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/454_semivariance_and_semivolatility_in_portf.py | FAIL | 成交日收盘后倒算成交量 | kept_from_origin: latest self-review is not PASS |
| strategies/492_optimal_portfolio_construction_a_reinfor.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/549_drawdown_risk_beyond_brownian_motion_a_m.py | FAIL | 换仓买入金额由成交日收盘值决定 | kept_from_origin: latest self-review is not PASS |
| strategies/391_evolutionary_factor_searching_for_sparse.py | FAIL | 成交数量依赖成交日收盘价; 股票池有事后成分选择和幸存者偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/107_illiquidity_premium_amihud.py | FAIL | 成交数量规则不通过; 股票池存在幸存者和事后选样偏差 | kept_from_origin: latest self-review is not PASS |
| strategies/309_intangible_value_sixth_factor.py | FAIL | 以成交日收盘价倒算成交量; 成本有披露，成交摩擦仅由固定费率近似 | kept_from_origin: latest self-review is not PASS |
| strategies/130_ml_risk_budget_allocation.py | PASS | see review | kept_from_origin: latest self-review is not PASS |
| strategies/438_the_mathematics_of_heuristic_portfolio_o.py | 待修 | see review | kept_from_origin: latest self-review is not PASS |
| strategies/448_continuous_cash_overlay_filters_for_a_st.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/468_cleaning_correlation_matrices_with_exter.py | FAIL | 成交数量依赖成交日收盘 | kept_from_origin: latest self-review is not PASS |
| strategies/324_volatility_conditions_weekend_effect.py | FAIL | 开仓数量隐含使用成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/466_hierarchical_risk_parity_for_portfolio_a.py | FAIL | 成交量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/455_hierarchical_risk_parity_for_portfolio_a.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/460_understanding_the_long_only_minimum_vari.py | FAIL | 成交定量使用执行日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/469_two_sides_of_schur_damping_high_dimensio.py | FAIL | 成交数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/474_multiperiod_bond_portfolio_optimization.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/512_nested_clustered_optimization_is_one_end.py | FAIL | 用成交日收盘价反推同一收盘的成交规模; 全市场缺行可能改变信号及成交日期 | kept_from_origin: latest self-review is not PASS |
| strategies/197_tsmom_boundaries.py | FAIL | 开盘换仓金额依赖当日收盘字段 | kept_from_origin: latest self-review is not PASS |
| strategies/316_hidden_cost_etf_investing.py | FAIL | 收盘建仓和调仓数量依赖成交日收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/535_shrinkage_estimators_for_mean_and_covari.py | FAIL | 成交日收盘参与定量 | kept_from_origin: latest self-review is not PASS |
| strategies/310_factor_beta_overnight_intraday.py | FAIL | 收盘成交使用同一收盘价确定股数; 成本已计入，拍卖成交与做空摩擦仍是简化假设 | kept_from_origin: latest self-review is not PASS |
| strategies/353_hold_the_dip.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |
| strategies/445_dynamic_scenario_based_portfolio_optimiz.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/457_cleaning_correlation_matrices_with_exter.py | FAIL | 成交数量使用成交日收盘信息 | kept_from_origin: latest self-review is not PASS |
| strategies/482_dynamic_inclusion_and_bounded_multi_fact.py | FAIL | 调仓用成交日收盘价倒算数量 | kept_from_origin: latest self-review is not PASS |
| strategies/434_dynamic_scenario_based_portfolio_optimiz.py | FAIL | 调仓数量使用成交日收盘价; 固定 ETF 名单由事后公开的论文提供 | kept_from_origin: latest self-review is not PASS |
| strategies/528_schur_complementary_allocation_a_unifica.py | FAIL | look-ahead / same-close fill | kept_from_origin: latest self-review is not PASS |
| strategies/354_strategy.py | FAIL | 买入规模依赖成交收盘价 | kept_from_origin: latest self-review is not PASS |
| strategies/470_entropic_value_at_risk_parity_for_temper.py | FAIL | look-ahead / same-close fill | kept_from_origin: latest self-review is not PASS |
| strategies/463_simple_dynamic_stock_bond_gold_portfolio.py | 待修 | look-ahead / same-close fill | kept_from_origin: latest self-review is not PASS |
| strategies/479_noise_proofing_universal_portfolio_shrin.py | FAIL | 成交日收盘价反推成交数量和名义金额 | kept_from_origin: latest self-review is not PASS |
| strategies/156_cta_replication_trend_factors.py | FAIL | see review | kept_from_origin: latest self-review is not PASS |

# Deep-review findings (already published)

Already-published strategies are not removed from git history. If a later deep review finds a real problem (look-ahead, wrong data, or wrong numbers), the file is kept as last published on origin/main and listed here in a new commit. Fixes, if any, also land in a new commit.

Generated 2026-11-27 12:28 Europe/Oslo.

| File | Latest self-review | Category | Stage note |
|---|---|---|---|
| strategies/022_sector_residual_statarb.py | FAIL | 结算数量仍可由当日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/101_correlated_stress_reversal.py | PASS | 成交与资金可用性依赖日线假设 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/106_low_ivol_low_max.py | PASS | 固定存续股票池 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/109_factor_etf_vol_matched.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/113_kalman_garch_pairs.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/124_dollar_safe_haven_reversal.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/156_cta_replication_trend_factors.py | PASS | 成交和融资成本为简化假设 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/159_deep_momentum_market_trend.py | PASS | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/179_crude_oil_news_sentiment.py | PASS | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/186_federal_holiday_effect.py | PASS | 成交成本是简化假设 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/188_opex_week.py | FAIL | 周末收盘买入的实际份额由当日收盘价决定; 固定成本和日线成交假设限制执行结论 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/195_century_reversals_volatility.py | PASS | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/197_tsmom_boundaries.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/310_factor_beta_overnight_intraday.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/315_leveraged_funds_robust_replication.py | FAIL | 资金不足时用成交日收盘价改变实际成交份额 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/324_volatility_conditions_weekend_effect.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/338_optimal_portfolio_using_factor.py | PASS | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/341_monitoring_attention_inflation_news.py | FAIL | 现金不足时，实际买入股数取决于成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/353_hold_the_dip.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/354_strategy.py | PASS | 成交和资金假设限制结果的解释 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/395_signature_based_optimal_execution_for_st.py | FAIL | 资金约束在同一收盘价重定成交股数 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/400_noxtrader_lstm_based_stock_return_moment.py | PASS | 幸存者与事后选样偏差 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/431_mitigating_return_drag_and_regime_whipsa.py | FAIL | 实际买入份额会按成交收盘价缩减 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/432_revisiting_equity_bond_diversification_u.py | FAIL | 共用模拟器在成交收盘重新确定买入份额; 收盘模型使用事后复权快照与简化费用 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/434_dynamic_scenario_based_portfolio_optimiz.py | PASS | 事后资产池与回顾性样本外; 模拟成交和成本的适用范围 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/442_mitigating_return_drag_and_regime_whipsa.py | FAIL | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/443_revisiting_equity_bond_diversification_u.py | FAIL | 最终买入量可由成交日收盘价决定; 动态配置缺少实际成交证据 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/445_dynamic_scenario_based_portfolio_optimiz.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/449_the_mathematics_of_heuristic_portfolio_o.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/450_a_declining_cvar_glidepath_framework_for.py | FAIL | 成交日收盘价改变实际买入份额 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/451_a_declining_cvar_glidepath_framework_for.py | FAIL | 资金不足时，最终买入份额由成交日收盘价反算 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/452_simple_dynamic_stock_bond_gold_portfolio.py | FAIL | 现金不足时，成交日收盘价决定实际买入份额 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/454_semivariance_and_semivolatility_in_portf.py | FAIL | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/455_hierarchical_risk_parity_for_portfolio_a.py | PASS | 复权价格和现金处理简化了成交 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/456_an_entropic_factor_model_for_robust_port.py | PASS | 成交模型对流动性和现金冲击简化 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/457_cleaning_correlation_matrices_with_exter.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/458_two_sides_of_schur_damping_high_dimensio.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/459_entropic_value_at_risk_parity_for_temper.py | FAIL | 实际买入份额由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/460_understanding_the_long_only_minimum_vari.py | PASS | 复权与成交模型的限制 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/461_breaking_the_trend_how_to_avoid_cherry_p.py | FAIL | 成交日收盘价决定了实际成交数量; 资金与成交模型只覆盖日频 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/465_semivariance_and_semivolatility_in_portf.py | FAIL | 复权快照限制成交与数据时点的可追溯性 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/467_an_entropic_factor_model_for_robust_port.py | FAIL | 实际买入量依赖成交收盘价; 固定股票池存在幸存者与事后筛选偏差; 复权成交与资金约束是简化模型 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/468_cleaning_correlation_matrices_with_exter.py | PASS | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/469_two_sides_of_schur_damping_high_dimensio.py | PASS | 资产池存在事后存续筛选; 成交模型简化了现金与流动性 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/470_entropic_value_at_risk_parity_for_temper.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/472_breaking_the_trend_how_to_avoid_cherry_p.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/474_multiperiod_bond_portfolio_optimization.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/476_conditional_value_at_risk_under_reward_p.py | FAIL | 成交日收盘价参与决定实际买入数量 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/478_portfolio_allocation_under_heterogeneous.py | PASS | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/479_noise_proofing_universal_portfolio_shrin.py | PASS | 成交与资金可用性依赖模拟假设 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/480_topological_risk_parity.py | FAIL | 实际买入份额可由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/481_optimal_portfolio_construction_a_reinfor.py | FAIL | 现金不足时以成交日收盘价决定实际买量 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/482_dynamic_inclusion_and_bounded_multi_fact.py | PASS | 固定候选池存在幸存者偏差及事后选择偏差; 成交及现金约束是简化模型 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/483_wasserstein_penalisation_as_an_endogenou.py | FAIL | 最终买入份额由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/484_nonlinear_shrinkage_for_higher_moments_e.py | FAIL | 实际买入份额可由成交日收盘价决定; 股票池是事后固定的幸存者样本 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/487_conditional_value_at_risk_under_reward_p.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/490_noise_proofing_universal_portfolio_shrin.py | FAIL | 成交日收盘价决定实际买入量 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/491_topological_risk_parity.py | FAIL | 成交数量依赖成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/494_wasserstein_penalisation_as_an_endogenou.py | FAIL | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/497_interpretable_allocation_and_model_predi.py | PASS | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/500_dynamic_portfolio_optimization_under_cva.py | FAIL | 收盘价参与确定同一收盘的实际买量 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/501_nested_clustered_optimization_is_one_end.py | FAIL | 收盘价决定实际买入份额 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/503_risk_limited_asset_allocation_with_a_bud.py | FAIL | 收盘后重定买入份额 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/505_competitive_optimal_portfolio_selection.py | FAIL | 成交日收盘价决定部分实际买入份额 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/509_beyond_de_prado_and_cotton_hierarchical.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/510_some_general_results_on_risk_budgeting_p.py | FAIL | 实际买入数量由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/514_risk_limited_asset_allocation_with_a_bud.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/520_beyond_de_prado_and_cotton_hierarchical.py | FAIL | 实际买入份额依赖成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/522_asset_allocation_using_a_markov_process.py | FAIL | 实际买入份额取决于成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/528_schur_complementary_allocation_a_unifica.py | PASS | 复权收盘成交与线性成本是近似 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/532_robust_enhanced_index_tracking_portfolio.py | FAIL | 资金不足时用成交日收盘价改写买入股数; 事后固定选池造成幸存者偏差和成分选择偏差 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/533_special_markowitz_thermodynamic_formalis.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/534_portfolio_diversification_and_concentrat.py | FAIL | 最终买入数量依赖成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/535_shrinkage_estimators_for_mean_and_covari.py | PASS | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/537_hierarchical_minimum_variance_portfolios.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/538_mirror_descent_algorithms_for_risk_budge.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/539_schur_complementary_allocation_a_unifica.py | FAIL | 现金不足时用成交日收盘价重算买入份额 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/541_entropy_regularization_under_no_shorting.py | FAIL | 最终买入份额使用成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/543_robust_enhanced_index_tracking_portfolio.py | FAIL | 最终买入量依赖成交日收盘价; 固定股票池存在事后选股与幸存者偏差 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/544_special_markowitz_thermodynamic_formalis.py | FAIL | 成交日收盘价决定部分实际买入份额 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/545_portfolio_diversification_and_concentrat.py | FAIL | 实际买入份额依赖成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/546_generalizing_markowitz_portfolio_optimiz.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/547_generalizing_markowitz_portfolio_optimiz.py | FAIL | 最终买入数量可由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/548_entropic_value_at_risk_portfolio_optimiz.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/551_squeezed_covariance_matrix_estimation_an.py | FAIL | 成交时用收盘价缩减买量; 事后固定股票池 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/552_a_novel_robust_mixed_integer_linear_prog.py | FAIL | 现金不足时按成交日收盘改写实际买入股数; 固定候选池存在幸存者偏差及事后选样; 模拟成交与实际市场价格、容量和公司行动有差距 | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/553_large_scale_portfolio_optimization_probl.py | FAIL | see review | restored_from_origin: kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |

# Deep-review findings (already published)

The listed strategies remain in the repository with warnings in their file headers. Any fixes land in new commits.

Generated 2026-11-27 16:46 Europe/Oslo. Single defect list synced with the README deep-review table. Backtest numbers for every strategy listed here are invalid under the Gatekeeper ruling.

| File | Backtest status | Category | Stage note |
|---|---|---|---|
| strategies/022_sector_residual_statarb.py | INVALID | 结算数量仍可由当日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/109_factor_etf_vol_matched.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/113_kalman_garch_pairs.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/156_cta_replication_trend_factors.py | INVALID | 成交和融资成本为简化假设 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/159_deep_momentum_market_trend.py | INVALID | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/186_federal_holiday_effect.py | INVALID | 成交成本是简化假设 | labelled per Gatekeeper ruling 2026-11-27 12:33 Europe/Oslo |
| strategies/188_opex_week.py | INVALID | 周末收盘买入的实际份额由当日收盘价决定; 固定成本和日线成交假设限制执行结论 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/197_tsmom_boundaries.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/310_factor_beta_overnight_intraday.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/315_leveraged_funds_robust_replication.py | INVALID | 资金不足时用成交日收盘价改变实际成交份额 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/324_volatility_conditions_weekend_effect.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/341_monitoring_attention_inflation_news.py | INVALID | 现金不足时，实际买入股数取决于成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/353_hold_the_dip.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/354_strategy.py | INVALID | 成交和资金假设限制结果的解释 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/395_signature_based_optimal_execution_for_st.py | INVALID | 资金约束在同一收盘价重定成交股数 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/400_noxtrader_lstm_based_stock_return_moment.py | INVALID | 幸存者与事后选样偏差 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/431_mitigating_return_drag_and_regime_whipsa.py | INVALID | 实际买入份额会按成交收盘价缩减 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/432_revisiting_equity_bond_diversification_u.py | INVALID | 共用模拟器在成交收盘重新确定买入份额; 收盘模型使用事后复权快照与简化费用 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/434_dynamic_scenario_based_portfolio_optimiz.py | INVALID | 事后资产池与回顾性样本外; 模拟成交和成本的适用范围 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/442_mitigating_return_drag_and_regime_whipsa.py | INVALID | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/443_revisiting_equity_bond_diversification_u.py | INVALID | 最终买入量可由成交日收盘价决定; 动态配置缺少实际成交证据 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/445_dynamic_scenario_based_portfolio_optimiz.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/448_continuous_cash_overlay_filters_for_a_st.py | INVALID | look-ahead / same-close fill | labelled per Gatekeeper ruling 2026-11-27 12:33 Europe/Oslo |
| strategies/449_the_mathematics_of_heuristic_portfolio_o.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/450_a_declining_cvar_glidepath_framework_for.py | INVALID | 成交日收盘价改变实际买入份额 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/451_a_declining_cvar_glidepath_framework_for.py | INVALID | 资金不足时，最终买入份额由成交日收盘价反算 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/452_simple_dynamic_stock_bond_gold_portfolio.py | INVALID | 现金不足时，成交日收盘价决定实际买入份额 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/454_semivariance_and_semivolatility_in_portf.py | INVALID | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/455_hierarchical_risk_parity_for_portfolio_a.py | INVALID | 复权价格和现金处理简化了成交 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/456_an_entropic_factor_model_for_robust_port.py | INVALID | 成交模型对流动性和现金冲击简化 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/457_cleaning_correlation_matrices_with_exter.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/458_two_sides_of_schur_damping_high_dimensio.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/459_entropic_value_at_risk_parity_for_temper.py | INVALID | 实际买入份额由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/460_understanding_the_long_only_minimum_vari.py | INVALID | 复权与成交模型的限制 | labelled per Gatekeeper ruling 2026-11-27 12:33 Europe/Oslo |
| strategies/461_breaking_the_trend_how_to_avoid_cherry_p.py | INVALID | 成交日收盘价决定了实际成交数量; 资金与成交模型只覆盖日频 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/465_semivariance_and_semivolatility_in_portf.py | INVALID | 复权快照限制成交与数据时点的可追溯性 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/467_an_entropic_factor_model_for_robust_port.py | INVALID | 实际买入量依赖成交收盘价; 固定股票池存在幸存者与事后筛选偏差; 复权成交与资金约束是简化模型 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/468_cleaning_correlation_matrices_with_exter.py | INVALID | look-ahead / same-close fill | labelled per Gatekeeper ruling 2026-11-27 12:33 Europe/Oslo |
| strategies/469_two_sides_of_schur_damping_high_dimensio.py | INVALID | 资产池存在事后存续筛选; 成交模型简化了现金与流动性 | labelled per Gatekeeper ruling 2026-11-27 12:33 Europe/Oslo |
| strategies/470_entropic_value_at_risk_parity_for_temper.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/472_breaking_the_trend_how_to_avoid_cherry_p.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/474_multiperiod_bond_portfolio_optimization.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/476_conditional_value_at_risk_under_reward_p.py | INVALID | 成交日收盘价参与决定实际买入数量 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/478_portfolio_allocation_under_heterogeneous.py | INVALID | look-ahead / same-close fill | labelled per Gatekeeper ruling 2026-11-27 12:33 Europe/Oslo |
| strategies/479_noise_proofing_universal_portfolio_shrin.py | INVALID | 成交与资金可用性依赖模拟假设 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/480_topological_risk_parity.py | INVALID | 实际买入份额可由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/481_optimal_portfolio_construction_a_reinfor.py | INVALID | 现金不足时以成交日收盘价决定实际买量 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/482_dynamic_inclusion_and_bounded_multi_fact.py | INVALID | 固定候选池存在幸存者偏差及事后选择偏差; 成交及现金约束是简化模型 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/483_wasserstein_penalisation_as_an_endogenou.py | INVALID | 最终买入份额由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/484_nonlinear_shrinkage_for_higher_moments_e.py | INVALID | 实际买入份额可由成交日收盘价决定; 股票池是事后固定的幸存者样本 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/487_conditional_value_at_risk_under_reward_p.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/490_noise_proofing_universal_portfolio_shrin.py | INVALID | 成交日收盘价决定实际买入量 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/491_topological_risk_parity.py | INVALID | 成交数量依赖成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/494_wasserstein_penalisation_as_an_endogenou.py | INVALID | look-ahead / same-close fill | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/497_interpretable_allocation_and_model_predi.py | INVALID | look-ahead / same-close fill | labelled per Gatekeeper ruling 2026-11-27 12:33 Europe/Oslo |
| strategies/500_dynamic_portfolio_optimization_under_cva.py | INVALID | 收盘价参与确定同一收盘的实际买量 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/501_nested_clustered_optimization_is_one_end.py | INVALID | 收盘价决定实际买入份额 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/503_risk_limited_asset_allocation_with_a_bud.py | INVALID | 收盘后重定买入份额 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/505_competitive_optimal_portfolio_selection.py | INVALID | 成交日收盘价决定部分实际买入份额 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/509_beyond_de_prado_and_cotton_hierarchical.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/510_some_general_results_on_risk_budgeting_p.py | INVALID | 实际买入数量由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/512_nested_clustered_optimization_is_one_end.py | INVALID | see review | labelled per Gatekeeper ruling 2026-11-27 12:33 Europe/Oslo |
| strategies/514_risk_limited_asset_allocation_with_a_bud.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/520_beyond_de_prado_and_cotton_hierarchical.py | INVALID | 实际买入份额依赖成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/522_asset_allocation_using_a_markov_process.py | INVALID | 实际买入份额取决于成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/528_schur_complementary_allocation_a_unifica.py | INVALID | 复权收盘成交与线性成本是近似 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/532_robust_enhanced_index_tracking_portfolio.py | INVALID | 资金不足时用成交日收盘价改写买入股数; 事后固定选池造成幸存者偏差和成分选择偏差 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/533_special_markowitz_thermodynamic_formalis.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/534_portfolio_diversification_and_concentrat.py | INVALID | 最终买入数量依赖成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/535_shrinkage_estimators_for_mean_and_covari.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/537_hierarchical_minimum_variance_portfolios.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/538_mirror_descent_algorithms_for_risk_budge.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/539_schur_complementary_allocation_a_unifica.py | INVALID | 现金不足时用成交日收盘价重算买入份额 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/541_entropy_regularization_under_no_shorting.py | INVALID | 最终买入份额使用成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/543_robust_enhanced_index_tracking_portfolio.py | INVALID | 最终买入量依赖成交日收盘价; 固定股票池存在事后选股与幸存者偏差 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/544_special_markowitz_thermodynamic_formalis.py | INVALID | 成交日收盘价决定部分实际买入份额 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/545_portfolio_diversification_and_concentrat.py | INVALID | 实际买入份额依赖成交日收盘价 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/546_generalizing_markowitz_portfolio_optimiz.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/547_generalizing_markowitz_portfolio_optimiz.py | INVALID | 最终买入数量可由成交日收盘价决定 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/548_entropic_value_at_risk_portfolio_optimiz.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/551_squeezed_covariance_matrix_estimation_an.py | INVALID | 成交时用收盘价缩减买量; 事后固定股票池 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/552_a_novel_robust_mixed_integer_linear_prog.py | INVALID | 现金不足时按成交日收盘改写实际买入股数; 固定候选池存在幸存者偏差及事后选样; 模拟成交与实际市场价格、容量和公司行动有差距 | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |
| strategies/553_large_scale_portfolio_optimization_probl.py | INVALID | see review | restored_from_origin: kept_from_origin: latest self-review lacks a qualifying post-fix execution-sizing check |

# Deep-review findings (already published)

Already-published strategies are not removed from git history. If a later deep review finds a real problem (look-ahead, wrong data, or wrong numbers), the file is kept as last published on origin/main and listed here in a new commit. Fixes, if any, also land in a new commit.

Generated 2026-11-27 18:43 Europe/Oslo.

| File | Latest self-review | Category | Stage note |
|---|---|---|---|
| strategies/501_nested_clustered_optimization_is_one_end.py | INVALID | 固定资产池存在幸存者与选择偏差; 样本外为事后切分的历史留出 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/451_a_declining_cvar_glidepath_framework_for.py | INVALID | 固定代理存在幸存者选择偏差; 历史复权版本与行情可得时点缺少记录 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/510_some_general_results_on_risk_budgeting_p.py | INVALID | look-ahead / same-close fill | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/452_simple_dynamic_stock_bond_gold_portfolio.py | INVALID | 固定存续 ETF 池存在幸存者与事后选择偏差; 复权快照缺少逐日数据版本 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/494_wasserstein_penalisation_as_an_endogenou.py | INVALID | 固定成交假设与现金约束 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/533_special_markowitz_thermodynamic_formalis.py | INVALID | 资金缓冲有失效边界，成交成本为固定假设 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/514_risk_limited_asset_allocation_with_a_bud.py | INVALID | 资金约束和成交假设限制结果解释 | kept_from_origin: 完整性不合格：has_broker_or_order_code |
| strategies/484_nonlinear_shrinkage_for_higher_moments_e.py | INVALID | 事后选股造成的幸存者偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/476_conditional_value_at_risk_under_reward_p.py | INVALID | 收盘成交与现金可用性采用简化模型 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/432_revisiting_equity_bond_diversification_u.py | INVALID | 事后选池及存续偏差; 动态配置缺乏成交证据; 成交及资金约束是简化模型 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/505_competitive_optimal_portfolio_selection.py | INVALID | 成交与资金约束采用简化模型 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/539_schur_complementary_allocation_a_unifica.py | INVALID | 固定资产池存在幸存者与事后选池偏差; 样本外为事后留出段; 执行对成交价跳变和现金余量敏感 | kept_from_origin: 完整性不合格：lag_ok |
| strategies/449_the_mathematics_of_heuristic_portfolio_o.py | INVALID | 基金池存在幸存者偏差与事后选样 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/483_wasserstein_penalisation_as_an_endogenou.py | INVALID | 固定基金组连续报价是回测条件; 快照缺少逐日数据版本 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/503_risk_limited_asset_allocation_with_a_bud.py | INVALID | 收盘结算的资金与流动性假设 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/431_mitigating_return_drag_and_regime_whipsa.py | INVALID | 模拟执行依赖收盘成交及资金假设 | kept_from_origin: not latest-tree self-review PASS with 成交数量检查 |
| strategies/532_robust_enhanced_index_tracking_portfolio.py | INVALID | 事后固定股票池及幸存者偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/534_portfolio_diversification_and_concentrat.py | INVALID | 固定份额的资金与成交假设 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/509_beyond_de_prado_and_cotton_hierarchical.py | INVALID | 成交和资金模型对价差与流动性变化覆盖有限 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/520_beyond_de_prado_and_cotton_hierarchical.py | INVALID | 成交模型依赖现金缓冲和固定摩擦 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/395_signature_based_optimal_execution_for_st.py | INVALID | 事后选对与幸存者偏差; 成交成本与资金约束有明确的模拟边界 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/450_a_declining_cvar_glidepath_framework_for.py | INVALID | 收盘成交与资金模型 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/551_squeezed_covariance_matrix_estimation_an.py | INVALID | 论文表 3 通道权重的发布时滞晚于样本期末; 固定名单存在幸存者与事后选股偏差 | kept_from_origin: 完整性不合格：lag_ok |
| strategies/481_optimal_portfolio_construction_a_reinfor.py | INVALID | 固定现金缓冲与全额收盘成交假设限制执行稳健性 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/500_dynamic_portfolio_optimization_under_cva.py | INVALID | 历史复权版本与发布时滞无法从现有快照证实; 固定资产池存在幸存者与事后选取偏差 | kept_from_origin: 完整性不合格：lag_ok |
| strategies/541_entropy_regularization_under_no_shorting.py | INVALID | 历史行情与成交假设限制外推 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/459_entropic_value_at_risk_parity_for_temper.py | INVALID | 固定资产池存在幸存者偏差和事后选择偏差; 固定成本和收盘结算限制了结果的执行含义 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/497_interpretable_allocation_and_model_predi.py | INVALID | 固定研究池存在幸存者偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/442_mitigating_return_drag_and_regime_whipsa.py | INVALID | 历史样本外与论文参数评估期重叠; 全期完整报价造成事后条件 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/022_sector_residual_statarb.py | INVALID | 成交和资金模型有简化 | kept_from_origin: 完整性不合格：lag_ok,reads_local_files_other_than_config_and_cache |
| strategies/465_semivariance_and_semivolatility_in_portf.py | INVALID | 成交和现金取决于简化假设 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/113_kalman_garch_pairs.py | INVALID | 固定存续配对限制结果的外推; 成交与资金约束为简化模型 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/315_leveraged_funds_robust_replication.py | INVALID | 事后选池限制结论的适用范围 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/341_monitoring_attention_inflation_news.py | INVALID | 入库目录缺少必需行情快照 | kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/545_portfolio_diversification_and_concentrat.py | INVALID | 固定资产池有幸存者偏差和事后选样 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/552_a_novel_robust_mixed_integer_linear_prog.py | INVALID | 固定股票池存在幸存者偏差; 成交与异常行情采用简化规则 | kept_from_origin: 完整性不合格：lag_ok |
| strategies/461_breaking_the_trend_how_to_avoid_cherry_p.py | INVALID | 固定基金名单带来幸存者偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/487_conditional_value_at_risk_under_reward_p.py | INVALID | 股票池含事后信息; 执行结果依赖现金缓冲及简化成交假设 | kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/538_mirror_descent_algorithms_for_risk_budge.py | INVALID | 现金与收盘成交假设限制结果外推 | kept_from_origin: 完整性不合格：has_broker_or_order_code |
| strategies/467_an_entropic_factor_model_for_robust_port.py | INVALID | 股票池使用事后信息; 成交和资金处理有简化 | kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/478_portfolio_allocation_under_heterogeneous.py | INVALID | 缺少官方行情快照; 开盘成交假设及资金约束 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/491_topological_risk_parity.py | INVALID | 事后固定投资池; 成交与资金模型的适用范围 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/188_opex_week.py | INVALID | 开收盘成交及资金假设偏强 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/443_revisiting_equity_bond_diversification_u.py | INVALID | 动态配置缺少成交证据 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/159_deep_momentum_market_trend.py | INVALID | 固定基金池存在幸存者与事后选择偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/109_factor_etf_vol_matched.py | INVALID | 基金选择与幸存者偏差; OOS 的事前性无法追溯 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/186_federal_holiday_effect.py | INVALID | 开盘换仓依赖成交与资金假设 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/490_noise_proofing_universal_portfolio_shrin.py | INVALID | 固定存续 ETF 池产生幸存者与选择偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/456_an_entropic_factor_model_for_robust_port.py | INVALID | 成交与成本模型简化 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/537_hierarchical_minimum_variance_portfolios.py | INVALID | 执行模型简化收盘流动性与现金约束 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/400_noxtrader_lstm_based_stock_return_moment.py | INVALID | 固定股票池引入幸存者和事后选样偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/458_two_sides_of_schur_damping_high_dimensio.py | INVALID | 固定标的池存在幸存者偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/454_semivariance_and_semivolatility_in_portf.py | INVALID | 资金与成交模型简化 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/448_continuous_cash_overlay_filters_for_a_st.py | INVALID | 名单和策略设计带有事后信息; 成交和分配简化 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/468_cleaning_correlation_matrices_with_exter.py | INVALID | 固定存续标的池存在幸存者偏差，须在入库前处理; 收盘批量成交和摩擦模型有限 | kept_from_origin: latest self-review is not a post-fix PASS with 成交数量检查 |
| strategies/324_volatility_conditions_weekend_effect.py | INVALID | 固定标的与全期报价条件存在事后选择风险 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/460_understanding_the_long_only_minimum_vari.py | INVALID | 成交模型与资金缓冲有现实限制 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/469_two_sides_of_schur_damping_high_dimensio.py | INVALID | 固定基金池有幸存者偏差与事后选样偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/474_multiperiod_bond_portfolio_optimization.py | INVALID | 规划成本与实际成交近似不同 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/512_nested_clustered_optimization_is_one_end.py | INVALID | 固定资产池的幸存者与选择偏差; 样本外为事后切分的历史留出 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/197_tsmom_boundaries.py | INVALID | 开盘成交和股息现金的近似 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/553_large_scale_portfolio_optimization_probl.py | INVALID | 候选池有幸存者和事后选样偏差; 成交与资金模型简化 | kept_from_origin: 完整性不合格：lag_ok,has_broker_or_order_code |
| strategies/310_factor_beta_overnight_intraday.py | INVALID | OOS 的独立性不足; 费用与成交模型简化 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/353_hold_the_dip.py | INVALID | look-ahead / same-close fill | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/445_dynamic_scenario_based_portfolio_optimiz.py | INVALID | ETF 池有幸存者和事后选择偏差; 复权成交与资金处理是模型假设 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/482_dynamic_inclusion_and_bounded_multi_fact.py | INVALID | 固定候选池有幸存者和事后选择偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/434_dynamic_scenario_based_portfolio_optimiz.py | INVALID | 事后资产池与回顾性样本切分 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/528_schur_complementary_allocation_a_unifica.py | INVALID | 固定基金池存在幸存者偏差与资产选择偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/354_strategy.py | INVALID | 成交假设 | kept_from_origin: 完整性不合格：reads_local_files_other_than_config_and_cache |
| strategies/470_entropic_value_at_risk_parity_for_temper.py | INVALID | 固定基金池有幸存者偏差及事后选池偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/522_asset_allocation_using_a_markov_process.py | INVALID | 收盘执行依赖固定滑点和现金缓冲 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/479_noise_proofing_universal_portfolio_shrin.py | INVALID | 基金池存在事后选择和幸存者偏差; 模拟成交依赖复权日线和固定摩擦 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |
| strategies/156_cta_replication_trend_factors.py | INVALID | 固定 ETF 池缺少退市替换规则，存在事后存续筛选偏差 | kept_from_origin: latest-tree PASS but META is missing or older than the current code |

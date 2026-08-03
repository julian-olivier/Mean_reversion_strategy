import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd

from src.cluster import extract_features, cluster_assets, get_cluster_groups
from src.cointegration import calculate_spread, test_pair_cointegration, find_cointegrated_pairs, test_stationarity_consistency, select_top_distinct_pairs
from src.strategy import fit_vasicek_model, simulate_vasicek, generate_vasicek_signals
from src.backtest import (
    calculate_performance_metrics,
    run_pair_backtest,
    sweep_entry_deviations,
    run_multi_pair_portfolio_backtest,
    calculate_portfolio_vs_benchmark_metrics
)


def synthetic_prices():
    """Generates synthetic stationary cointegrated price series for testing."""
    np.random.seed(42)
    n_days = 300
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')

    # Common random walk
    x_walk = 100.0 + np.cumsum(np.random.normal(0, 1, size=n_days))
    
    # Cointegrated series Y = 1.5 * X + stationary noise
    stationary_noise = np.zeros(n_days)
    for i in range(1, n_days):
        stationary_noise[i] = 0.7 * stationary_noise[i-1] + np.random.normal(0, 0.5)
        
    y_walk = 1.5 * x_walk + 10.0 + stationary_noise

    # Second independent series Z
    z_walk = 50.0 + np.cumsum(np.random.normal(0, 1.2, size=n_days))
    
    # Negative beta cointegrated pair W = -0.8 * X + noise
    w_walk = -0.8 * x_walk + 150.0 + stationary_noise

    df = pd.DataFrame({
        'Stock_Y': y_walk,
        'Stock_X': x_walk,
        'Stock_Z': z_walk,
        'Stock_W': w_walk
    }, index=dates)
    return df


def test_cluster_module(synthetic_prices):
    feat_df, pca = extract_features(synthetic_prices, n_components=2)
    assert feat_df.shape[0] == 4
    assert 'PC_1' in feat_df.columns

    cluster_df = cluster_assets(synthetic_prices, method='agglomerative', n_clusters=2)
    assert len(cluster_df) == 4
    assert 'Cluster' in cluster_df.columns

    groups = get_cluster_groups(cluster_df)
    assert isinstance(groups, dict)


def test_cointegration_module(synthetic_prices):
    spread, beta, alpha = calculate_spread(synthetic_prices['Stock_Y'], synthetic_prices['Stock_X'])
    assert np.isclose(beta, 1.5, atol=0.2)

    coint_res = test_pair_cointegration(synthetic_prices['Stock_Y'], synthetic_prices['Stock_X'])
    assert 'p_value' in coint_res
    assert coint_res['p_value'] < 0.05  # Strongly cointegrated

    coint_df = find_cointegrated_pairs(synthetic_prices, p_value_threshold=0.05)
    assert not coint_df.empty
    # Verify strict Engle-Granger test filtering
    assert all(coint_df['Coint_PValue'] <= 0.05)

    consistency = test_stationarity_consistency(synthetic_prices, 'Stock_Y', 'Stock_X')
    assert 'InSample_ADF_PValue' in consistency
    assert 'OutSample_ADF_PValue' in consistency

    distinct_pairs = select_top_distinct_pairs(coint_df, max_pairs=2)
    assert len(distinct_pairs) <= 2


def test_strategy_module(synthetic_prices):
    spread, _, _ = calculate_spread(synthetic_prices['Stock_Y'], synthetic_prices['Stock_X'])
    v_params = fit_vasicek_model(spread)

    assert v_params['kappa'] > 0
    assert np.isfinite(v_params['half_life'])
    assert v_params['half_life'] > 0

    sim_spread = simulate_vasicek(
        theta=v_params['theta'],
        kappa=v_params['kappa'],
        sigma=v_params['sigma'],
        state_0=spread.iloc[0],
        n_steps=50,
        seed=42
    )
    assert len(sim_spread) == 50

    signals = generate_vasicek_signals(spread, v_params, entry_z=1.5)
    assert 'Z_Score' in signals.columns
    assert 'Position' in signals.columns


def test_backtest_in_and_out_of_sample(synthetic_prices):
    # Test Out-of-sample backtest
    eq_out, trades_out, metrics_out = run_pair_backtest(
        synthetic_prices, 'Stock_Y', 'Stock_X',
        use_out_of_sample_only=True, train_ratio=0.7
    )
    assert not eq_out.empty
    assert 'Sharpe_Ratio' in metrics_out

    # Test In-sample / full sample backtest (split_idx == 0) - verifies index length alignment bug fix
    eq_full, trades_full, metrics_full = run_pair_backtest(
        synthetic_prices, 'Stock_Y', 'Stock_X',
        use_out_of_sample_only=False
    )
    assert len(eq_full) == len(synthetic_prices) - 1
    assert 'Sharpe_Ratio' in metrics_full


def test_sortino_ratio_robustness():
    # Test case 1: 0 negative return days
    eq_curve_pos = pd.Series(1.0 + np.linspace(0, 0.5, 100))
    metrics_pos = calculate_performance_metrics(eq_curve_pos, pd.DataFrame())
    assert not np.isnan(metrics_pos['Sortino_Ratio'])

    # Test case 2: 1 negative return day
    returns = np.array([0.01] * 99 + [-0.005])
    eq_curve_one_neg = pd.Series(np.cumprod(1 + returns))
    metrics_one_neg = calculate_performance_metrics(eq_curve_one_neg, pd.DataFrame())
    assert not np.isnan(metrics_one_neg['Sortino_Ratio'])
    assert metrics_one_neg['Sortino_Ratio'] > 0


def test_multi_pair_portfolio_backtest(synthetic_prices):
    coint_df = find_cointegrated_pairs(synthetic_prices, p_value_threshold=0.05)
    port_df, trades_df, metrics = run_multi_pair_portfolio_backtest(
        prices=synthetic_prices,
        selected_pairs_df=coint_df,
        entry_z=1.5,
        stop_z=3.0,
        train_ratio=0.6,
        use_out_of_sample_only=True
    )

    assert 'Equity' in port_df.columns
    assert 'Active_Long_Legs' in port_df.columns
    assert 'Active_Short_Legs' in port_df.columns
    assert metrics['Total_Return'] is not None

    bench_series = pd.Series(1.0 + np.linspace(0, 0.1, len(port_df)), index=port_df.index)
    comp_res = calculate_portfolio_vs_benchmark_metrics(
        portfolio_equity=port_df['Equity'],
        benchmark_equity=bench_series,
        trades_df=trades_df
    )
    assert 'comparison_df' in comp_res


if __name__ == '__main__':
    synthetic_df = synthetic_prices()
    print("Running test_cluster_module...")
    test_cluster_module(synthetic_df)
    print("Running test_cointegration_module...")
    test_cointegration_module(synthetic_df)
    print("Running test_strategy_module...")
    test_strategy_module(synthetic_df)
    print("Running test_backtest_in_and_out_of_sample...")
    test_backtest_in_and_out_of_sample(synthetic_df)
    print("Running test_sortino_ratio_robustness...")
    test_sortino_ratio_robustness()
    print("Running test_multi_pair_portfolio_backtest...")
    test_multi_pair_portfolio_backtest(synthetic_df)
    print("\nALL 6 TEST SUITE VERIFICATIONS PASSED SUCCESSFULLY!")


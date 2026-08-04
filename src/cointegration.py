import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
import statsmodels.api as sm
from src.cluster import get_cluster_groups


def calculate_spread(
    series_y: pd.Series,
    series_x: pd.Series
) -> tuple[pd.Series, float, float]:
    """
    Computes spread S_t = Y_t - (beta * X_t + alpha) using OLS regression.
    Returns: (spread_series, beta, alpha)
    """
    y = series_y.to_numpy()
    x = series_x.to_numpy()
    var_x = np.var(x)
    if var_x < 1e-8:
        beta = 0.0
        alpha = 0.0
    else:
        cov_xy = np.cov(y, x)[0, 1]
        beta = float(cov_xy / var_x)
        alpha = float(np.mean(y) - beta * np.mean(x))
    spread = series_y - (beta * series_x + alpha)
    return spread, beta, alpha


def test_pair_cointegration(
    series_y: pd.Series,
    series_x: pd.Series
) -> dict:
    """
    Performs Engle-Granger 2-step cointegration test and Dickey-Fuller stationarity test on spread.
    """
    spread, beta, alpha = calculate_spread(series_y, series_x)
    spread_arr = spread.dropna().to_numpy()

    if len(spread_arr) < 20:
        return {
            'coint_stat': 0.0,
            'p_value': 1.0,
            'adf_stat': 0.0,
            'adf_pvalue': 1.0,
            'beta': beta,
            'alpha': alpha,
            'spread_std': 0.0,
            'spread_mean': 0.0
        }

    adf_result = adfuller(spread_arr, maxlag=1)
    adf_stat = adf_result[0]
    adf_pvalue = adf_result[1]

    return {
        'coint_stat': adf_stat,
        'p_value': adf_pvalue,
        'adf_stat': adf_stat,
        'adf_pvalue': adf_pvalue,
        'beta': beta,
        'alpha': alpha,
        'spread_std': float(np.std(spread_arr)),
        'spread_mean': float(np.mean(spread_arr))
    }


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    cluster_df: pd.DataFrame = None,
    p_value_threshold: float = 0.05,
    max_pairs_per_cluster: int = 50,
    min_cluster_size: int = 2
) -> pd.DataFrame:
    """
    Searches intra-cluster stock pairs for statistically significant cointegration.
    If cluster_df is None, tests all pairwise combinations.
    """
    cointegrated_pairs = []

    if cluster_df is not None:
        groups = get_cluster_groups(cluster_df)
        pair_candidates = []
        for c_id, tickers in groups.items():
            if c_id == -1 or len(tickers) < min_cluster_size:  # Skip noise/unclustered or small clusters
                continue
            pair_candidates.extend(list(itertools.combinations(tickers, 2)))
    else:
        tickers = list(prices.columns)
        pair_candidates = list(itertools.combinations(tickers, 2))

    print(f"Testing cointegration across {len(pair_candidates)} intra-cluster pair candidates...")

    for ticker_a, ticker_b in pair_candidates:
        if ticker_a not in prices.columns or ticker_b not in prices.columns:
            continue

        series_a = prices[ticker_a].dropna()
        series_b = prices[ticker_b].dropna()

        # Align indices
        common_idx = series_a.index.intersection(series_b.index)
        if len(common_idx) < 100:
            continue

        s_a = series_a.loc[common_idx]
        s_b = series_b.loc[common_idx]

        res = test_pair_cointegration(s_a, s_b)

        if res['p_value'] <= p_value_threshold:
            cointegrated_pairs.append({
                'Stock_Y': ticker_a,
                'Stock_X': ticker_b,
                'Beta': res['beta'],
                'Alpha': res['alpha'],
                'Coint_PValue': res['p_value'],
                'ADF_Stat': res['adf_stat'],
                'ADF_PValue': res['adf_pvalue'],
                'Spread_Std': res['spread_std']
            })

    results_df = pd.DataFrame(cointegrated_pairs)
    if not results_df.empty:
        results_df = results_df.sort_values(by='Coint_PValue').reset_index(drop=True)
    return results_df


def test_stationarity_consistency(
    prices: pd.DataFrame,
    stock_y: str,
    stock_x: str,
    train_ratio: float = 0.7
) -> dict:
    """
    Evaluates consistency of Dickey-Fuller stationarity results in-sample vs out-of-sample.
    """
    series_y = prices[stock_y].dropna()
    series_x = prices[stock_x].dropna()

    common_idx = series_y.index.intersection(series_x.index)
    series_y = series_y.loc[common_idx]
    series_x = series_x.loc[common_idx]

    split_idx = int(len(series_y) * train_ratio)

    y_in = series_y.iloc[:split_idx]
    x_in = series_x.iloc[:split_idx]

    y_out = series_y.iloc[split_idx:]
    x_out = series_x.iloc[split_idx:]

    # In-sample cointegration test (Engle-Granger test p-value to avoid OLS fitting bias)
    coint_stat_in, pval_in, _ = coint(y_in, x_in)
    spread_in, beta, alpha = calculate_spread(y_in, x_in)

    # Out-of-sample spread using in-sample parameters (ADF test valid on fixed spread)
    spread_out = y_out - (beta * x_out + alpha)
    adf_out = adfuller(spread_out.dropna(), maxlag=1)

    in_stationary = pval_in <= 0.05
    out_stationary = adf_out[1] <= 0.05

    return {
        'Stock_Y': stock_y,
        'Stock_X': stock_x,
        'InSample_ADF_PValue': pval_in,
        'InSample_IsStationary': in_stationary,
        'OutSample_ADF_PValue': adf_out[1],
        'OutSample_IsStationary': out_stationary,
        'Consistent': (in_stationary == out_stationary),
        'Retained_Stationarity': (in_stationary and out_stationary),
        'Beta': beta,
        'Alpha': alpha
    }


def evaluate_universe_df_consistency(
    prices: pd.DataFrame,
    cointegrated_pairs_df: pd.DataFrame,
    train_ratio: float = 0.7,
    max_pairs_eval: int = 50
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluates in-sample vs out-of-sample Dickey-Fuller stationarity consistency across all candidate pairs.
    Returns: (consistency_df, summary_stats_dict)
    """
    if cointegrated_pairs_df.empty:
        return pd.DataFrame(), {}

    eval_results = []
    eval_subset = cointegrated_pairs_df.head(max_pairs_eval)

    for _, row in eval_subset.iterrows():
        res = test_stationarity_consistency(prices, row['Stock_Y'], row['Stock_X'], train_ratio=train_ratio)
        eval_results.append(res)

    results_df = pd.DataFrame(eval_results)

    total_eval = len(results_df)
    consistent_count = results_df['Consistent'].sum()
    retained_count = results_df['Retained_Stationarity'].sum()

    summary_stats = {
        'Total_Evaluated': total_eval,
        'Consistent_Count': consistent_count,
        'Consistency_Rate_%': (consistent_count / total_eval) * 100 if total_eval > 0 else 0,
        'Retained_Stationarity_Count': retained_count,
        'Retained_Stationarity_Rate_%': (retained_count / total_eval) * 100 if total_eval > 0 else 0,
        'Avg_InSample_PValue': results_df['InSample_ADF_PValue'].mean(),
        'Avg_OutSample_PValue': results_df['OutSample_ADF_PValue'].mean()
    }

    return results_df, summary_stats


def select_top_distinct_pairs(
    cointegrated_df: pd.DataFrame,
    max_pairs: int = 10,
    allow_stock_reuse: bool = False
) -> pd.DataFrame:
    """
    Selects top cointegrated pairs sorted by ADF statistical significance (most negative ADF_Stat / lowest ADF_PValue).
    If allow_stock_reuse is False, ensures each stock appears in at most one pair to prevent asset concentration.
    """
    if cointegrated_df.empty:
        return pd.DataFrame()

    # Sort by strongest cointegration (lowest Coint_PValue / lowest ADF_Stat)
    sorted_df = cointegrated_df.sort_values(by=['Coint_PValue', 'ADF_Stat'], ascending=[True, True]).reset_index(drop=True)

    if allow_stock_reuse:
        return sorted_df.head(max_pairs)

    selected_rows = []
    used_stocks = set()

    for _, row in sorted_df.iterrows():
        stock_y = row['Stock_Y']
        stock_x = row['Stock_X']

        if stock_y not in used_stocks and stock_x not in used_stocks:
            selected_rows.append(row)
            used_stocks.add(stock_y)
            used_stocks.add(stock_x)

        if len(selected_rows) >= max_pairs:
            break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


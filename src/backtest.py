import numpy as np
import pandas as pd
import yfinance as yf
from src.cointegration import calculate_spread
from src.strategy import fit_vasicek_model


def calculate_performance_metrics(
    equity_curve: pd.Series,
    trades_df: pd.DataFrame,
    risk_free_rate: float = 0.02
) -> dict:
    """
    Computes performance metrics from daily equity curve and trade logs.
    """
    daily_returns = equity_curve.pct_change().dropna()
    total_return = equity_curve.iloc[-1] - 1.0
    n_days = len(equity_curve)

    if n_days > 1:
        ann_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0
        ann_vol = daily_returns.std() * np.sqrt(252)
        rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
        excess_ret = daily_returns - rf_daily
        sharpe_ratio = (excess_ret.mean() / (daily_returns.std() + 1e-8)) * np.sqrt(252)

        downside_std = daily_returns[daily_returns < 0].std() * np.sqrt(252)
        sortino_ratio = (ann_return - risk_free_rate) / (downside_std + 1e-8)
    else:
        ann_return = ann_vol = sharpe_ratio = sortino_ratio = 0.0

    # Max Drawdown
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    max_drawdown = drawdown.min()

    # Trade stats
    if not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['Return'] > 0])
        win_rate = winning_trades / total_trades
        avg_trade_return = trades_df['Return'].mean()
        avg_holding_days = trades_df['Holding_Days'].mean()

        gains = trades_df[trades_df['Return'] > 0]['Return'].sum()
        losses = abs(trades_df[trades_df['Return'] < 0]['Return'].sum())
        profit_factor = gains / losses if losses > 0 else np.nan
        exit_reasons = trades_df['Exit_Reason'].value_counts().to_dict()
    else:
        total_trades = win_rate = avg_trade_return = avg_holding_days = profit_factor = 0.0
        exit_reasons = {}

    return {
        'Total_Return': total_return,
        'Annualized_Return': ann_return,
        'Annualized_Vol': ann_vol,
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Max_Drawdown': max_drawdown,
        'Total_Trades': total_trades,
        'Win_Rate': win_rate,
        'Avg_Trade_Return': avg_trade_return,
        'Avg_Holding_Days': avg_holding_days,
        'Profit_Factor': profit_factor,
        'Exit_Reasons': exit_reasons
    }


def run_pair_backtest(
    prices: pd.DataFrame,
    stock_y: str,
    stock_x: str,
    vasicek_params: dict = None,
    entry_z: float = 2.0,
    stop_z: float = 3.5,
    max_holding_days: int = 20,
    transaction_cost: float = 0.0005,
    train_ratio: float = 0.7,
    use_out_of_sample_only: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Backtests a mean-reversion pair strategy incorporating:
      - In-sample estimation of cointegration & Vasicek parameters (first train_ratio of data)
      - Out-of-sample backtest simulation (remaining 1 - train_ratio of data)
      - Entry at Z-score thresholds (z <= -entry_z or z >= entry_z)
      - Exit 1: Mean Reversion (spread crosses Vasicek mean theta)
      - Exit 2: Stop Loss (spread moves past stop_z threshold)
      - Exit 3: Max Holding Duration (holding period > max_holding_days)

    Returns: (equity_df, trades_df, summary_metrics)
    """
    series_y = prices[stock_y].dropna()
    series_x = prices[stock_x].dropna()
    common_idx = series_y.index.intersection(series_x.index)

    series_y = series_y.loc[common_idx]
    series_x = series_x.loc[common_idx]

    split_idx = int(len(common_idx) * train_ratio) if (use_out_of_sample_only and train_ratio < 1.0) else 0

    if split_idx > 0:
        s_y_in = series_y.iloc[:split_idx]
        s_x_in = series_x.iloc[:split_idx]
        spread_in, beta, alpha = calculate_spread(s_y_in, s_x_in)
        if vasicek_params is None:
            vasicek_params = fit_vasicek_model(spread_in)
        spread = series_y - (beta * series_x + alpha)
    else:
        spread, beta, alpha = calculate_spread(series_y, series_x)
        if vasicek_params is None:
            vasicek_params = fit_vasicek_model(spread)

    theta = vasicek_params['theta']
    kappa = vasicek_params['kappa']
    sigma = vasicek_params['sigma']

    sigma_eq = sigma / np.sqrt(2.0 * max(kappa, 1e-6)) if kappa > 0 else spread.std()
    z_score = (spread - theta) / (sigma_eq + 1e-8)

    eval_idx = common_idx[split_idx:] if split_idx > 0 else common_idx

    portfolio_value = 1.0
    equity = []

    position = 0          # 0: Cash, +1: Long Spread (Buy Y, Sell X), -1: Short Spread (Sell Y, Buy X)
    entry_date = None
    entry_z_val = 0.0
    entry_spread = 0.0
    entry_portfolio_val = 1.0
    holding_days = 0

    trades = []
    df_signals = pd.DataFrame(index=eval_idx)
    df_signals['Spread'] = spread.loc[eval_idx]
    df_signals['Z_Score'] = z_score.loc[eval_idx]
    df_signals['Position'] = 0
    positions_series = np.zeros(len(eval_idx))

    start_loop = split_idx if split_idx > 0 else 1

    for step, i in enumerate(range(start_loop, len(common_idx))):
        date = common_idx[i]
        prev_date = common_idx[i-1]

        z_curr = z_score.iloc[i]
        spread_curr = spread.iloc[i]

        s_y_curr, s_y_prev = series_y.iloc[i], series_y.iloc[i-1]
        s_x_curr, s_x_prev = series_x.iloc[i], series_x.iloc[i-1]

        delta_y = s_y_curr - s_y_prev
        delta_x = s_x_curr - s_x_prev
        capital_prev = s_y_prev + abs(beta) * s_x_prev

        # Spread daily return relative to capital invested
        if position == 1:
            daily_pnl = (delta_y - beta * delta_x) / capital_prev
        elif position == -1:
            daily_pnl = (-delta_y + beta * delta_x) / capital_prev
        else:
            daily_pnl = 0.0

        # Position tracking & exit condition checks
        exit_triggered = False
        exit_reason = None

        if position != 0:
            holding_days += 1

            # Check Exit 1: Mean Reversion
            if (position == 1 and z_curr >= 0.0) or (position == -1 and z_curr <= 0.0):
                exit_triggered = True
                exit_reason = 'Mean_Reversion'
            # Check Exit 2: Stop Loss
            elif (position == 1 and z_curr <= -stop_z) or (position == -1 and z_curr >= stop_z):
                exit_triggered = True
                exit_reason = 'Stop_Loss'
            # Check Exit 3: Max Holding Duration
            elif holding_days >= max_holding_days:
                exit_triggered = True
                exit_reason = 'Max_Duration'

        if exit_triggered:
            # Apply transaction cost on exit
            daily_pnl -= transaction_cost
            trade_return = (portfolio_value * (1 + daily_pnl) - entry_portfolio_val) / entry_portfolio_val
            trades.append({
                'Stock_Y': stock_y,
                'Stock_X': stock_x,
                'Entry_Date': entry_date,
                'Exit_Date': date,
                'Position': 'Long' if position == 1 else 'Short',
                'Entry_Z': entry_z_val,
                'Exit_Z': z_curr,
                'Entry_Spread': entry_spread,
                'Exit_Spread': spread_curr,
                'Return': trade_return,
                'Holding_Days': holding_days,
                'Exit_Reason': exit_reason
            })
            position = 0
            holding_days = 0
        elif position == 0:
            # Check Entry conditions
            if z_curr <= -entry_z:
                position = 1
                entry_date = date
                entry_z_val = z_curr
                entry_spread = spread_curr
                entry_portfolio_val = portfolio_value
                holding_days = 0
                daily_pnl -= transaction_cost  # Apply transaction cost on entry
            elif z_curr >= entry_z:
                position = -1
                entry_date = date
                entry_z_val = z_curr
                entry_spread = spread_curr
                entry_portfolio_val = portfolio_value
                holding_days = 0
                daily_pnl -= transaction_cost  # Apply transaction cost on entry

        portfolio_value *= (1.0 + daily_pnl)
        equity.append(portfolio_value)
        positions_series[step] = position

    equity_series = pd.Series(equity, index=eval_idx)
    trades_df = pd.DataFrame(trades)
    df_signals['Position'] = positions_series
    df_signals['Equity'] = equity_series

    metrics = calculate_performance_metrics(equity_series, trades_df)

    return df_signals, trades_df, metrics


def sweep_entry_deviations(
    prices: pd.DataFrame,
    stock_y: str,
    stock_x: str,
    vasicek_params: dict = None,
    deviations: list = [1.0, 1.5, 2.0, 2.5, 3.0],
    stop_z: float = 3.5,
    max_holding_days: int = 20,
    train_ratio: float = 0.7,
    use_out_of_sample_only: bool = True
) -> pd.DataFrame:
    """
    Evaluates strategy performance across different entry deviation thresholds.
    """
    sweep_results = []

    for dev in deviations:
        _, trades_df, metrics = run_pair_backtest(
            prices=prices,
            stock_y=stock_y,
            stock_x=stock_x,
            vasicek_params=vasicek_params,
            entry_z=dev,
            stop_z=stop_z,
            max_holding_days=max_holding_days,
            train_ratio=train_ratio,
            use_out_of_sample_only=use_out_of_sample_only
        )
        sweep_results.append({
            'Entry_Dev_Z': dev,
            'Total_Return_%': metrics['Total_Return'] * 100,
            'Sharpe_Ratio': metrics['Sharpe_Ratio'],
            'Max_Drawdown_%': metrics['Max_Drawdown'] * 100,
            'Total_Trades': metrics['Total_Trades'],
            'Win_Rate_%': metrics['Win_Rate'] * 100,
            'Avg_Holding_Days': metrics['Avg_Holding_Days'],
            'Profit_Factor': metrics['Profit_Factor']
        })

    return pd.DataFrame(sweep_results)


def calculate_adf_weights(selected_pairs_df: pd.DataFrame) -> dict:
    """
    Computes normalized capital weights proportional to the magnitude of the ADF stationarity statistic:
    w_i = |ADF_Stat_i| / sum(|ADF_Stat_j|)
    """
    if selected_pairs_df.empty:
        return {}

    adf_stats = selected_pairs_df.set_index(['Stock_Y', 'Stock_X'])['ADF_Stat'].abs()
    total_stat = adf_stats.sum()
    if total_stat == 0:
        weights = {pair: 1.0 / len(adf_stats) for pair in adf_stats.index}
    else:
        weights = (adf_stats / total_stat).to_dict()
    return weights


def fetch_spx_benchmark(price_index: pd.DatetimeIndex, ticker: str = '^GSPC') -> pd.Series:
    """
    Fetches S&P 500 benchmark prices matching the exact datetime index of the price matrix.
    Normalizes equity curve starting at 1.0.
    """
    start_date = price_index[0].strftime('%Y-%m-%d')
    end_date = (price_index[-1] + pd.Timedelta(days=5)).strftime('%Y-%m-%d')

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Close' in df.columns.get_level_values(0):
                spx = df['Close'].iloc[:, 0]
            else:
                spx = df.iloc[:, 0]
        elif 'Close' in df.columns:
            spx = df['Close']
        elif 'Adj Close' in df.columns:
            spx = df['Adj Close']
        else:
            spx = df.iloc[:, 0]

        spx = spx.reindex(price_index).ffill().bfill()
        spx_normalized = spx / spx.iloc[0]
        return spx_normalized
    except Exception as e:
        print(f"Warning: Could not fetch {ticker} from yfinance ({e}). Using fallback benchmark.")
        return pd.Series(1.0, index=price_index)



def run_multi_pair_portfolio_backtest(
    prices: pd.DataFrame,
    selected_pairs_df: pd.DataFrame,
    entry_z: float = 2.0,
    stop_z: float = 3.5,
    max_holding_days: int = 20,
    transaction_cost: float = 0.0005,
    rebalance_threshold: float = 0.05,
    train_ratio: float = 0.7,
    use_out_of_sample_only: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Backtests a multi-pair statistical arbitrage portfolio:
      - Fits spreads & Vasicek parameters on In-Sample data (first train_ratio of dates) if use_out_of_sample_only=True
      - Simulates portfolio execution strictly on Out-of-Sample data (remaining 1 - train_ratio of dates)
      - Capital weighting is proportional to the ADF stationarity statistic of each pair.
      - Dynamic rebalancing is triggered ONLY when target weight drift exceeds `rebalance_threshold`
        or when the active position set changes (pair enters/exits).
      - Logs total rebalance events and rebalance transaction costs to ensure cost efficiency.

    Returns: (portfolio_df, all_trades_df, metrics_dict)
    """
    if selected_pairs_df.empty:
        raise ValueError("selected_pairs_df cannot be empty")

    common_idx = prices.index
    split_idx = int(len(common_idx) * train_ratio) if (use_out_of_sample_only and train_ratio < 1.0) else 0
    eval_idx = common_idx[split_idx:] if split_idx > 0 else common_idx[1:]

    # 1. Precompute spreads, Vasicek parameters, and Z-scores for all candidate pairs
    pair_data = {}
    for _, row in selected_pairs_df.iterrows():
        sy, sx = row['Stock_Y'], row['Stock_X']
        pair_key = (sy, sx)

        series_y = prices[sy].dropna()
        series_x = prices[sx].dropna()
        p_idx = series_y.index.intersection(series_x.index)
        s_y = series_y.loc[p_idx]
        s_x = series_x.loc[p_idx]

        if split_idx > 0:
            p_split = int(len(p_idx) * train_ratio)
            s_y_in = s_y.iloc[:p_split]
            s_x_in = s_x.iloc[:p_split]
            spread_in, beta, alpha = calculate_spread(s_y_in, s_x_in)
            vas_params = fit_vasicek_model(spread_in)
            spread = s_y - (beta * s_x + alpha)
        else:
            spread, beta, alpha = calculate_spread(s_y, s_x)
            vas_params = fit_vasicek_model(spread)

        theta = vas_params['theta']
        kappa = vas_params['kappa']
        sigma = vas_params['sigma']
        sigma_eq = sigma / np.sqrt(2.0 * max(kappa, 1e-6)) if kappa > 0 else spread.std()

        z_score = (spread - theta) / (sigma_eq + 1e-8)

        # Align with common_idx
        spread = spread.reindex(common_idx).ffill()
        z_score = z_score.reindex(common_idx).ffill()
        s_y = s_y.reindex(common_idx).ffill()
        s_x = s_x.reindex(common_idx).ffill()

        pair_data[pair_key] = {
            'stock_y': sy,
            'stock_x': sx,
            'beta': beta,
            'alpha': alpha,
            'adf_stat': row['ADF_Stat'],
            'adf_pvalue': row['ADF_PValue'],
            'series_y': s_y,
            'series_x': s_x,
            'spread': spread,
            'z_score': z_score,
            'theta': theta,
            'sigma_eq': sigma_eq,
            'position': 0,
            'holding_days': 0,
            'entry_date': None,
            'entry_z': 0.0,
            'entry_spread': 0.0,
            'entry_capital': 0.0
        }

    portfolio_val = 1.0
    portfolio_equity = []

    rebalance_count = 0
    total_rebalance_costs = 0.0
    all_trades = []

    current_active_weights = {}  # pair_key -> float weight

    port_df = pd.DataFrame(index=eval_idx)
    port_df['Equity'] = 1.0
    port_df['Active_Pairs'] = 0
    port_df['Active_Long_Legs'] = 0
    port_df['Active_Short_Legs'] = 0
    port_df['Active_Legs'] = 0
    port_df['Rebalance_Triggered'] = False

    start_loop = split_idx if split_idx > 0 else 1

    for step, i in enumerate(range(start_loop, len(common_idx))):
        date = common_idx[i]

        active_pairs = []
        entry_exit_occurred = False
        daily_spread_returns = {}

        for pair_key, p in pair_data.items():
            sy, sx = p['stock_y'], p['stock_x']
            beta = p['beta']
            pos = p['position']

            s_y_curr, s_y_prev = p['series_y'].iloc[i], p['series_y'].iloc[i-1]
            s_x_curr, s_x_prev = p['series_x'].iloc[i], p['series_x'].iloc[i-1]

            delta_y = s_y_curr - s_y_prev
            delta_x = s_x_curr - s_x_prev
            capital_prev = s_y_prev + abs(beta) * s_x_prev

            if pos == 1:
                spread_ret = (delta_y - beta * delta_x) / capital_prev
            elif pos == -1:
                spread_ret = (-delta_y + beta * delta_x) / capital_prev
            else:
                spread_ret = 0.0

            daily_spread_returns[pair_key] = spread_ret

            z_curr = p['z_score'].iloc[i]
            spread_curr = p['spread'].iloc[i]

            exit_triggered = False
            exit_reason = None

            if pos != 0:
                p['holding_days'] += 1
                if (pos == 1 and z_curr >= 0.0) or (pos == -1 and z_curr <= 0.0):
                    exit_triggered = True
                    exit_reason = 'Mean_Reversion'
                elif (pos == 1 and z_curr <= -stop_z) or (pos == -1 and z_curr >= stop_z):
                    exit_triggered = True
                    exit_reason = 'Stop_Loss'
                elif p['holding_days'] >= max_holding_days:
                    exit_triggered = True
                    exit_reason = 'Max_Duration'

            if exit_triggered:
                capital_entry = p['entry_capital']
                trade_ret = (spread_curr - p['entry_spread']) / (capital_entry + 1e-8) if pos == 1 else (p['entry_spread'] - spread_curr) / (capital_entry + 1e-8)
                trade_ret -= transaction_cost * 2
                all_trades.append({
                    'Stock_Y': sy,
                    'Stock_X': sx,
                    'Entry_Date': p['entry_date'],
                    'Exit_Date': date,
                    'Position': 'Long' if pos == 1 else 'Short',
                    'Entry_Z': p['entry_z'],
                    'Exit_Z': z_curr,
                    'Return': trade_ret,
                    'Holding_Days': p['holding_days'],
                    'Exit_Reason': exit_reason
                })
                p['position'] = 0
                p['holding_days'] = 0
                entry_exit_occurred = True
            elif pos == 0:
                if z_curr <= -entry_z:
                    p['position'] = 1
                    p['entry_date'] = date
                    p['entry_z'] = z_curr
                    p['entry_spread'] = spread_curr
                    p['entry_capital'] = s_y_curr + abs(beta) * s_x_curr
                    p['holding_days'] = 0
                    entry_exit_occurred = True
                elif z_curr >= entry_z:
                    p['position'] = -1
                    p['entry_date'] = date
                    p['entry_z'] = z_curr
                    p['entry_spread'] = spread_curr
                    p['entry_capital'] = s_y_curr + abs(beta) * s_x_curr
                    p['holding_days'] = 0
                    entry_exit_occurred = True

            if p['position'] != 0:
                active_pairs.append(pair_key)

        rebalance_today = False
        rebalance_cost = 0.0

        # Target weight map across all candidate pairs
        active_adf_sum = sum(abs(pair_data[pk]['adf_stat']) for pk in active_pairs) if len(active_pairs) > 0 else 0.0
        target_weights = {
            pk: (abs(pair_data[pk]['adf_stat']) / active_adf_sum if active_adf_sum > 0 and pk in active_pairs else 0.0)
            for pk in pair_data.keys()
        }

        weight_drift = 0.0
        if len(active_pairs) > 0:
            for pk in active_pairs:
                curr_w = current_active_weights.get(pk, 0.0)
                targ_w = target_weights[pk]
                weight_drift = max(weight_drift, abs(targ_w - curr_w))

        if entry_exit_occurred or (len(active_pairs) > 0 and weight_drift >= rebalance_threshold):
            rebalance_today = True
            rebalance_count += 1

            turnover = sum(abs(target_weights[pk] - current_active_weights.get(pk, 0.0)) for pk in pair_data.keys())
            rebalance_cost = turnover * transaction_cost
            total_rebalance_costs += rebalance_cost

            current_active_weights = target_weights.copy()

        daily_port_return = 0.0
        if len(active_pairs) > 0:
            for pk in active_pairs:
                w = current_active_weights.get(pk, 1.0 / len(active_pairs))
                daily_port_return += w * daily_spread_returns[pk]

        if rebalance_today:
            daily_port_return -= rebalance_cost

        portfolio_val *= (1.0 + daily_port_return)
        portfolio_equity.append(portfolio_val)

        long_stocks = set()
        short_stocks = set()
        for pk in active_pairs:
            pos = pair_data[pk]['position']
            sy, sx = pair_data[pk]['stock_y'], pair_data[pk]['stock_x']
            if pos == 1:
                long_stocks.add(sy)
                short_stocks.add(sx)
            elif pos == -1:
                short_stocks.add(sy)
                long_stocks.add(sx)

        n_long = len(long_stocks)
        n_short = len(short_stocks)
        n_legs = n_long + n_short

        port_df.loc[date, 'Equity'] = portfolio_val
        port_df.loc[date, 'Active_Pairs'] = len(active_pairs)
        port_df.loc[date, 'Active_Long_Legs'] = n_long
        port_df.loc[date, 'Active_Short_Legs'] = n_short
        port_df.loc[date, 'Active_Legs'] = n_legs
        port_df.loc[date, 'Rebalance_Triggered'] = rebalance_today

    equity_series = pd.Series(portfolio_equity, index=eval_idx)
    trades_df = pd.DataFrame(all_trades)
    port_df['Equity'] = equity_series

    metrics = calculate_performance_metrics(equity_series, trades_df)
    metrics['Rebalance_Count'] = rebalance_count
    metrics['Total_Rebalance_Costs'] = total_rebalance_costs
    metrics['Total_Active_Pairs_Tested'] = len(selected_pairs_df)

    return port_df, trades_df, metrics


def calculate_portfolio_vs_benchmark_metrics(
    portfolio_equity: pd.Series,
    benchmark_equity: pd.Series,
    trades_df: pd.DataFrame,
    rebalance_count: int = 0,
    total_rebalance_costs: float = 0.0,
    risk_free_rate: float = 0.02
) -> dict:
    """
    Computes comparative performance metrics for Stat-Arb Portfolio vs SPX Benchmark,
    including Sharpe, Beta, Jensen's Alpha, Correlation, Max Drawdown, and Rebalance statistics.
    """
    port_metrics = calculate_performance_metrics(portfolio_equity, trades_df, risk_free_rate)
    bench_metrics = calculate_performance_metrics(benchmark_equity, pd.DataFrame(), risk_free_rate)

    port_returns = portfolio_equity.pct_change().dropna()
    bench_returns = benchmark_equity.pct_change().dropna()

    common_dates = port_returns.index.intersection(bench_returns.index)
    pr = port_returns.loc[common_dates]
    br = bench_returns.loc[common_dates]

    cov_matrix = np.cov(pr, br)
    var_bench = cov_matrix[1, 1]
    cov_port_bench = cov_matrix[0, 1]

    beta = cov_port_bench / (var_bench + 1e-8)
    correlation = np.corrcoef(pr, br)[0, 1]

    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    alpha_ann = (port_metrics['Annualized_Return'] - risk_free_rate) - beta * (bench_metrics['Annualized_Return'] - risk_free_rate)

    comparison_df = pd.DataFrame([
        {
            'Metric': 'Total Return (%)',
            'Stat_Arb_Portfolio': f"{port_metrics['Total_Return']*100:.2f}%",
            'SPX_Benchmark': f"{bench_metrics['Total_Return']*100:.2f}%"
        },
        {
            'Metric': 'Annualized Return (%)',
            'Stat_Arb_Portfolio': f"{port_metrics['Annualized_Return']*100:.2f}%",
            'SPX_Benchmark': f"{bench_metrics['Annualized_Return']*100:.2f}%"
        },
        {
            'Metric': 'Annualized Volatility (%)',
            'Stat_Arb_Portfolio': f"{port_metrics['Annualized_Vol']*100:.2f}%",
            'SPX_Benchmark': f"{bench_metrics['Annualized_Vol']*100:.2f}%"
        },
        {
            'Metric': 'Sharpe Ratio',
            'Stat_Arb_Portfolio': f"{port_metrics['Sharpe_Ratio']:.2f}",
            'SPX_Benchmark': f"{bench_metrics['Sharpe_Ratio']:.2f}"
        },
        {
            'Metric': 'Sortino Ratio',
            'Stat_Arb_Portfolio': f"{port_metrics['Sortino_Ratio']:.2f}",
            'SPX_Benchmark': f"{bench_metrics['Sortino_Ratio']:.2f}"
        },
        {
            'Metric': 'Max Drawdown (%)',
            'Stat_Arb_Portfolio': f"{port_metrics['Max_Drawdown']*100:.2f}%",
            'SPX_Benchmark': f"{bench_metrics['Max_Drawdown']*100:.2f}%"
        },
        {
            'Metric': 'Beta to SPX',
            'Stat_Arb_Portfolio': f"{beta:.3f}",
            'SPX_Benchmark': "1.000"
        },
        {
            'Metric': 'Correlation to SPX',
            'Stat_Arb_Portfolio': f"{correlation:.3f}",
            'SPX_Benchmark': "1.000"
        },
        {
            'Metric': "Jensen's Alpha (%)",
            'Stat_Arb_Portfolio': f"{alpha_ann*100:.2f}%",
            'SPX_Benchmark': "0.00%"
        },
        {
            'Metric': 'Total Rebalances Count',
            'Stat_Arb_Portfolio': f"{rebalance_count}",
            'SPX_Benchmark': "N/A"
        },
        {
            'Metric': 'Rebalance Transaction Costs (%)',
            'Stat_Arb_Portfolio': f"{total_rebalance_costs*100:.3f}%",
            'SPX_Benchmark': "0.00%"
        },
        {
            'Metric': 'Total Trades Completed',
            'Stat_Arb_Portfolio': f"{port_metrics['Total_Trades']}",
            'SPX_Benchmark': "1 (Hold)"
        },
        {
            'Metric': 'Win Rate (%)',
            'Stat_Arb_Portfolio': f"{port_metrics['Win_Rate']*100:.1f}%",
            'SPX_Benchmark': "N/A"
        }
    ])

    summary_dict = {
        'portfolio_metrics': port_metrics,
        'benchmark_metrics': bench_metrics,
        'beta': beta,
        'correlation': correlation,
        'alpha_ann': alpha_ann,
        'comparison_df': comparison_df
    }

    return summary_dict


import numpy as np
import pandas as pd
from src.cointegration import calculate_spread


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
    vasicek_params: dict,
    entry_z: float = 2.0,
    stop_z: float = 3.5,
    max_holding_days: int = 20,
    transaction_cost: float = 0.0005
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Backtests a mean-reversion pair strategy incorporating:
      - Entry at different mean deviations (z <= -entry_z or z >= entry_z)
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

    spread, beta, alpha = calculate_spread(series_y, series_x)
    theta = vasicek_params['theta']
    kappa = vasicek_params['kappa']
    sigma = vasicek_params['sigma']

    sigma_eq = sigma / np.sqrt(2.0 * max(kappa, 1e-6)) if kappa > 0 else spread.std()
    z_score = (spread - theta) / (sigma_eq + 1e-8)

    portfolio_value = 1.0
    equity = [portfolio_value]

    position = 0          # 0: Cash, +1: Long Spread (Buy Y, Sell X), -1: Short Spread (Sell Y, Buy X)
    entry_date = None
    entry_z_val = 0.0
    entry_spread = 0.0
    holding_days = 0

    trades = []
    df_signals = pd.DataFrame(index=common_idx)
    df_signals['Spread'] = spread
    df_signals['Z_Score'] = z_score
    df_signals['Position'] = 0
    positions_series = np.zeros(len(common_idx))

    # Total combined portfolio value invested = Price_Y + beta * Price_X
    combined_portfolio_price = series_y + abs(beta) * series_x

    for i in range(1, len(common_idx)):
        date = common_idx[i]
        prev_date = common_idx[i-1]

        z_curr = z_score.iloc[i]
        spread_curr = spread.iloc[i]

        port_price_curr = combined_portfolio_price.iloc[i]
        port_price_prev = combined_portfolio_price.iloc[i-1]

        # Calculate asset price returns
        ret_y = (series_y.iloc[i] - series_y.iloc[i-1]) / series_y.iloc[i-1]
        ret_x = (series_x.iloc[i] - series_x.iloc[i-1]) / series_x.iloc[i-1]

        # Spread daily return proportional to asset weightings
        if position == 1:
            daily_pnl = ret_y - beta * ret_x
        elif position == -1:
            daily_pnl = -ret_y + beta * ret_x
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
        positions_series[i] = position

    equity_series = pd.Series(equity, index=common_idx)
    trades_df = pd.DataFrame(trades)
    df_signals['Position'] = positions_series
    df_signals['Equity'] = equity_series

    metrics = calculate_performance_metrics(equity_series, trades_df)

    return df_signals, trades_df, metrics


def sweep_entry_deviations(
    prices: pd.DataFrame,
    stock_y: str,
    stock_x: str,
    vasicek_params: dict,
    deviations: list = [1.0, 1.5, 2.0, 2.5, 3.0],
    stop_z: float = 3.5,
    max_holding_days: int = 20
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
            max_holding_days=max_holding_days
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

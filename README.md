# Mean Reversion Statistical Arbitrage Strategy

End-to-end quantitative statistical arbitrage trading strategy built on **Asset Clustering**, **Cointegration Testing**, **Vasicek (Ornstein-Uhlenbeck) SDE Modeling**, **Advanced Exit Rules (Mean Reversion, Stop-Loss, Max Holding Duration)**, and **Out-of-Sample Dickey-Fuller Consistency Analysis**.

---

## 📁 Repository Structure

```
Mean_reversion_strategy/
├── data/
│   └── top_100_prices.csv           # Cached historical daily price data for Top 100 liquid equities
├── src/
│   ├── cluster.py                   # Step 1: Asset Clustering (PCA + Agglomerative/OPTICS)
│   ├── cointegration.py             # Step 2: Cointegration & Dickey-Fuller Stationarity Testing
│   ├── strategy.py                  # Vasicek / Ornstein-Uhlenbeck Model & Signal Generator
│   └── backtest.py                  # Backtest Engine, Deviation Sweeps & Position Exit Rules
├── main.ipynb                       # Interactive Jupyter Notebook pipeline with charts & analytics
├── task_description.txt             # Task requirements
└── README.md                        # Project documentation
```

---

## ⚙️ Strategy Workflow & Core Components

### 1. Step 1: Asset Clustering (`src/cluster.py`)
- **Data Retrieval**: Ingests daily price data for Top 100 liquid equities via `yfinance` (with synthetic dataset fallback).
- **PCA Factor Exposures**: Extracts principal components from standardized daily returns.
- **Hierarchical Correlation Clustering**: Clusters assets using return correlation distance metric $D_{ij} = \sqrt{2(1 - \rho_{ij})}$.

### 2. Step 2: Cointegration & Stationarity Testing (`src/cointegration.py`)
- **Intra-Cluster Pair Screening**: Narrows down pair search space from 4,950 to intra-cluster candidates.
- **Engle-Granger Two-Step Method**: Fits $Y_t = \beta X_t + \alpha + \epsilon_t$ to determine hedge ratio $\beta$ and intercept $\alpha$.
- **Augmented Dickey-Fuller (ADF) Test**: Evaluates residual spread $S_t = Y_t - \beta X_t - \alpha$ for stationarity ($p \le 0.05$).
- **Out-of-Sample Stationarity Consistency**: Evaluates whether cointegrated stationary pairs retain stationarity out-of-sample across train/test splits.

### 3. Vasicek / Ornstein-Uhlenbeck Model (`src/strategy.py`)
- **Continuous SDE**:
  $$dS_t = \kappa (\theta - S_t) dt + \sigma dW_t$$
- **AR(1) Parameter Estimation**:
  $$S_t = a + b S_{t-1} + \epsilon_t$$
  - Reversion Speed ($\kappa$): $\kappa = -\frac{\ln(b)}{\Delta t}$
  - Equilibrium Mean ($\theta$): $\theta = \frac{a}{1 - b}$
  - Volatility ($\sigma$): $\sigma = \sigma_\epsilon \sqrt{\frac{-2 \ln(b)}{\Delta t (1 - b^2)}}$
  - Half-Life ($t_{1/2}$): $t_{1/2} = \frac{\ln(2)}{\kappa}$

### 4. Backtest Engine & Advanced Exit Rules (`src/backtest.py`)
- **Deviation Threshold Sweep**: Tests strategy performance across entry thresholds $z_{entry} \in [1.0, 1.5, 2.0, 2.5, 3.0]$.
- **Exit Rules**:
  - **Mean Reversion Exit**: Closes position when spread returns to Vasicek mean $\theta$ ($z = 0$).
  - **Stop-Loss Exit**: Closes position if spread moves past $z_{stop} = 3.5$ std.
  - **Max Duration Expiry**: Closes position if open longer than $T_{max}$ days (e.g., $2 \times t_{1/2}$).
- **Performance Metrics**: Calculates Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, Profit Factor, and Exit Reason Breakdown.

---

## 🚀 Quickstart

### Running Python Pipeline
```python
from src.cluster import fetch_top_100_data, cluster_assets
from src.cointegration import find_cointegrated_pairs, calculate_spread, evaluate_universe_df_consistency
from src.strategy import fit_vasicek_model
from src.backtest import run_pair_backtest, sweep_entry_deviations

# 1. Fetch & Cluster Data
prices = fetch_top_100_data()
clusters = cluster_assets(prices, n_clusters=10)

# 2. Find Cointegrated Pairs
pairs = find_cointegrated_pairs(prices, cluster_df=clusters)
top_pair = pairs.iloc[0]

# 3. Fit Vasicek Model
spread, beta, alpha = calculate_spread(prices[top_pair['Stock_Y']], prices[top_pair['Stock_X']])
vasicek_params = fit_vasicek_model(spread)

# 4. Run Backtest with Exit Rules
equity_df, trades_df, metrics = run_pair_backtest(
    prices=prices,
    stock_y=top_pair['Stock_Y'],
    stock_x=top_pair['Stock_X'],
    vasicek_params=vasicek_params,
    entry_z=2.0,
    stop_z=3.5,
    max_holding_days=20
)

print("Backtest Summary Metrics:", metrics)
```

### Interactive Notebook
Open `main.ipynb` in VS Code or Jupyter Notebook to view interactive plots for PCA explained variance, cluster projections, cointegration tables, Vasicek Monte Carlo forecasts, entry deviation sweeps, backtest equity curves, and Dickey-Fuller consistency charts.
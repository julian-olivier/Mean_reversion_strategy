# Mean Reversion Statistical Arbitrage Strategy

An end-to-end quantitative statistical arbitrage trading strategy built on **Asset Clustering**, **Cointegration Testing**, **Vasicek (Ornstein-Uhlenbeck) SDE Modeling**, **Advanced Exit Rules (Mean Reversion, Stop-Loss, Max Holding Duration)**, **Out-of-Sample Dickey-Fuller Consistency Analysis**, **Static & Dynamic Rolling Multi-Pair Portfolio Engines**, and **Multi-Interval Horizon Sweeps**.

---

## 📁 Repository Structure

```
Mean_reversion_strategy/
├── data/
│   └── top_100_prices.csv           # Cached historical daily price data for Top 100 liquid U.S. equities (2006–2026)
├── src/
│   ├── cluster.py                   # Step 1: Asset Clustering (PCA Factor Extraction + Agglomerative/OPTICS)
│   ├── cointegration.py             # Step 2: Engle-Granger Cointegration & OOS Dickey-Fuller Stationarity Testing
│   ├── strategy.py                  # Step 3: Vasicek / Ornstein-Uhlenbeck Model & Signal Generator
│   └── backtest.py                  # Steps 4–9: Backtest Engine, Deviation Sweeps, Exit Rules & Portfolio Engines
├── tests/
│   └── test_strategy_repo.py        # Automated test suite (7 comprehensive test modules)
├── main.ipynb                       # Interactive Jupyter Notebook pipeline with charts & analytics (Steps 1–9)
├── codebase_audit_report.md         # Quantitative launch audit report & production readiness assessment
├── task_description.txt             # Task requirements documentation
└── README.md                        # Project documentation
```

---

## ⚙️ Strategy Workflow & Core Components

### 1. Asset Clustering (`src/cluster.py`)
- **Data Ingestion (`fetch_top_100_data`)**: Retrieves daily price data for Top 100 liquid U.S. equities via `yfinance` with local disk caching (`data/top_100_prices.csv`) and synthetic dataset fallback generator (`generate_synthetic_top_100_data`).
- **PCA Return Factors (`extract_features`)**: Standardizes daily return series across assets and extracts principal components capturing systemic market and sector risk drivers.
- **Hierarchical & Density Clustering (`cluster_assets`)**: Clusters assets using the return correlation distance metric $D_{ij} = \sqrt{2(1 - \rho_{ij})}$. Supports both Agglomerative Hierarchical Clustering with average linkage and density-based OPTICS clustering.

### 2. Cointegration & Stationarity Testing (`src/cointegration.py`)
- **Intra-Cluster Candidate Screening (`find_cointegrated_pairs`)**: Restricts pair candidate search to intra-cluster stocks, reducing search space from $\binom{100}{2} = 4,950$ to $\sim 2,200$ pairs.
- **Engle-Granger Two-Step Method (`calculate_spread`)**: Regresses $Y_t = \beta X_t + \alpha + \epsilon_t$ via Ordinary Least Squares (OLS) to derive hedge ratio $\beta$ and intercept $\alpha$. Fully supports negative cointegration slopes ($\beta < 0$) with proportional capital scaling ($S_y + |\beta| S_x$).
- **Augmented Dickey-Fuller (ADF) Test (`test_pair_cointegration`)**: Evaluates residual spread $S_t = Y_t - \beta X_t - \alpha$ for stationarity ($p \le 0.05$).
- **Out-of-Sample Stationarity Consistency (`test_stationarity_consistency`, `evaluate_universe_df_consistency`)**: Measures stationarity retention out-of-sample across train/test splits across the entire asset universe.
- **Distinct Pair Selection (`select_top_distinct_pairs`)**: Selects top cointegrated pairs sorted by ADF statistical significance while preventing asset concentration (no stock reuse).

### 3. Vasicek / Ornstein-Uhlenbeck Model (`src/strategy.py`)
- **Continuous SDE**:
  $$dS_t = \kappa (\theta - S_t) dt + \sigma dW_t$$
- **AR(1) Parameter Estimation (`fit_vasicek_model`)**:
  $$S_t = a + b S_{t-1} + \epsilon_t$$
  - Reversion Speed ($\kappa$): $\kappa = -\frac{\ln(b)}{\Delta t}$
  - Equilibrium Mean ($\theta$): $\theta = \frac{a}{1 - b}$
  - Volatility ($\sigma$): $\sigma = \sigma_\epsilon \sqrt{\frac{-2 \ln(b)}{\Delta t (1 - b^2)}}$
  - Half-Life ($t_{1/2}$): $t_{1/2} = \frac{\ln(2)}{\kappa}$
- **Non-Stationary Safeguards**: Handles weak mean reversion edge cases ($b \ge 1.0$ or $b \le 0.0$) by bounding $\kappa = 10^{-4}$ and setting half-life to $\infty$ to prevent numerical overflow.
- **Monte Carlo Simulation & Signals (`simulate_vasicek`, `generate_vasicek_signals`)**: Simulates continuous spread trajectories and generates Z-score trading signals based on equilibrium volatility $\sigma_{eq} = \frac{\sigma}{\sqrt{2\kappa}}$.

### 4. Backtest Engine, Portfolio Allocation, & Advanced Exit Rules (`src/backtest.py`)
- **Single-Pair Backtest (`run_pair_backtest`)**: Simulates pair execution with transaction friction (5 bps) and strict out-of-sample parameter estimation.
- **Three-Tier Exit Rules**:
  - **Mean Reversion Exit**: Closes position when spread returns to Vasicek mean $\theta$ ($z = 0$).
  - **Stop-Loss Exit**: Closes position if spread breaches $z_{stop} = \pm 3.5$ standard deviations.
  - **Max Duration Expiry**: Closes position if open longer than $T_{max}$ trading days (e.g., 20 days).
- **Entry Threshold Sweep (`sweep_entry_deviations`)**: Evaluates performance metrics across entry thresholds $z_{entry} \in [1.0, 1.5, 2.0, 2.5, 3.0]$.
- **Robust Metrics Engine (`calculate_performance_metrics`)**: Computes Annualized Return, Volatility, Sharpe Ratio, Sortino Ratio (with $10^{-8}$ downside variance guardrail), Max Drawdown, Win Rate, and Profit Factor.
- **Static ADF-Weighted Multi-Pair Portfolio Engine (`run_multi_pair_portfolio_backtest`)**: Scales strategy to a portfolio of distinct cointegrated pairs with capital weighting proportional to ADF stationarity statistics ($w_i \propto |t_{\text{ADF}, i}|$) and dynamic rebalancing when weight drift exceeds $\tau_{rebalance} = 5\%$.
- **Dynamic Rolling Pair Re-Selection Engine (`run_dynamic_multi_pair_portfolio_backtest`)**: Periodically (every $T_{reselect} = 63$ trading days) re-clusters the universe, re-tests cointegration, and rotates capital into top candidate pairs using a rolling lookback ($T_{lookback} = 252$ days). Open positions in dropped pairs are allowed to run to natural exit.
- **SPX Benchmark Comparator (`fetch_spx_benchmark`, `calculate_portfolio_vs_benchmark_metrics`)**: Benchmarks portfolio equity curves strictly out-of-sample against the S&P 500 (`^GSPC`), calculating Beta, Jensen's Alpha, and Correlation.

### 5. Automated Test Suite (`tests/test_strategy_repo.py`)
- Includes 7 automated unit tests covering clustering feature extraction, cointegration testing, Vasicek parameter estimation, in/out-of-sample backtesting, zero-downside Sortino robustness, static multi-pair portfolio execution, and dynamic rolling re-selection.

---

## 📊 Key Empirical Findings

> [!IMPORTANT]
> **Key Finding 1: Stationarity Decays Over Multi-Year Horizons**  
> Cointegration is a dynamic equilibrium, not a static law. Out-of-sample Dickey-Fuller evaluation reveals that static pair selections degrade over time ($\sim 40\%$ stationarity retention over 5+ years). **Dynamic Rolling Pair Re-Selection (`reselect_frequency=63` days) is required for live trading.**

> [!TIP]
> **Key Finding 2: Higher Entry Thresholds Improve Profit Factor**  
> Entry threshold sweeps ($z_{entry} \in [1.0, 3.0]$) demonstrate that higher entry thresholds ($z_{entry} \ge 2.5$) yield higher Profit Factors and fewer stop-loss breaches by filtering noise within the equilibrium band.

> [!NOTE]
> **Key Finding 3: Multi-Interval Re-Evaluation Horizon Sweep**  
> Testing pair re-evaluation intervals (1, 2, 4, 6, 10 years) over a 20-year horizon (2006–2026) proves that shorter re-evaluation windows prevent catastrophic drawdowns caused by uncoupling pairs.

---

## 🚀 Quickstart

### Running Python Pipeline

```python
from src.cluster import fetch_top_100_data, cluster_assets
from src.cointegration import find_cointegrated_pairs, select_top_distinct_pairs, calculate_spread
from src.strategy import fit_vasicek_model
from src.backtest import (
    run_pair_backtest,
    run_multi_pair_portfolio_backtest,
    run_dynamic_multi_pair_portfolio_backtest,
    fetch_spx_benchmark,
    calculate_portfolio_vs_benchmark_metrics
)

# 1. Fetch Price Data & Cluster Assets
prices = fetch_top_100_data()
clusters = cluster_assets(prices, method='agglomerative', n_clusters=10)

# 2. Find Intra-Cluster Cointegrated Pairs
coint_df = find_cointegrated_pairs(prices, cluster_df=clusters, p_value_threshold=0.05)
top_distinct_pairs = select_top_distinct_pairs(coint_df, max_pairs=10, allow_stock_reuse=False)

# 3. Fit Vasicek Model for Top Pair
top_pair = top_distinct_pairs.iloc[0]
spread, beta, alpha = calculate_spread(prices[top_pair['Stock_Y']], prices[top_pair['Stock_X']])
vasicek_params = fit_vasicek_model(spread)

# 4. Single-Pair Backtest with Exit Rules
eq_signals, trades_df, metrics = run_pair_backtest(
    prices=prices,
    stock_y=top_pair['Stock_Y'],
    stock_x=top_pair['Stock_X'],
    vasicek_params=vasicek_params,
    entry_z=2.0,
    stop_z=3.5,
    max_holding_days=20,
    use_out_of_sample_only=True
)
print("Single Pair Metrics:", metrics)

# 5. Dynamic Rolling Multi-Pair Portfolio Backtest
dyn_port_df, dyn_trades_df, dyn_metrics = run_dynamic_multi_pair_portfolio_backtest(
    prices=prices,
    lookback_window=252,
    reselect_frequency=63,
    max_pairs=10,
    entry_z=3.0,
    stop_z=3.5,
    max_holding_days=20,
    use_out_of_sample_only=True
)
print("Dynamic Rolling Portfolio Metrics:", dyn_metrics)
```

### Running Test Suite

To run the automated unit test suite:

```bash
python tests/test_strategy_repo.py
```

---

## 📓 Interactive Notebook

Open [`main.ipynb`](file:///c:/Users/julia/OneDrive/Documents/Mazi%20-%20internship/Next_Gen%20stuff/Mean_reversion_strategy/main.ipynb) to view interactive charts and analytics:
1. **PCA Explained Variance & Asset Clustering Plots**
2. **Cointegration Screening & Vasicek Monte Carlo Forecasts**
3. **Entry Deviation Parameter Sweeps**
4. **Single-Pair & Multi-Pair Out-of-Sample Equity Curves**
5. **Full-Universe Dickey-Fuller Consistency Charts**
6. **Dynamic Rolling Re-Selection vs. Static Portfolio & SPX Benchmark Comparisons**
7. **20-Year Horizon Sweeps across 1, 2, 4, 6, and 10-Year Re-Evaluation Intervals**
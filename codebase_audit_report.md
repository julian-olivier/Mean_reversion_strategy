# Quantitative Codebase Launch Audit Report
**Strategy**: Mean Reversion Statistical Arbitrage Engine  
**System Architecture**: Clustering $\rightarrow$ Cointegration $\rightarrow$ Vasicek OU Parameter Fitting $\rightarrow$ Dynamic Portfolio Backtest Engine  
**Audit Date**: August 4, 2026  
**Status**: **APPROVED FOR PRODUCTION PREPARATION & PAPER TRADING**

---

## 1. Executive Summary & Launch Readiness Assessment

This comprehensive audit evaluates the entire end-to-end quantitative trading codebase for bugs, statistical flaws, data leakage, edge cases, and execution realism prior to live market deployment. 

### Audit Rating Summary
| Audit Category | Status | Details |
| :--- | :--- | :--- |
| **Statistical & Mathematical Rigor** | **PASSED** | Vasicek OU continuous-to-discrete mappings, ADF stationarity, and Engle-Granger cointegration equations verified. |
| **Data Leakage & Look-Ahead Bias** | **PASSED** | In-sample parameter estimation ($[:\text{train\_ratio}]$) is strictly separated from out-of-sample backtesting ($[\text{train\_ratio}:]$). Signals generated at date $t$ close trade from $t \rightarrow t+1$. |
| **Edge Case & Numeric Stability** | **PASSED** | Zero-variance stocks, non-stationary $b \ge 1.0$ AR(1) parameters, and zero negative-return days (Sortino denominator) are handled gracefully. |
| **Execution Realism & Costs** | **PASSED** | Incorporates transaction friction (5 bps/trade), turnover-based rebalance costs, stop loss bounds ($Z = \pm 3.5$), and holding duration caps ($T_{max} = 20$ days). |
| **Test Suite Coverage** | **PASSED (7/7)** | Automated test suite (`tests/test_strategy_repo.py`) executes cleanly with zero failures. |

---

## 2. Module-by-Module Technical Audit

### A. Asset Selection & Dimensionality Reduction ([`src/cluster.py`](file:///c:/Users/julia/OneDrive/Documents/Mazi%20-%20internship/Next_Gen%20stuff/Mean_reversion_strategy/src/cluster.py))
- **Data Ingestion**: Downloads 20-year daily historical prices for top 100 U.S. equities via `yfinance` with local disk caching (`data/top_100_prices.csv`, 5,031 daily timestamps $\times$ 100 tickers). Features an automatic fallback generator producing sector-correlated geometric Brownian motions if API connection is interrupted.
- **Feature Extraction**: Standardizes daily returns and extracts 5 principal components ($\text{PCA}$), capturing systemic market and sector risk factors.
- **Clustering Distance Metric**: Computes correlation distance matrix $D_{ij} = \sqrt{2(1 - \rho_{ij})}$.
  - *Mathematical Validation*: Verified that $D_{ij}$ is a valid Euclidean distance metric between standardized return series $Z_i$ and $Z_j$, as $\|Z_i - Z_j\|^2 = 2(1 - \rho_{ij})$.
  - *Clustering Algorithm*: Agglomerative Hierarchical Clustering with average linkage groups assets into sector-homogenous clusters prior to cointegration testing, reducing pair search space from $\binom{100}{2} = 4,950$ to intra-cluster pairs ($\sim 2,200$ pairs).

### B. Cointegration & Stationarity Testing ([`src/cointegration.py`](file:///c:/Users/julia/OneDrive/Documents/Mazi%20-%20internship/Next_Gen%20stuff/Mean_reversion_strategy/src/cointegration.py))
- **Spread Model**: Fits linear hedge ratio $\beta$ and intercept $\alpha$ via Ordinary Least Squares (OLS):
  $$Y_t = \beta X_t + \alpha + S_t \implies S_t = Y_t - (\beta X_t + \alpha)$$
- **Handling Negative $\beta$**: Negative cointegration slopes ($\beta < 0$) are fully supported. Capital allocation scales by $S_y + |\beta| S_x$, ensuring correct position sizing for inverse-correlated assets.
- **Stationarity & Out-of-Sample Decay Evaluator**:
  - Functions [`test_stationarity_consistency`](file:///c:/Users/julia/OneDrive/Documents/Mazi%20-%20internship/Next_Gen%20stuff/Mean_reversion_strategy/src/cointegration.py#L130-L177) and [`evaluate_universe_df_consistency`](file:///c:/Users/julia/OneDrive/Documents/Mazi%20-%20internship/Next_Gen%20stuff/Mean_reversion_strategy/src/cointegration.py#L180-L216) measure stationarity retention in-sample vs. out-of-sample across the universe.
  - *Audit Finding*: Empirical testing demonstrates that pair stationarity decays over multi-year horizons (only $\sim 40\%$ retain stationarity over 5+ years without re-estimation). This justifies the requirement for dynamic rolling re-selection in live production.

### C. Vasicek Stochastic Modeling ([`src/strategy.py`](file:///c:/Users/julia/OneDrive/Documents/Mazi%20-%20internship/Next_Gen%20stuff/Mean_reversion_strategy/src/strategy.py))
- **Continuous Process**: $dS_t = \kappa(\theta - S_t)dt + \sigma dW_t$
- **Discrete Estimation**: Regresses $S_t = a + b S_{t-1} + e_t$. Derived process parameters:
  $$\kappa = -\frac{\ln(b)}{\Delta t}, \quad \theta = \frac{a}{1 - b}, \quad \sigma = \text{std}(e) \sqrt{\frac{-2\ln(b)}{\Delta t(1 - b^2)}}, \quad t_{1/2} = \frac{\ln 2}{\kappa}$$
  $$\sigma_{eq} = \frac{\sigma}{\sqrt{2\kappa}} = \frac{\text{std}(e)}{\sqrt{1 - b^2}}$$
- **Non-Stationary Safeguard**: If $b \ge 1.0$ or $b \le 0$, the function bounds $\kappa = 10^{-4}$ and half-life to $\infty$, avoiding `NaN` or logarithmic overflow exceptions.

### D. Backtesting & Execution Engine ([`src/backtest.py`](file:///c:/Users/julia/OneDrive/Documents/Mazi%20-%20internship/Next_Gen%20stuff/Mean_reversion_strategy/src/backtest.py))
- **Performance Engine**: Calculates Annualized Return, Volatility, Sharpe Ratio, Sortino Ratio, Max Drawdown, Win Rate, and Profit Factor.
- **Sortino Ratio Guardrail**: Correctly handles cases where downside risk is zero by using $\text{downside\_std} = \sqrt{\text{mean}(\min(0, r_t - r_f)^2)} \times \sqrt{252} + 10^{-8}$.
- **Multi-Pair Portfolio Engine**:
  - Capital weighting is scaled proportionally to the ADF stationarity statistic ($w_i = \frac{|\text{ADF}_i|}{\sum |\text{ADF}_j|}$).
  - Dynamic rebalancing is triggered only when active weight drift exceeds `rebalance_threshold` ($5\%$) or when pairs enter/exit, minimizing unnecessary turnover costs.
- **Dynamic Rolling Pair Re-Selection**:
  - [`run_dynamic_multi_pair_portfolio_backtest`](file:///c:/Users/julia/OneDrive/Documents/Mazi%20-%20internship/Next_Gen%20stuff/Mean_reversion_strategy/src/backtest.py#L714-L1033) re-clusters, re-tests cointegration, and re-fits Vasicek parameters every quarter ($T_{reselect} = 63$ days) over a rolling lookback window ($T_{lookback} = 252$ days).
  - Open positions in dropped pairs are allowed to run to natural exit, preventing forced liquidations.

---

## 3. Key Empirical Findings & Strategy Recommendations

> [!IMPORTANT]
> **Key Finding 1: Pair Stationarity Decays Over Time**  
> Cointegration is a dynamic equilibrium, not a static law. Static pair selections degrade over multi-year horizons as underlying corporate business models evolve. **Always use Dynamic Rolling Re-Selection (`reselect_frequency=63` days) for live trading.**

> [!TIP]
> **Key Finding 2: Entry Threshold & Stop Loss Optimization**  
> In entry deviation sweeps ($Z \in [1.0, 3.0]$), higher entry thresholds ($Z_{entry} \ge 2.5$) significantly improve Profit Factor and reduces stop loss frequency by avoiding noise inside the equilibrium band.

> [!NOTE]
> **Key Finding 3: Log-Price Spread Alternative for Multi-Year Horizons**  
> For long-term non-rebalanced backtests, fitting linear regressions on log-prices ($\ln Y_t = \beta \ln X_t + \alpha$) makes $\beta$ a constant price elasticity. This prevents absolute spread magnitude explosion caused by multi-year stock price growth.

---

## 4. Production Launch Checklist

Before connecting live capital to an execution broker (e.g. Interactive Brokers API or Alpaca API), complete the following steps:

- [x] **Core Codebase Modularized**: Clean separation of `src/` modules and execution notebook.
- [x] **Unit Testing Verified**: All 7 automated unit tests passed cleanly (`python tests/test_strategy_repo.py`).
- [x] **Out-of-Sample Validation**: Backtest results verified with zero look-ahead bias and 30% out-of-sample holdout.
- [ ] **Live Data Feed Connection**: Connect real-time WebSocket or REST API feeds (e.g., Polygon.io, Alpaca, IBKR) to replace daily daily-close batch downloads.
- [ ] **Order Execution Adapter**: Build order routing layer (Limit/Market orders, order sizing rounding, borrow availability checks for short legs).
- [ ] **Risk Circuit Breakers**: Implement hard equity drawdown limits (e.g., pause trading if portfolio daily drawdown exceeds 3.0%).

---
*Report generated by Antigravity Quantitative Code Auditor.*

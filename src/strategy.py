import numpy as np
import pandas as pd
import statsmodels.api as sm


def Ornstein_Uhlenbeck_Process(
    long_term_mean: float,
    reversion_speed: float,
    vol: float,
    wiener_process: float,
    state: float,
    dt: float = 1.0,
):
    """
    Continuous-time increment for the Vasicek / Ornstein-Uhlenbeck Process:
    dX_t = kappa * (theta - X_t) * dt + sigma * dW_t
    """
    return reversion_speed * (long_term_mean - state) * dt + vol * wiener_process


def fit_vasicek_model(spread: pd.Series, dt: float = 1.0) -> dict:
    """
    Fits the Vasicek (Ornstein-Uhlenbeck) model to a mean-reverting spread series using OLS regression.

    Discrete AR(1) representation:
        S_t = a + b * S_{t-1} + e_t

    Derived Vasicek parameters:
        kappa = -ln(b) / dt              (speed of mean reversion)
        theta = a / (1 - b)               (long-term mean level)
        sigma = std(e) * sqrt(-2*ln(b) / (dt * (1 - b^2)))  (volatility)
        half_life = ln(2) / kappa         (half-life of mean reversion in time steps)

    Returns dictionary containing parameters and model statistics.
    """
    s_clean = spread.dropna()
    s_lag = s_clean.shift(1).dropna()
    s_curr = s_clean.iloc[1:]

    # OLS Regression: S_t ~ a + b * S_{t-1}
    X = sm.add_constant(s_lag)
    model = sm.OLS(s_curr, X).fit()

    a = model.params.iloc[0]
    b = model.params.iloc[1]
    residuals = model.resid
    res_std = residuals.std()

    if b >= 1.0 or b <= 0.0:
        # Handle non-stationary / weak mean reversion edge case
        kappa = 1e-4
        theta = s_clean.mean()
        sigma = res_std
        half_life = np.inf
    else:
        kappa = -np.log(b) / dt
        theta = a / (1.0 - b)
        sigma = res_std * np.sqrt(-2.0 * np.log(b) / (dt * (1.0 - b**2)))
        half_life = np.log(2.0) / kappa

    return {
        'kappa': kappa,
        'theta': theta,
        'sigma': sigma,
        'half_life': half_life,
        'ar1_a': a,
        'ar1_b': b,
        'residual_std': res_std,
        'r_squared': model.rsquared
    }


def simulate_vasicek(
    theta: float,
    kappa: float,
    sigma: float,
    state_0: float,
    dt: float = 1.0,
    n_steps: int = 252,
    seed: int = None
) -> pd.Series:
    """
    Simulates a trajectory of the Vasicek / Ornstein-Uhlenbeck process.
    """
    if seed is not None:
        np.random.seed(seed)

    values = np.zeros(n_steps)
    values[0] = state_0

    for t in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))
        dx = Ornstein_Uhlenbeck_Process(
            long_term_mean=theta,
            reversion_speed=kappa,
            vol=sigma,
            wiener_process=dW,
            state=values[t-1],
            dt=dt
        )
        values[t] = values[t-1] + dx

    return pd.Series(values)


def generate_vasicek_signals(
    spread: pd.Series,
    vasicek_params: dict,
    entry_z: float = 2.0,
    exit_z: float = 0.0
) -> pd.DataFrame:
    """
    Generates trading signals for a cointegrated pair based on Vasicek equilibrium mean and volatility.

    Signals:
      +1 : Long Spread (Spread is below lower entry threshold theta - entry_z * sigma_eq)
      -1 : Short Spread (Spread is above upper entry threshold theta + entry_z * sigma_eq)
       0 : Exit / Neutral position (Spread reverts back to equilibrium mean theta)
    """
    theta = vasicek_params['theta']
    kappa = vasicek_params['kappa']
    sigma = vasicek_params['sigma']

    # Equilibrium asymptotic volatility of the OU process: sigma / sqrt(2 * kappa)
    sigma_eq = sigma / np.sqrt(2.0 * max(kappa, 1e-6)) if kappa > 0 else spread.std()

    z_score = (spread - theta) / (sigma_eq + 1e-8)

    signals = pd.DataFrame(index=spread.index)
    signals['Spread'] = spread
    signals['Z_Score'] = z_score
    signals['Upper_Entry'] = theta + entry_z * sigma_eq
    signals['Lower_Entry'] = theta - entry_z * sigma_eq
    signals['Equilibrium_Mean'] = theta
    signals['Position'] = 0

    current_pos = 0
    pos_series = np.zeros(len(spread))

    for i in range(len(spread)):
        z = z_score.iloc[i]
        if np.isnan(z):
            pos_series[i] = 0
            continue

        if current_pos == 0:
            if z <= -entry_z:
                current_pos = 1   # Buy spread (Long Y, Short X)
            elif z >= entry_z:
                current_pos = -1  # Sell spread (Short Y, Long X)
        elif current_pos == 1:
            if z >= exit_z:
                current_pos = 0   # Exit long position
        elif current_pos == -1:
            if z <= exit_z:
                current_pos = 0   # Exit short position

        pos_series[i] = current_pos

    signals['Position'] = pos_series
    return signals

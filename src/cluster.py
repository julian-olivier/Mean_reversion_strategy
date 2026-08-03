import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, OPTICS
import yfinance as yf

# Clean Top 100 U.S. Equities tickers
DEFAULT_TOP_100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'UNH', 'JNJ', 'JPM',
    'V', 'XOM', 'WMT', 'MA', 'PG', 'HD', 'COST', 'ABBV', 'MRK', 'ORCL',
    'CVX', 'BAC', 'KO', 'PEP', 'NFLX', 'AMD', 'TMO', 'LIN', 'DIS', 'CSCO',
    'ACN', 'ABT', 'PM', 'INTU', 'DHR', 'QCOM', 'CAT', 'WFC', 'TXN', 'GE',
    'IBM', 'AMAT', 'UNP', 'NOW', 'AMGN', 'ISRG', 'SPGI', 'HON', 'COP', 'BKNG',
    'GS', 'LOW', 'BA', 'SYK', 'TJX', 'BLK', 'VZ', 'SBUX', 'SCHW', 'DE',
    'REGN', 'MDLZ', 'MS', 'LLY', 'ADBE', 'T', 'BMY', 'LRCX', 'AMT', 'GILD',
    'CB', 'C', 'PFE', 'ADP', 'CI', 'CVS', 'MO', 'NEE', 'BX', 'ZTS',
    'EOG', 'SLB', 'CMCSA', 'SO', 'APH', 'ITW', 'MU', 'KLAC', 'SHW', 'ETN',
    'NKE', 'LMT', 'WM', 'MCD', 'BSX', 'PANW', 'PLTR', 'SNPS', 'CDNS', 'COR'
]


def fetch_top_100_data(
    tickers: list = None,
    period: str = "10y",
    cache_path: str = "data/top_100_prices.csv"
) -> pd.DataFrame:
    """
    Fetch daily closing price data for top 100 tickers.
    Uses local cached file if present and valid, otherwise downloads using yfinance.
    """
    if tickers is None:
        tickers = DEFAULT_TOP_100_TICKERS

    if cache_path and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not df.empty and df.shape[0] > 100 and df.shape[1] > 10:
                print(f"Loaded {df.shape[1]} tickers and {df.shape[0]} dates from cache: {cache_path}")
                return df
        except Exception:
            pass

    print(f"Downloading historical data for {len(tickers)} tickers via yfinance...")
    try:
        raw_data = yf.download(tickers, period=period, progress=False)
        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                prices = raw_data['Adj Close']
            elif 'Close' in raw_data.columns.get_level_values(0):
                prices = raw_data['Close']
            else:
                prices = raw_data.xs(raw_data.columns.levels[0][0], axis=1, level=0)
        else:
            prices = raw_data

        # Clean missing values
        prices = prices.dropna(how='all', axis=1).ffill().bfill().dropna(axis=1)

        if prices.empty or prices.shape[1] < 5:
            raise ValueError("Downloaded data has insufficient columns or rows.")

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            prices.to_csv(cache_path)
            print(f"Cached price data saved to {cache_path}")

        return prices
    except Exception as e:
        print(f"Warning: Could not download live market data ({e}). Using synthetic dataset.")
        return generate_synthetic_top_100_data(tickers, cache_path=cache_path)


def generate_synthetic_top_100_data(tickers: list = None, n_days: int = 504, cache_path: str = None) -> pd.DataFrame:
    """
    Generates synthetic price data with realistic sector correlation clusters.
    """
    if tickers is None:
        tickers = DEFAULT_TOP_100_TICKERS

    np.random.seed(42)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq='B')

    n_tickers = len(tickers)
    n_sectors = 10
    sector_assignments = np.random.randint(0, n_sectors, size=n_tickers)

    sector_returns = np.random.normal(0.0003, 0.012, size=(n_days, n_sectors))
    market_return = np.random.normal(0.0004, 0.01, size=(n_days, 1))

    price_dict = {}
    for idx, ticker in enumerate(tickers):
        sec = sector_assignments[idx]
        stock_ret = 0.5 * market_return[:, 0] + 0.4 * sector_returns[:, sec] + 0.1 * np.random.normal(0, 0.01, size=n_days)
        price_series = 100.0 * np.exp(np.cumsum(stock_ret))
        price_dict[ticker] = price_series

    df = pd.DataFrame(price_dict, index=dates)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_csv(cache_path)
        print(f"Synthetic dataset saved to {cache_path}")

    return df


def extract_features(prices: pd.DataFrame, n_components: int = 5) -> tuple[pd.DataFrame, PCA]:
    """
    Extracts features (daily returns, mean returns, volatility, correlation metric, PCA components)
    for asset clustering.
    """
    returns = prices.pct_change().dropna(how='all')
    returns = returns.fillna(0.0)

    # Standardize returns across time for each stock
    std_returns = returns.std()
    std_returns[std_returns == 0] = 1e-8
    norm_returns = (returns - returns.mean()) / std_returns

    # Transpose so each stock is a sample (row) and each date is a feature (column)
    transposed_returns = norm_returns.T.fillna(0.0)

    n_comp = min(n_components, transposed_returns.shape[0], transposed_returns.shape[1])
    pca = PCA(n_components=n_comp)
    pca_features = pca.fit_transform(transposed_returns.values)

    feature_df = pd.DataFrame(
        pca_features,
        index=prices.columns,
        columns=[f"PC_{i+1}" for i in range(pca_features.shape[1])]
    )
    feature_df['Mean_Return'] = returns.mean()
    feature_df['Volatility'] = returns.std()

    return feature_df, pca


def cluster_assets(
    prices: pd.DataFrame,
    method: str = 'agglomerative',
    n_clusters: int = 10,
    distance_threshold: float = None
) -> pd.DataFrame:
    """
    Runs clustering on top 100 stock price return profiles.

    Methods:
    - 'agglomerative': Hierarchical clustering on return correlation distance matrix.
    - 'optics': Density-based OPTICS clustering on PCA components.

    Returns:
    DataFrame with index = ticker, column 'Cluster'.
    """
    returns = prices.pct_change().dropna(how='all').fillna(0.0)
    corr_matrix = returns.corr().fillna(0.0)

    # Correlation distance matrix: D = sqrt(2 * (1 - correlation))
    dist_matrix = np.sqrt(np.maximum(0, 2 * (1 - corr_matrix.values)))

    feature_df, _ = extract_features(prices, n_components=5)

    if method == 'agglomerative':
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters if distance_threshold is None else None,
            metric='precomputed',
            linkage='average',
            distance_threshold=distance_threshold
        )
        labels = clusterer.fit_predict(dist_matrix)
    elif method == 'optics':
        clusterer = OPTICS(min_samples=3)
        labels = clusterer.fit_predict(feature_df.values)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    result_df = pd.DataFrame({'Cluster': labels}, index=prices.columns)
    return result_df


def get_cluster_groups(cluster_df: pd.DataFrame) -> dict[int, list[str]]:
    """
    Groups tickers by cluster label.
    """
    groups = {}
    for ticker, row in cluster_df.iterrows():
        c_id = int(row['Cluster'])
        if c_id not in groups:
            groups[c_id] = []
        groups[c_id].append(ticker)
    return groups

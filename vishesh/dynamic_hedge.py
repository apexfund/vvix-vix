import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import itertools # We'll use this to create the combinations

# Set plot style
sns.set(style="whitegrid")

# --- 0. Define Metrics Calculation Function (MODIFIED TO RETURN) ---

def calculate_metrics(returns_series):
    """
    Calculates and RETURNS key performance metrics for a returns series.
    """
    metrics = {
        'sharpe': 0.0,
        'annualized_return': 0.0,
        'max_drawdown': 0.0,
        'annualized_volatility': 0.0
    }
    
    if returns_series.empty or returns_series.std() == 0:
        return metrics
        
    trading_days = 252
    
    # Total Return
    total_return = (1 + returns_series).prod() - 1
    
    # Mean Daily Return
    mean_daily_return = returns_series.mean()
    
    # Daily Volatility
    std_daily_return = returns_series.std()
    
    # Annualized Return
    metrics['annualized_return'] = (1 + mean_daily_return) ** trading_days - 1
    
    # Annualized Volatility
    metrics['annualized_volatility'] = std_daily_return * np.sqrt(trading_days)
    
    # Sharpe Ratio (assuming 0 risk-free rate)
    if metrics['annualized_volatility'] != 0:
        metrics['sharpe'] = metrics['annualized_return'] / metrics['annualized_volatility']
    
    # Max Drawdown
    equity_curve = (1 + returns_series).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    return metrics

# --- 1. Define Tickers and Optimal Parameters ---

# Assets to test
target_tickers = ['SPY', 'QQQ', 'IWM', 'DIA']
# Signal and Rotation tickers
other_tickers = ['^VIX', '^VVIX', 'TLT', '^TNX'] # Add ^TNX for 10-yr Yield

all_tickers = target_tickers + other_tickers

# Optimal parameters found in 'full_optimization.py'
optimal_params = {
    'SPY': {'sma': 200, 'vix_win': 63, 'vix_q': 0.95},
    'QQQ': {'sma': 150, 'vix_win': 42, 'vix_q': 0.95},
    'IWM': {'sma': 250, 'vix_win': 42, 'vix_q': 0.95},
    'DIA': {'sma': 250, 'vix_win': 63, 'vix_q': 0.90}
}


# --- 2. Fetch Data (ONCE) ---
print(f"Downloading data for {', '.join(all_tickers)}...")
data_raw = yf.download(all_tickers, period="max")
print("Data download complete.")

# --- 3. Prepare Base Data (ONCE) ---
data = data_raw['Close']
data = data.dropna()

# Rename VIX/TNX columns
data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX', '^TNX': 'TNX'})

# Calculate VIX Signal
data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']

# Calculate returns for all target assets
for ticker in target_tickers + ['TLT']:
    data[f'{ticker}_Returns'] = data[ticker].pct_change()

# Calculate the interest rate (TNX) trend filter
# 50-day SMA is a common short-term trend for rates
data['TNX_SMA'] = data['TNX'].rolling(window=50).mean()

# Drop NaNs from pct_change and SMA
data = data.dropna() 
print("Base data preparation complete.")

# --- 4. Run Asset Rotation Loop ---
print("\n--- Starting Dynamic Hedge Test (using optimal parameters) ---")

results = [] # To store all our results

for ticker in target_tickers:
    
    print(f"\n[Testing Dynamic Hedge] on: {ticker}")
    
    # 1. Get parameters for this asset
    params = optimal_params[ticker]
    sma_window = params['sma']
    vix_window = params['vix_win']
    vix_quantile = params['vix_q']
    
    print(f"    -> Params: SMA={sma_window}, VIX_Win={vix_window}, VIX_Q={vix_quantile}")
    
    # Create a copy for this loop to avoid changing the base data
    data_loop = data.copy()
    
    # 2. Calculate loop-specific rolling data
    data_loop[f'{ticker}_SMA'] = data_loop[ticker].rolling(window=sma_window).mean()
    data_loop['VIX_Quantile'] = data_loop['VVIX_VIX_Ratio'].rolling(window=vix_window).quantile(vix_quantile)
    
    # 3. Drop NaNs created by the rolling windows
    data_loop = data_loop.dropna()
    
    if data_loop.empty:
        print(f"    -> Not enough data for {ticker}. Skipping.")
        continue

    # 4. Apply Final Logic
    # Signal 1: Asset Trend
    is_trend_up = data_loop[ticker].shift(1) > data_loop[f'{ticker}_SMA'].shift(1)
    # Signal 2: Asset Sentiment
    is_not_complacent = data_loop['VVIX_VIX_Ratio'].shift(1) < data_loop['VIX_Quantile'].shift(1)
    # Signal 3: Rate Trend (Hedge Choice)
    # 1 = Rates Rising (use CASH), 0 = Rates Falling (use TLT)
    is_rates_rising = data_loop['TNX'].shift(1) > data_loop['TNX_SMA'].shift(1)

    # Position = 1 means hold Asset, Position = 0 means HEDGE
    data_loop['position'] = np.where(is_trend_up & is_not_complacent, 1, 0)
    
    # 5. Calculate Returns & Metrics (THE DYNAMIC HEDGE)
    
    # Start with default return (Cash)
    data_loop['Strategy_Returns'] = 0.0
    
    # Rule 1: If position is 1, use Asset returns
    data_loop.loc[data_loop['position'] == 1, 'Strategy_Returns'] = data_loop[f'{ticker}_Returns']
    
    # Rule 2: If position is 0 (Hedge) AND rates are FALLING, use TLT returns
    data_loop.loc[(data_loop['position'] == 0) & (is_rates_rising == False), 'Strategy_Returns'] = data_loop['TLT_Returns']
    
    # Rule 3: If position is 0 (Hedge) AND rates are RISING, use 0.0 (already set)
    
    
    metrics_strategy = calculate_metrics(data_loop['Strategy_Returns'])
    metrics_buy_hold = calculate_metrics(data_loop[f'{ticker}_Returns'])
    
    # 6. Store Results
    result_entry = {
        'Asset': ticker,
        'Strategy_Sharpe': metrics_strategy['sharpe'],
        'BH_Sharpe': metrics_buy_hold['sharpe'],
        'Strategy_Return': metrics_strategy['annualized_return'],
        'BH_Return': metrics_buy_hold['annualized_return'],
        'Strategy_Drawdown': metrics_strategy['max_drawdown'],
        'BH_Drawdown': metrics_buy_hold['max_drawdown'],
    }
    results.append(result_entry)

print("\n--- Dynamic Hedge Test Complete ---")

# --- 5. Analyze and Save Results ---

if not results:
    print("No results were generated. Exiting.")
    exit()

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Asset') # Set 'Asset' as the row index

print("\n--- Dynamic Hedge vs. Buy & Hold (BH) Performance ---")
print(results_df.to_string(float_format="%.2f"))

# Save results to file
plot_dir = './gemini_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

save_path = os.path.join(plot_dir, 'dynamic_hedge_results.csv')
results_df.to_csv(save_path)

print(f"\nFull dynamic hedge test results saved to '{save_path}'")
print("\nAll analysis complete.")

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

# --- 1. Define Tickers ---

# Assets to test
target_tickers = ['SPY', 'QQQ', 'IWM', 'DIA']
# Signal tickers
signal_tickers = ['^VIX', '^VVIX']

all_tickers = target_tickers + signal_tickers

# --- 2. Fetch Data (ONCE) ---
print(f"Downloading data for {', '.join(all_tickers)}...")
data_raw = yf.download(all_tickers, period="max")
print("Data download complete.")

# --- 3. Prepare Base Data (ONCE) ---
data = data_raw['Close']
# Drop all rows where any ticker is missing data
data = data.dropna()

# *** THIS IS THE FIX ***
# Rename columns to remove carets for easier access
data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX'})
# *** END OF FIX ***

# Calculate VIX Signal
data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']

# Calculate returns for all target assets
for ticker in target_tickers:
    data[f'{ticker}_Returns'] = data[ticker].pct_change()

# Drop NaNs from pct_change
data = data.dropna() 
print("Base data preparation complete.")

# --- 4. Run Asset Loop ---
print("\n--- Starting Asset Robustness Test ---")

# Define our winning parameters
sma_window = 200
vix_window = 63
vix_quantile = 0.95

results = [] # To store all our results

for ticker in target_tickers:
    
    print(f"\n[Testing Strategy] on: {ticker}")
    
    # Create a copy for this loop to avoid changing the base data
    data_loop = data.copy()
    
    # 1. Calculate loop-specific rolling data
    data_loop[f'{ticker}_SMA'] = data_loop[ticker].rolling(window=sma_window).mean()
    data_loop['VIX_Quantile'] = data_loop['VVIX_VIX_Ratio'].rolling(window=vix_window).quantile(vix_quantile)
    
    # 2. Drop NaNs created by the rolling windows
    data_loop = data_loop.dropna()
    
    if data_loop.empty:
        print(f"    -> Not enough data for {ticker}. Skipping.")
        continue

    # 3. Apply Final Logic
    is_trend_up = data_loop[ticker].shift(1) > data_loop[f'{ticker}_SMA'].shift(1)
    is_not_complacent = data_loop['VVIX_VIX_Ratio'].shift(1) < data_loop['VIX_Quantile'].shift(1)
    
    data_loop['position'] = np.where(is_trend_up & is_not_complacent, 1, 0)
    
    # 4. Calculate Returns & Metrics
    data_loop['Strategy_Returns'] = data_loop[f'{ticker}_Returns'] * data_loop['position']
    
    metrics_strategy = calculate_metrics(data_loop['Strategy_Returns'])
    metrics_buy_hold = calculate_metrics(data_loop[f'{ticker}_Returns'])
    
    # 5. Store Results
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

print("\n--- Robustness Test Complete ---")

# --- 5. Analyze and Save Results ---

if not results:
    print("No results were generated. Exiting.")
    exit()

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Asset') # Set 'Asset' as the row index

print("\n--- Strategy vs. Buy & Hold (BH) Performance ---")
print(results_df.to_string(float_format="%.2f"))

# Save results to file
plot_dir = './gemini_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

save_path = os.path.join(plot_dir, 'asset_robustness_results.csv')
results_df.to_csv(save_path)

print(f"\nFull robustness test results saved to '{save_path}'")
print("\nAll analysis complete.")
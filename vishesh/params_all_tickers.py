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

# --- 1. Define Parameter Ranges and Tickers ---

# Assets to test
target_tickers = ['SPY', 'QQQ', 'IWM', 'DIA']
# Signal tickers
signal_tickers = ['^VIX', '^VVIX']

all_tickers = target_tickers + signal_tickers

# Parameter lists
sma_window_list = [150, 200, 250]
vix_window_list = [21, 42, 63, 84] # 1, 2, 3, 4 months
vix_quantile_list = [0.90, 0.95]

# --- 2. Fetch Data (ONCE) ---
print(f"Downloading data for {', '.join(all_tickers)}...")
data_raw = yf.download(all_tickers, period="max")
print("Data download complete.")

# --- 3. Prepare Base Data (ONCE) ---
data = data_raw['Close']
data = data.dropna()

# Rename VIX columns to remove carets
data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX'})

# Calculate VIX Signal
data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']

# Calculate returns for all target assets
for ticker in target_tickers:
    data[f'{ticker}_Returns'] = data[ticker].pct_change()

# Drop NaNs from pct_change
data = data.dropna() 
print("Base data preparation complete.")

# --- 4. Run Full Optimization Loop ---
print("\n--- Starting Full Optimization Loop (This may take a few minutes) ---")

results = [] # To store all our results

# Create all parameter combinations
param_combinations = list(itertools.product(sma_window_list, vix_window_list, vix_quantile_list))
total_param_runs = len(param_combinations)
total_overall_runs = total_param_runs * len(target_tickers)
run_count = 0

# --- Outer Loop: By Ticker ---
for ticker in target_tickers:
    
    print(f"\n--- Optimizing for: {ticker} ---")
    
    # --- Inner Loop: By Parameter ---
    for i, (sma_window, vix_window, vix_quantile) in enumerate(param_combinations):
        
        run_count += 1
        print(f"[Run {run_count}/{total_overall_runs}] Testing {ticker}: SMA={sma_window}, VIX_Win={vix_window}, VIX_Q={vix_quantile}")
        
        # Create a copy for this loop to avoid changing the base data
        data_loop = data.copy()
        
        # 1. Calculate loop-specific rolling data
        data_loop[f'{ticker}_SMA'] = data_loop[ticker].rolling(window=sma_window).mean()
        data_loop['VIX_Quantile'] = data_loop['VVIX_VIX_Ratio'].rolling(window=vix_window).quantile(vix_quantile)
        
        # 2. Drop NaNs created by the rolling windows
        data_loop = data_loop.dropna()
        
        if data_loop.empty:
            print("    -> Not enough data for this combination. Skipping.")
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
            'SMA_Window': sma_window,
            'VIX_Window': vix_window,
            'VIX_Quantile': vix_quantile,
            'Strategy_Sharpe': metrics_strategy['sharpe'],
            'BH_Sharpe': metrics_buy_hold['sharpe'],
            'Strategy_Return': metrics_strategy['annualized_return'],
            'BH_Return': metrics_buy_hold['annualized_return'],
            'Strategy_Drawdown': metrics_strategy['max_drawdown'],
            'BH_Drawdown': metrics_buy_hold['max_drawdown'],
            'StartDate': data_loop.index[0].date()
        }
        results.append(result_entry)

print("\n--- Full Optimization Complete ---")

# --- 5. Analyze and Save Results ---

if not results:
    print("No results were generated. Exiting.")
    exit()

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)

# Save results to file
plot_dir = './gemini_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

save_path = os.path.join(plot_dir, 'full_optimization_results.csv')
results_df.to_csv(save_path, index=False)
print(f"Full optimization results saved to '{save_path}'")

# --- 6. Find and Print Best Parameters for Each Asset ---

print("\n--- Best Parameters by Asset (Sorted by Strategy_Sharpe) ---")

# Group by Asset, then find the row (idx) with the max Sharpe in each group
best_params_idx = results_df.groupby('Asset')['Strategy_Sharpe'].idxmax()

# Select just those best rows
best_params_df = results_df.loc[best_params_idx]

# Sort by Asset name
best_params_df = best_params_df.sort_values(by='Asset')

# Print the key columns
columns_to_show = [
    'Strategy_Sharpe', 'BH_Sharpe', 
    'SMA_Window', 'VIX_Window', 'VIX_Quantile',
    'Strategy_Return', 'BH_Return',
    'Strategy_Drawdown', 'BH_Drawdown'
]
print(best_params_df.to_string(columns=columns_to_show, float_format="%.2f"))


print("\nAll analysis complete.")
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
        print("    -> No trades or no volatility. Skipping metrics.")
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

# --- 1. Define Parameter Ranges to Test ---

# You can add or remove any numbers from these lists!
# SMA Windows (Long-term Trend)
sma_window_list = [150, 200, 250]

# VIX/VVIX Windows (Short-term Signal)
vix_window_list = [21, 42, 63, 84] # 1, 2, 3, 4 months

# VIX/VVIX Quantiles (Sell Trigger)
vix_quantile_list = [0.90, 0.95]

# --- 2. Fetch Data (ONCE) ---
tickers = ['^VIX', '^VVIX', 'SPY']
print(f"Downloading data for {', '.join(tickers)}...")
data_raw = yf.download(tickers, period="max")
print("Data download complete.")

# --- 3. Prepare Base Data (ONCE) ---
data = data_raw['Close']
data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX', 'SPY': 'SPY'})
data = data.dropna()
data['SPY_Returns'] = data['SPY'].pct_change()
data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']
data = data.dropna() # Drop rows from pct_change
print("Base data preparation complete.")

# --- 4. Run Parameter Tuning Loop ---
print("\n--- Starting Parameter Tuning Loop ---")

results = [] # To store all our results

# Create all combinations
param_combinations = list(itertools.product(sma_window_list, vix_window_list, vix_quantile_list))
total_runs = len(param_combinations)

for i, (sma_window, vix_window, vix_quantile) in enumerate(param_combinations):
    
    print(f"\n[Run {i+1}/{total_runs}] Testing: SMA={sma_window}, VIX_Window={vix_window}, VIX_Quantile={vix_quantile}")
    
    # Create a copy for this loop to avoid changing the base data
    data_loop = data.copy()
    
    # 1. Calculate loop-specific rolling data
    data_loop['SPY_SMA'] = data_loop['SPY'].rolling(window=sma_window).mean()
    data_loop['VIX_Quantile'] = data_loop['VVIX_VIX_Ratio'].rolling(window=vix_window).quantile(vix_quantile)
    
    # 2. Drop NaNs created by the rolling windows
    # The start date will be different for each loop, which is correct
    data_loop = data_loop.dropna()
    
    if data_loop.empty:
        print("    -> Not enough data for this combination. Skipping.")
        continue

    # 3. Apply Final Logic
    is_trend_up = data_loop['SPY'].shift(1) > data_loop['SPY_SMA'].shift(1)
    is_not_complacent = data_loop['VVIX_VIX_Ratio'].shift(1) < data_loop['VIX_Quantile'].shift(1)
    
    data_loop['position'] = np.where(is_trend_up & is_not_complacent, 1, 0)
    
    # 4. Calculate Returns & Metrics
    data_loop['Strategy_Returns'] = data_loop['SPY_Returns'] * data_loop['position']
    
    # Also calculate Buy & Hold for the *exact same period*
    metrics_strategy = calculate_metrics(data_loop['Strategy_Returns'])
    metrics_buy_hold = calculate_metrics(data_loop['SPY_Returns'])
    
    # 5. Store Results
    result_entry = {
        'SMA_Window': sma_window,
        'VIX_Window': vix_window,
        'VIX_Quantile': vix_quantile,
        'Strategy_Sharpe': metrics_strategy['sharpe'],
        'BH_Sharpe': metrics_buy_hold['sharpe'],
        'Strategy_Return': metrics_strategy['annualized_return'],
        'BH_Return': metrics_buy_hold['annualized_return'],
        'Strategy_Drawdown': metrics_strategy['max_drawdown'],
        'BH_Drawdown': metrics_buy_hold['max_drawdown'],
        'StartDate': data_loop.index[0].date() # Good to know
    }
    results.append(result_entry)

print("\n--- Parameter Tuning Complete ---")

# --- 5. Analyze and Save Results ---

if not results:
    print("No results were generated. Exiting.")
    exit()

# Convert results to a DataFrame for analysis
results_df = pd.DataFrame(results)

# Sort by Sharpe Ratio
results_df = results_df.sort_values(by='Strategy_Sharpe', ascending=False)

print("\n--- Top 10 Best Strategies (by Sharpe Ratio) ---")
print(results_df.head(10).to_string()) # .to_string() prints it nicely

# Find the single best strategy
best_strategy = results_df.iloc[0]

print(f"\n--- Best Strategy Details ---")
print(f"SMA Window: {best_strategy['SMA_Window']}")
print(f"VIX Window: {best_strategy['VIX_Window']}")
print(f"VIX Quantile: {best_strategy['VIX_Quantile']}")
print(f"Strategy Sharpe: {best_strategy['Strategy_Sharpe']:.2f} (vs. BH {best_strategy['BH_Sharpe']:.2f})")
print(f"Strategy Return: {best_strategy['Strategy_Return']:.2%} (vs. BH {best_strategy['BH_Return']:.2%})")
print(f"Strategy Drawdown: {best_strategy['Strategy_Drawdown']:.2%} (vs. BH {best_strategy['BH_Drawdown']:.2%})")

# Save results to file
plot_dir = './gemini_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

save_path = os.path.join(plot_dir, 'parameter_tuning_results.csv')
results_df.to_csv(save_path, index=False)

print(f"\nFull tuning results saved to '{save_path}'")
print("\nAll analysis complete.")
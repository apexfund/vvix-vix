import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set plot style
sns.set(style="whitegrid")

# --- 0. Define Metrics Calculation Function ---

def calculate_metrics(returns_series, name="Strategy"):
    """
    Calculates and prints key performance metrics for a returns series.
    """
    if returns_series.empty or returns_series.std() == 0:
        print(f"\n--- {name} Metrics ---")
        print("Not enough data or trades to calculate metrics.")
        return
        
    trading_days = 252
    
    # Total Return
    total_return = (1 + returns_series).prod() - 1
    
    # Mean Daily Return
    mean_daily_return = returns_series.mean()
    
    # Daily Volatility
    std_daily_return = returns_series.std()
    
    # Annualized Return
    annualized_return = (1 + mean_daily_return) ** trading_days - 1
    
    # Annualized Volatility
    annualized_volatility = std_daily_return * np.sqrt(trading_days)
    
    # Sharpe Ratio (assuming 0 risk-free rate)
    sharpe_ratio = annualized_return / annualized_volatility
    
    # Max Drawdown
    equity_curve = (1 + returns_series).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Print Metrics
    print(f"\n--- {name} Metrics ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio (Rf=0): {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")

# --- 1. Fetch Data ---
# Add 'SPY' to the list of tickers
tickers = ['^VIX', '^VVIX', 'SPY']
print(f"Downloading data for {', '.join(tickers)}...")

data_raw = yf.download(tickers, period="max")

if data_raw.empty:
    print("No data downloaded. Check tickers or network connection.")
    exit()
    
print("Data download complete.")

# --- 2. Data Preparation & Feature Engineering ---

# We use 'Close' for indices and 'SPY'
data = data_raw['Close']

# Rename columns for easier access
data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX', 'SPY': 'SPY'})

# CRITICAL: Drop rows where any of the 3 tickers are missing
# This aligns the data to the start date of ^VVIX (the newest)
data = data.dropna()

# Calculate SPY Buy & Hold Returns
data['SPY_Returns'] = data['SPY'].pct_change()

# Calculate the VVIX/VIX Ratio
data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']

# Drop the first row after pct_change() which will be NaN
data = data.dropna()

if data.empty:
    print("Data is empty after processing. Not enough overlapping data.")
    exit()

print("Data preparation complete. Data head:")
print(data.head())

# --- 3. Statistical Analysis & Band Definition ---

print("\n--- Descriptive Statistics ---")
desc_stats = data[['VIX', 'VVIX', 'VVIX_VIX_Ratio']].describe(
    percentiles=[.05, .10, .25, .5, .75, .90, .95]
)
print(desc_stats)

# Define the "Panic" and "Complacency" bands
q_05 = data['VVIX_VIX_Ratio'].quantile(0.05)
q_95 = data['VVIX_VIX_Ratio'].quantile(0.95)
q_50 = data['VVIX_VIX_Ratio'].quantile(0.50)

print(f"\nStrategy Bands Defined:")
print(f"Panic/Buy Threshold (5th Percentile): {q_05:.3f}")
print(f"Complacency/Sell Threshold (95th Percentile): {q_95:.3f}")

# --- 4. Backtesting Strategy ---

# Create signals: 1 = Buy, -1 = Sell, 0 = Hold
data['signal'] = 0
data.loc[data['VVIX_VIX_Ratio'] <= q_05, 'signal'] = 1
data.loc[data['VVIX_VIX_Ratio'] >= q_95, 'signal'] = -1

# Create position: 1 = In Market, 0 = Out of Market
# Use signal to create a temporary column, ffill to hold position
data['signal_temp'] = data['signal'].replace(0, np.nan)
data['position'] = data['signal_temp'].ffill()

# Clean up: Replace -1 (sell) with 0 (cash) and fill initial NaNs
data['position'] = data['position'].replace(-1, 0).fillna(0)

# Calculate strategy returns
# We shift position by 1 to avoid lookahead bias
data['Strategy_Returns'] = data['SPY_Returns'] * data['position'].shift(1)

# Drop the first row after shift
data = data.dropna()

print("\nBacktest complete.")

# --- 5. Calculate & Print Metrics ---
calculate_metrics(data['Strategy_Returns'], name="VVIX/VIX Strategy")
calculate_metrics(data['SPY_Returns'], name="SPY Buy & Hold")

# --- 6. Plotting & Saving ---

# Define the folder name
plot_dir = './gemini_plots'

# Create the folder if it doesn't exist
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
    
# --- Plot 1: Original Time Series Plots ---
fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

axes[0].plot(data.index, data['SPY'], label='SPY Close', color='purple')
axes[0].set_title('SPY Price Over Time', fontsize=16)
axes[0].legend()

axes[1].plot(data.index, data['VVIX_VIX_Ratio'], label='VVIX/VIX Ratio', color='green', alpha=0.8)
axes[1].axhline(q_95, color='red', linestyle='--', label=f'95th Quantile ({q_95:.2f}) - "Complacency Zone"')
axes[1].axhline(q_50, color='purple', linestyle='--', label=f'50th Quantile (Median) ({q_50:.2f})')
axes[1].axhline(q_05, color='darkgreen', linestyle='--', label=f'5th Quantile ({q_05:.2f}) - "Panic Zone"')
axes[1].set_title('VVIX/VIX Ratio Over Time', fontsize=16)
axes[1].legend(loc='upper left')

# Plot trade signals on a third plot
axes[2].plot(data.index, data['position'], label='Strategy Position (1=In, 0=Out)', color='blue')
axes[2].set_title('Strategy Position Over Time', fontsize=16)
axes[2].set_yticks([0, 1])
axes[2].set_xlabel('Date')
axes[2].legend(loc='upper left')

plt.tight_layout()
save_path = os.path.join(plot_dir, 'spy_and_ratio_timeseries.png')
plt.savefig(save_path)
plt.close()
print(f"Time series plots saved as '{save_path}'")

# --- Plot 2: Strategy PnL vs. Buy & Hold ---

# Calculate cumulative equity curves
data['Strategy_Cumulative'] = (1 + data['Strategy_Returns']).cumprod()
data['Buy_Hold_Cumulative'] = (1 + data['SPY_Returns']).cumprod()

plt.figure(figsize=(15, 8))
plt.plot(data['Strategy_Cumulative'], label='VVIX/VIX Strategy', color='blue')
plt.plot(data['Buy_Hold_Cumulative'], label='SPY Buy & Hold', color='orange', alpha=0.8)

plt.title('Strategy vs. Buy & Hold Cumulative Returns', fontsize=16)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.yscale('log') # Use log scale to see relative performance

save_path = os.path.join(plot_dir, 'strategy_vs_buy_hold.png')
plt.savefig(save_path)
plt.close()
print(f"Strategy PnL plot saved as '{save_path}'")

# --- 7. Save "Diff File" ---

# Select key columns for the backtest file
backtest_data = data[['SPY', 'VIX', 'VVIX', 'VVIX_VIX_Ratio', 
                      'signal', 'position', 'SPY_Returns', 
                      'Strategy_Returns', 'Strategy_Cumulative', 'Buy_Hold_Cumulative']]

diff_file_path = os.path.join(plot_dir, 'gemini_backtest_data.csv')
backtest_data.to_csv(diff_file_path)
print(f"Backtest data 'diff file' saved as '{diff_file_path}'")

print("\nAll analysis complete.")
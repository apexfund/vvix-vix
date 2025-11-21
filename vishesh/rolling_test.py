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
    sharpe_ratio = 0.0
    if annualized_volatility != 0:
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
tickers = ['^VIX', '^VVIX', 'SPY']
print(f"Downloading data for {', '.join(tickers)}...")

data_raw = yf.download(tickers, period="max")
print("Data download complete.")

# --- 2. Data Preparation & Feature Engineering ---

data = data_raw['Close']
data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX', 'SPY': 'SPY'})
data = data.dropna()

data['SPY_Returns'] = data['SPY'].pct_change()
data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']
data = data.dropna()

print("Data preparation complete.")

# --- 3. Statistical Analysis & ROLLING Band Definition ---

print("\n--- Defining ROLLING Bands (252-day window) ---")
rolling_window = 252 # 1 trading year

# Calculate rolling quantiles
data['q_05_rolling'] = data['VVIX_VIX_Ratio'].rolling(window=rolling_window).quantile(0.05)
data['q_95_rolling'] = data['VVIX_VIX_Ratio'].rolling(window=rolling_window).quantile(0.95)

# We must drop the first 'rolling_window' rows, as they won't have quantile data
data = data.dropna()

if data.empty:
    print("Data is empty after calculating rolling windows. Try a shorter window or more data.")
    exit()

print(f"Rolling bands defined. Usable data starts at: {data.index[0].date()}")


# --- 4. Backtesting Strategy (with ROLLING Bands) ---

# Create signals: 1 = Buy, -1 = Sell, 0 = Hold
data['signal'] = 0
# SHIFT(1) is critical here to avoid lookahead bias on the day of calculation
data.loc[data['VVIX_VIX_Ratio'].shift(1) <= data['q_05_rolling'].shift(1), 'signal'] = 1
data.loc[data['VVIX_VIX_Ratio'].shift(1) >= data['q_95_rolling'].shift(1), 'signal'] = -1

# Create position: 1 = In Market, 0 = Out of Market
data['signal_temp'] = data['signal'].replace(0, np.nan)
data['position'] = data['signal_temp'].ffill()
data['position'] = data['position'].replace(-1, 0).fillna(0) # Start in cash

# Calculate strategy returns
# Position is already based on yesterday's data (shifted), so no shift needed here
data['Strategy_Returns'] = data['SPY_Returns'] * data['position']

# Drop any remaining NaNs (e.g., from SPY_Returns calculation at start)
data = data.dropna(subset=['Strategy_Returns', 'SPY_Returns'])

print("\nBacktest complete.")

# --- 5. Calculate & Print Metrics ---
calculate_metrics(data['Strategy_Returns'], name="ROLLING VVIX/VIX Strategy")
calculate_metrics(data['SPY_Returns'], name="SPY Buy & Hold (Matched Period)")

# --- 6. Plotting & Saving ---

plot_dir = './gemini_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
    
# --- Plot 1: ROLLING Time Series Plots ---
fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

axes[0].plot(data.index, data['SPY'], label='SPY Close', color='purple')
axes[0].set_title('SPY Price Over Time', fontsize=16)
axes[0].legend()

axes[1].plot(data.index, data['VVIX_VIX_Ratio'], label='VVIX/VIX Ratio', color='green', alpha=0.8)
# Plot the DYNAMIC bands
axes[1].plot(data.index, data['q_95_rolling'], color='red', linestyle='--', label=f'Rolling 95th Quantile - "Complacency"')
axes[1].plot(data.index, data['q_05_rolling'], color='darkgreen', linestyle='--', label=f'Rolling 5th Quantile - "Panic"')
axes[1].set_title('VVIX/VIX Ratio and DYNAMIC Bands', fontsize=16)
axes[1].legend(loc='upper left')

axes[2].plot(data.index, data['position'], label='Strategy Position (1=In, 0=Out)', color='blue')
axes[2].set_title('Strategy Position Over Time', fontsize=16)
axes[2].set_yticks([0, 1])
axes[2].set_xlabel('Date')
axes[2].legend(loc='upper left')

plt.tight_layout()
save_path = os.path.join(plot_dir, 'rolling_spy_and_ratio.png')
plt.savefig(save_path)
plt.close()
print(f"ROLLING time series plots saved as '{save_path}'")

# --- Plot 2: ROLLING Strategy PnL vs. Buy & Hold ---

data['Strategy_Cumulative'] = (1 + data['Strategy_Returns']).cumprod()
data['Buy_Hold_Cumulative'] = (1 + data['SPY_Returns']).cumprod()

plt.figure(figsize=(15, 8))
plt.plot(data['Strategy_Cumulative'], label='ROLLING VVIX/VIX Strategy', color='blue')
plt.plot(data['Buy_Hold_Cumulative'], label='SPY Buy & Hold', color='orange', alpha=0.8)

plt.title('ROLLING Strategy vs. Buy & Hold Cumulative Returns', fontsize=16)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.yscale('log')
    
save_path = os.path.join(plot_dir, 'rolling_strategy_vs_buy_hold.png')
plt.savefig(save_path)
plt.close()
print(f"ROLLING Strategy PnL plot saved as '{save_path}'")

# --- 7. Save "Diff File" ---
diff_file_path = os.path.join(plot_dir, 'rolling_gemini_backtest_data.csv')
data.to_csv(diff_file_path)
print(f"ROLLING Backtest data 'diff file' saved as '{diff_file_path}'")

print("\nAll analysis complete.")
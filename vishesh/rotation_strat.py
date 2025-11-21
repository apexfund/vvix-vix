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
# *** ADD TLT (Bonds) TO THE LIST ***
tickers = ['^VIX', '^VVIX', 'SPY', 'TLT']
print(f"Downloading data for {', '.join(tickers)}...")

data_raw = yf.download(tickers, period="max")
print("Data download complete.")

# --- 2. Data Preparation & Feature Engineering ---

data = data_raw['Close']
data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX', 'SPY': 'SPY', 'TLT': 'TLT'})
data = data.dropna()

# Calculate returns for BOTH assets
data['SPY_Returns'] = data['SPY'].pct_change()
data['TLT_Returns'] = data['TLT'].pct_change()

data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']

# Add SMA 200 Trend Filter
data['SPY_SMA_200'] = data['SPY'].rolling(window=200).mean()

data = data.dropna() # Drop rows from pct_change and SMA_200

print("Data preparation complete.")

# --- 3. Statistical Analysis & ROLLING Band Definition ---

print("\n--- Defining ROLLING Bands (Optimal Parameters) ---")
rolling_window = 63 # VIX Window
vix_quantile = 0.95 # VIX Quantile
sma_window = 200    # SMA Window

data['q_95_rolling'] = data['VVIX_VIX_Ratio'].rolling(window=rolling_window).quantile(vix_quantile)

# We must drop the first 'rolling_window' rows
data = data.dropna()

if data.empty:
    print("Data is empty after calculating rolling windows.")
    exit()

print(f"Rolling bands defined. Usable data starts at: {data.index[0].date()}")


# --- 4. Backtesting Strategy (SPY/TLT Rotation) ---

print("\nRunning Backtest: SPY/TLT Rotation Strategy")

# 1. Define Trend-Up Condition (Master Switch)
is_trend_up = data['SPY'].shift(1) > data['SPY_SMA_200'].shift(1)

# 2. Define Non-Complacent Condition (VIX/VVIX Switch)
is_not_complacent = data['VVIX_VIX_Ratio'].shift(1) < data['q_95_rolling'].shift(1)

# 3. Combine Logic
# We are "in SPY" (position=1) ONLY IF the trend is up AND we are not complacent.
# Otherwise, we are "in TLT" (position=0).
data['position'] = np.where(is_trend_up & is_not_complacent, 1, 0)

# 4. Calculate Strategy Returns (THE KEY CHANGE)
# When position is 1, use SPY returns. When position is 0, use TLT returns.
data['Strategy_Returns'] = np.where(
    data['position'] == 1, 
    data['SPY_Returns'], 
    data['TLT_Returns']
)

data = data.dropna(subset=['Strategy_Returns', 'SPY_Returns'])
print("Backtest complete.")

# --- 5. Calculate & Print Metrics ---
calculate_metrics(data['Strategy_Returns'], name="SPY/TLT Rotation Strategy")
calculate_metrics(data['SPY_Returns'], name="SPY Buy & Hold (Matched Period)")

# --- 6. Plotting & Saving ---

plot_dir = './gemini_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
    
# --- Plot 1: Rotation Logic Time Series Plots (3 panels) ---
fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

# Panel 1: SPY Price and SMA 200
axes[0].plot(data.index, data['SPY'], label='SPY Close', color='purple')
axes[0].plot(data.index, data['SPY_SMA_200'], label='SPY 200-day SMA', color='orange', linestyle='--')
axes[0].set_title('SPY Price and 200-day SMA', fontsize=16)
axes[0].legend()

# Panel 2: VIX Ratio and 63d Band
axes[1].plot(data.index, data['VVIX_VIX_Ratio'], label='VVIX/VIX Ratio', color='green', alpha=0.8)
axes[1].plot(data.index, data['q_95_rolling'], color='red', linestyle='--', label=f'Rolling 63d 95th Quantile - "Sell Signal"')
axes[1].set_title('VVIX/VIX Ratio and DYNAMIC 63-day Band', fontsize=16)
axes[1].legend(loc='upper left')

# Panel 3: Final Combined Position
axes[2].plot(data.index, data['position'], label='Final Position (1=SPY, 0=TLT)', color='blue')
axes[2].set_title('Strategy Position (1=SPY, 0=TLT)', fontsize=16)
axes[2].set_yticks([0, 1])
axes[2].set_xlabel('Date')
axes[2].legend(loc='upper left')

plt.tight_layout()
save_path = os.path.join(plot_dir, 'rotation_spy_and_ratio.png')
plt.savefig(save_path)
plt.close()
print(f"Rotation Logic time series plots saved as '{save_path}'")

# --- Plot 2: Rotation Strategy PnL vs. Buy & Hold ---

data['Strategy_Cumulative'] = (1 + data['Strategy_Returns']).cumprod()
data['Buy_Hold_Cumulative'] = (1 + data['SPY_Returns']).cumprod()

plt.figure(figsize=(15, 8))
plt.plot(data['Strategy_Cumulative'], label='SPY/TLT Rotation Strategy', color='blue')
plt.plot(data['Buy_Hold_Cumulative'], label='SPY Buy & Hold', color='orange', alpha=0.8)

plt.title('SPY/TLT Rotation Strategy vs. Buy & Hold', fontsize=16)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.yscale('log')
    
save_path = os.path.join(plot_dir, 'rotation_strategy_vs_buy_hold.png')
plt.savefig(save_path)
plt.close()
print(f"Rotation Strategy PnL plot saved as '{save_path}'")

# --- 7. Save "Diff File" ---
diff_file_path = os.path.join(plot_dir, 'rotation_gemini_backtest_data.csv')
data.to_csv(diff_file_path)
print(f"Rotation Backtest data 'diff file' saved as '{diff_file_path}'")

print("\nAll analysis complete.")

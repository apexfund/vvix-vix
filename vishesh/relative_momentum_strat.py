import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Set plot style
sns.set(style="whitegrid")

# --- 0. Define Metrics Calculation Function (MODIFIED TO RETURN) ---

def calculate_metrics(returns_series, name="Strategy"):
    """
    Calculates and prints key performance metrics for a returns series.
    """
    metrics = {
        'sharpe': 0.0,
        'annualized_return': 0.0,
        'max_drawdown': 0.0,
        'annualized_volatility': 0.0
    }
    
    if returns_series.empty or returns_series.std() == 0:
        print(f"\n--- {name} Metrics ---")
        print("Not enough data or trades to calculate metrics.")
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
    
    # Print Metrics
    print(f"\n--- {name} Metrics ---")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio (Rf=0): {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    return metrics

# --- 1. Define Tickers and Parameters ---
target_tickers = ['SPY', 'QQQ', 'IWM', 'DIA']
signal_tickers = ['^VIX', '^VVIX']
all_tickers = target_tickers + signal_tickers

# "Master Switch" parameters (from SPY optimization)
sma_window = 200
vix_window = 63
vix_quantile = 0.95

# "Relative Momentum" lookback
momentum_window = 63 # 3-month momentum

# --- 2. Fetch Data (ONCE) ---
print(f"Downloading data for {', '.join(all_tickers)}...")
data_raw = yf.download(all_tickers, period="max")
print("Data download complete.")

# --- 3. Prepare Base Data (ONCE) ---
data = data_raw['Close']
data = data.dropna()
data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX'})

# Calculate VIX Signal
data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']

# Calculate returns for all target assets
for ticker in target_tickers:
    data[f'{ticker}_Returns'] = data[ticker].pct_change()

# --- 4. Calculate Signals ---

# 4.1: "Master Switch" (Absolute Momentum on SPY)
data['SPY_SMA'] = data['SPY'].rolling(window=sma_window).mean()
data['VIX_Quantile'] = data['VVIX_VIX_Ratio'].rolling(window=vix_window).quantile(vix_quantile)

# 4.2: "Relative Momentum" (Asset Strength)
momentum_cols = []
for ticker in target_tickers:
    col_name = f'{ticker}_Momentum'
    data[col_name] = data[ticker].pct_change(periods=momentum_window)
    momentum_cols.append(col_name)

# 4.3: Drop all NaNs from rolling calculations
data = data.dropna()
print("Signal calculation complete.")

# --- 5. Backtesting Strategy (Relative Momentum) ---

print("\nRunning Backtest: Relative Momentum Strategy")

# 5.1: Define "Master Switch" (Risk-On/Off)
# Note: We shift() to avoid lookahead bias
is_trend_up = data['SPY'].shift(1) > data['SPY_SMA'].shift(1)
is_not_complacent = data['VVIX_VIX_Ratio'].shift(1) < data['VIX_Quantile'].shift(1)

data['is_risk_on'] = np.where(is_trend_up & is_not_complacent, 1, 0)

# 5.2: Find the "Winner" for Relative Momentum
# Find the column name of the highest momentum asset each day
data['winner_momentum_col'] = data[momentum_cols].shift(1).idxmax(axis=1) # Shift() to avoid lookahead
# Convert column name (e.g., "SPY_Momentum") to ticker name ("SPY")
data['winner_ticker'] = data['winner_momentum_col'].str.replace('_Momentum', '')

# 5.3: Calculate Strategy Returns (Vectorized)
data['Strategy_Returns'] = 0.0

# This vectorized logic looks for the "is_risk_on" signal AND if the asset was the winner
# and assigns the correct return.
for ticker in target_tickers:
    is_winner = (data['winner_ticker'] == ticker)
    data['Strategy_Returns'] = np.where(
        (data['is_risk_on'] == 1) & (is_winner == True), 
        data[f'{ticker}_Returns'], 
        data['Strategy_Returns']
    )

data = data.dropna(subset=['Strategy_Returns'])
print("Backtest complete.")

# --- 6. Calculate & Print Metrics ---
metrics_strategy = calculate_metrics(data['Strategy_Returns'], name="Relative Momentum Strategy")

# --- Benchmarks ---
metrics_spy = calculate_metrics(data['SPY_Returns'], name="SPY Buy & Hold (Matched Period)")
metrics_qqq = calculate_metrics(data['QQQ_Returns'], name="QQQ Buy & Hold (Matched Period)")


# --- 7. Plotting & Saving ---

plot_dir = './gemini_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
    
# --- Plot 1: Time Series Plots (3 panels) ---
fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)

# Panel 1: SPY Price and SMA
axes[0].plot(data.index, data['SPY'], label='SPY Close', color='purple')
axes[0].plot(data.index, data['SPY_SMA'], label=f'SPY {sma_window}-day SMA', color='orange', linestyle='--')
axes[0].set_title('SPY Price and 200-day SMA', fontsize=16)
axes[0].legend()

# Panel 2: VIX Ratio and Band
axes[1].plot(data.index, data['VVIX_VIX_Ratio'], label='VVIX/VIX Ratio', color='green', alpha=0.8)
axes[1].plot(data.index, data['VIX_Quantile'], color='red', linestyle='--', label=f'Rolling {vix_window}d {vix_quantile} Quantile')
axes[1].set_title('VVIX/VIX Ratio and DYNAMIC Band', fontsize=16)
axes[1].legend(loc='upper left')

# Panel 3: Master "Risk-On" Switch
axes[2].plot(data.index, data['is_risk_on'], label='Master Signal (1=Risk-On, 0=Cash)', color='blue')
axes[2].set_title('Final Strategy Position (Scaled)', fontsize=16)
axes[2].set_yticks([0, 1])
axes[2].set_xlabel('Date')
axes[2].legend(loc='upper left')

plt.tight_layout()
save_path = os.path.join(plot_dir, 'relative_momentum_signals.png')
plt.savefig(save_path)
plt.close()
print(f"Relative Momentum signal plots saved as '{save_path}'")

# --- Plot 2: Relative Momentum PnL vs. Buy & Hold ---

data['Strategy_Cumulative'] = (1 + data['Strategy_Returns']).cumprod()
data['SPY_Cumulative'] = (1 + data['SPY_Returns']).cumprod()
data['QQQ_Cumulative'] = (1 + data['QQQ_Returns']).cumprod()

plt.figure(figsize=(15, 8))
plt.plot(data['Strategy_Cumulative'], label='Relative Momentum Strategy', color='blue')
plt.plot(data['SPY_Cumulative'], label='SPY Buy & Hold', color='orange', alpha=0.7)
plt.plot(data['QQQ_Cumulative'], label='QQQ Buy & Hold', color='green', alpha=0.7)

plt.title('Relative Momentum Strategy vs. Buy & Hold', fontsize=16)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.yscale('log')
    
save_path = os.path.join(plot_dir, 'relative_momentum_vs_buy_hold.png')
plt.savefig(save_path)
plt.close()
print(f"Relative Momentum PnL plot saved as '{save_path}'")

# --- 7. Save "Diff File" ---
diff_file_path = os.path.join(plot_dir, 'relative_momentum_backtest_data.csv')
data.to_csv(diff_file_path)
print(f"Relative Momentum Backtest data 'diff file' saved as '{diff_file_path}'")

print("\nAll analysis complete.")

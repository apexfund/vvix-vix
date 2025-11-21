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
    Returns a dictionary.
    """
    metrics = {
        'name': name,
        'sharpe': 0.0,
        'annualized_return': 0.0,
        'max_drawdown': 0.0,
        'annualized_volatility': 0.0,
        'total_return': 0.0
    }
    
    if returns_series.empty or returns_series.std() == 0:
        print(f"\n--- {name} Metrics ---")
        print("Not enough data or trades to calculate metrics.")
        return metrics
        
    trading_days = 252
    
    # Total Return
    metrics['total_return'] = (1 + returns_series).prod() - 1
    
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
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annualized_return']:.2%}")
    print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
    print(f"Sharpe Ratio (Rf=0): {metrics['sharpe']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    return metrics

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

# --- 3. Define Optimal Parameters for SPY ---
sma_window = 200
vix_window = 63
vix_quantile = 0.95

print(f"\n--- Using Optimal SPY Params: SMA={sma_window}, VIX_Win={vix_window}, VIX_Q={vix_quantile} ---")

data['SPY_SMA'] = data['SPY'].rolling(window=sma_window).mean()
data['VIX_Quantile'] = data['VVIX_VIX_Ratio'].rolling(window=vix_window).quantile(vix_quantile)

# Drop NaNs created by the rolling windows
data = data.dropna()

if data.empty:
    print("Data is empty after calculating rolling windows.")
    exit()

print(f"Usable data starts at: {data.index[0].date()}")


# --- 4. Backtesting Strategy (with Slippage) ---

print("\nRunning Backtest: Slippage & Commission Test")

# Define our cost per trade (round trip)
# 0.0005 = 5 basis points. This simulates slippage + commission
COST_PER_TRADE = 0.0005 

# 1. Define Conditions
is_trend_up = data['SPY'].shift(1) > data['SPY_SMA'].shift(1)
is_not_complacent = data['VVIX_VIX_Ratio'].shift(1) < data['VIX_Quantile'].shift(1)

# 2. Assign Position (0 or 1)
data['position'] = np.where(is_trend_up & is_not_complacent, 1, 0)

# 3. Calculate Trades and Costs
# .diff() finds a change, .abs() makes it positive. 1 = a trade happened
data['trades'] = data['position'].diff().abs()
# Apply cost to each trade
data['trade_costs'] = data['trades'] * COST_PER_TRADE

# 4. Calculate Gross and Net Returns
data['Gross_Strategy_Returns'] = (data['SPY_Returns'] * data['position'])
data['Net_Strategy_Returns'] = data['Gross_Strategy_Returns'] - data['trade_costs']

data = data.dropna(subset=['Net_Strategy_Returns', 'SPY_Returns'])
print("Slippage test complete.")

# --- 5. Calculate & Print Metrics (Gross vs. Net) ---
metrics_gross = calculate_metrics(data['Gross_Strategy_Returns'], name="Gross Strategy (No Costs)")
metrics_net = calculate_metrics(data['Net_Strategy_Returns'], name="Net Strategy (with 5bps Cost)")
metrics_bh = calculate_metrics(data['SPY_Returns'], name="SPY Buy & Hold (Matched Period)")

# --- 6. Monte Carlo (Bootstrapping) Test ---
print("\n--- Running Monte Carlo (Bootstrapping) Test ---")
N_SIMULATIONS = 1000

# Get the Net Returns (with costs) as our basis
net_returns = data['Net_Strategy_Returns']
net_returns_array = net_returns.values
T = len(net_returns_array)

# Create an array to hold all equity curves
sim_curves = np.zeros((T, N_SIMULATIONS))

for i in range(N_SIMULATIONS):
    # Shuffle the returns (this is the "bootstrap")
    shuffled_returns = np.random.permutation(net_returns_array)
    # Build the cumulative equity curve
    sim_curves[:, i] = (1 + shuffled_returns).cumprod()

# --- 7. Plotting & Saving ---
plot_dir = './gemini_plots'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")

# --- Plot 1: Monte Carlo Simulation ---
plt.figure(figsize=(15, 8))

# Plot all 1000 simulations with transparency
plt.plot(sim_curves, 'grey', alpha=0.05)

# Plot the "actual" Net Strategy on top
actual_curve = (1 + net_returns).cumprod()
plt.plot(actual_curve, 'red', linewidth=2, label='Actual Net Strategy')

plt.title(f'Monte Carlo Stress Test ({N_SIMULATIONS} Simulations)', fontsize=16)
plt.ylabel('Cumulative Returns')
plt.xlabel('Trading Days')
plt.legend(loc='upper left')
plt.yscale('log')
    
save_path = os.path.join(plot_dir, 'monte_carlo_bootstrap.png')
plt.savefig(save_path)
plt.close()
print(f"Monte Carlo simulation plot saved as '{save_path}'")

# --- Plot 2: Gross vs. Net vs. B&H Equity ---
data['Gross_Cumulative'] = (1 + data['Gross_Strategy_Returns']).cumprod()
data['Net_Cumulative'] = (1 + data['Net_Strategy_Returns']).cumprod()
data['Buy_Hold_Cumulative'] = (1 + data['SPY_Returns']).cumprod()

plt.figure(figsize=(15, 8))
plt.plot(data['Gross_Cumulative'], label='Gross Strategy (No Costs)', color='blue')
plt.plot(data['Net_Cumulative'], label='Net Strategy (with 5bps Cost)', color='red', linestyle='--')
plt.plot(data['Buy_Hold_Cumulative'], label='SPY Buy & Hold', color='orange', alpha=0.7)

plt.title('Strategy Performance with Trading Costs', fontsize=16)
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.legend(loc='upper left')
plt.yscale('log')
    
save_path = os.path.join(plot_dir, 'slippage_test_vs_buy_hold.png')
plt.savefig(save_path)
plt.close()
print(f"Slippage test PnL plot saved as '{save_path}'")

# --- 8. Save "Diff File" ---
diff_file_path = os.path.join(plot_dir, 'stress_test_backtest_data.csv')
data.to_csv(diff_file_path)
print(f"Stress test Backtest data 'diff file' saved as '{diff_file_path}'")

print("\nAll analysis complete.")

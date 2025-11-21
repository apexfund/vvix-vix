import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os  # <-- ADDED THIS MODULE

# Set plot style
sns.set(style="whitegrid")

# --- 1. Fetch Data ---
# Download all available historical data for VIX and VVIX
print("Downloading VIX and VVIX data...")
tickers = ['^VIX', '^VVIX']
try:
    data_raw = yf.download(tickers, period="max")
    
    if data_raw.empty:
        print("No data downloaded. Check tickers or network connection.")
        exit()
        
    print("Data download complete.")
    
    # --- 2. Data Preparation & Feature Engineering ---
    
    # We use 'Close' for indices, not 'Adj Close'
    data = data_raw['Close']
    
    # Rename columns for easier access
    data = data.rename(columns={'^VIX': 'VIX', '^VVIX': 'VVIX'})
    
    # Calculate the VVIX/VIX Ratio
    data['VVIX_VIX_Ratio'] = data['VVIX'] / data['VIX']
    
    # Drop any rows with missing values (e.g., before VVIX existed)
    data = data.dropna()
    
    if data.empty:
        print("Data is empty after dropping NA. Not enough overlapping data.")
        exit()

    print("Data preparation complete. Data head:")
    print(data.head())
    
    # --- 3. Statistical Analysis ---
    
    # 3.1: Descriptive Statistics
    print("\n--- Descriptive Statistics ---")
    desc_stats = data[['VIX', 'VVIX', 'VVIX_VIX_Ratio']].describe(
        percentiles=[.05, .10, .25, .5, .75, .90, .95]
    )
    print(desc_stats)
    
    # 3.2: Correlation Matrix
    print("\n--- Correlation Matrix ---")
    corr_matrix = data[['VIX', 'VVIX', 'VVIX_VIX_Ratio']].corr()
    print(corr_matrix)

    # --- 4. Plotting Setup ---
    
    # Define the folder name
    plot_dir = './gemini_plots'
    
    # Create the folder if it doesn't exist
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"Created directory: {plot_dir}")
        
    # --- 5. Plotting & Saving ---
    
    # 5.1: Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Correlation Matrix (VIX, VVIX, Ratio)')
    
    # Updated save path
    save_path = os.path.join(plot_dir, 'correlation_heatmap.png')
    plt.savefig(save_path)
    plt.close() # Close plot
    print(f"Correlation heatmap saved as '{save_path}'")


    # 5.2: Time Series Plots (VIX, VVIX, and Ratio)
    q_05 = data['VVIX_VIX_Ratio'].quantile(0.05)
    q_50 = data['VVIX_VIX_Ratio'].quantile(0.50)
    q_95 = data['VVIX_VIX_Ratio'].quantile(0.95)

    fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
    
    axes[0].plot(data.index, data['VIX'], label='VIX', color='blue')
    axes[0].set_title('VIX Index Over Time', fontsize=16)
    axes[0].set_ylabel('VIX Value')
    axes[0].legend()
    
    axes[1].plot(data.index, data['VVIX'], label='VVIX', color='orange')
    axes[1].set_title('VVIX Index Over Time', fontsize=16)
    axes[1].set_ylabel('VVIX Value')
    axes[1].legend()
    
    axes[2].plot(data.index, data['VVIX_VIX_Ratio'], label='VVIX/VIX Ratio', color='green', alpha=0.8)
    axes[2].axhline(q_95, color='red', linestyle='--', label=f'95th Quantile ({q_95:.2f}) - "Complacency Zone"')
    axes[2].axhline(q_50, color='purple', linestyle='--', label=f'50th Quantile (Median) ({q_50:.2f})')
    axes[2].axhline(q_05, color='darkgreen', linestyle='--', label=f'5th Quantile ({q_05:.2f}) - "Panic Zone"')
    axes[2].set_title('VVIX/VIX Ratio Over Time', fontsize=16)
    axes[2].set_ylabel('Ratio Value')
    axes[2].set_xlabel('Date')
    axes[2].legend(loc='upper left')

    plt.tight_layout()
    # Updated save path
    save_path = os.path.join(plot_dir, 'vix_vvix_timeseries_plots.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Time series plots saved as '{save_path}'")

    # 5.3: Analysis Plots (Scatter & Distribution)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.regplot(data=data, x='VIX', y='VVIX', ax=axes[0], 
                scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
    axes[0].set_title('VIX vs. VVIX Relationship', fontsize=16)
    axes[0].set_xlabel('VIX')
    axes[0].set_ylabel('VVIX')
    
    sns.histplot(data=data, x='VVIX_VIX_Ratio', kde=True, ax=axes[1], bins=100)
    axes[1].axvline(q_95, color='red', linestyle='--', label=f'95th Quantile')
    axes[1].axvline(q_50, color='purple', linestyle='--', label=f'50th Quantile')
    axes[1].axvline(q_05, color='darkgreen', linestyle='--', label=f'5th Quantile')
    axes[1].set_title('Distribution of VVIX/VIX Ratio', fontsize=16)
    axes[1].set_xlabel('Ratio Value')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    # Updated save path
    save_path = os.path.join(plot_dir, 'vix_vvix_analysis_plots.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Analysis plots saved as '{save_path}'")

    print("\nAll analysis complete.")

except Exception as e:
    print(f"An error occurred: {e}")
#!/usr/bin/env python3
"""
VVIX/VIX Ratio Hypothesis Analysis Script

This script tests the hypothesis that the relationship between VVIX and VIX 
can signal market confidence and predict future returns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings

def main():
    """
    Main function to run the analysis.
    """
    warnings.filterwarnings('ignore')
    plt.style.use('seaborn-v0_8-darkgrid')
    print("Libraries imported.")

    # 1. Data Collection and Preparation
    print("\n--- 1. Collecting and Preparing Data ---")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)

    tickers = ['^VIX', '^VVIX', 'SPY']
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    data = data.dropna()

    data.rename(columns={'^VIX': 'vix', '^VVIX': 'vvix', 'SPY': 'spy'}, inplace=True)

    data['vvix_vix_ratio'] = data['vvix'] / data['vix']
    data['spy_return'] = data['spy'].pct_change()

    for days in [1, 5, 10, 20]:
        data[f'spy_fwd_return_{days}d'] = data['spy'].pct_change(days).shift(-days)

    data = data.dropna()

    print(f"Data prepared with {len(data)} observations.")
    print(data.head())

    # 2. Plot VIX, VVIX, SPY, and SPY Returns
    print("\n--- 2. Displaying VIX, VVIX, SPY, and SPY Returns Plot ---")
    fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})
    fig1.suptitle('VIX, VVIX, SPY, and SPY Returns Over Time', fontsize=16, fontweight='bold')

    # Top subplot for VIX and VVIX
    ax1.plot(data.index, data['vix'], label='VIX', color='orange', linewidth=2)
    ax1.plot(data.index, data['vvix'], label='VVIX', color='purple', linewidth=2)
    ax1.set_ylabel('Index Value')
    ax1.set_title('VIX and VVIX')
    ax1.legend()
    ax1.grid(True, alpha=0.5)

    # Middle subplot for SPY price and VVIX/VIX Ratio
    ax2.plot(data.index, data['spy'], label='SPY Price', color='green', linewidth=2)
    ax2.set_ylabel('SPY Price ($)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_title('SPY Price and VVIX/VIX Ratio')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.5)

    ax2b = ax2.twinx()
    ax2b.plot(data.index, data['vvix_vix_ratio'], label='VVIX/VIX Ratio', color='blue', linewidth=1, alpha=0.7)
    ax2b.set_ylabel('VVIX/VIX Ratio', color='blue')
    ax2b.tick_params(axis='y', labelcolor='blue')
    ax2b.legend(loc='upper right')

    # Bottom subplot for SPY returns
    ax3.bar(data.index, data['spy_return'], label='SPY Daily Return', color='gray', alpha=0.7)
    ax3.set_ylabel('Daily Return')
    ax3.set_title('SPY Daily Returns')
    ax3.legend()
    ax3.grid(True, alpha=0.5)

    plt.xlabel('Date')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('vix_vvix_spy_overlay.png')
    plt.show()

    # 3. Defining Market Regimes
    print("\n--- 3. Defining Market Regimes ---")
    vix_low_q = data['vix'].quantile(0.25)
    vix_high_q = data['vix'].quantile(0.75)
    vvix_low_q = data['vvix'].quantile(0.25)
    vvix_high_q = data['vvix'].quantile(0.75)

    def get_regime(row):
        vix_is_high = row['vix'] > vix_high_q
        vix_is_low = row['vix'] < vix_low_q
        vvix_is_high = row['vvix'] > vvix_high_q
        vvix_is_low = row['vvix'] < vvix_low_q

        if vvix_is_low and vix_is_high:
            return 'Confident High Vol'
        elif vvix_is_low and vix_is_low:
            return 'Confident Low Vol'
        elif vvix_is_high and vix_is_high:
            return 'Panic High Vol'
        elif vvix_is_high and vix_is_low:
            return 'Panic Low Vol'
        else:
            return 'Neutral'

    data['regime'] = data.apply(get_regime, axis=1)

    print("Regime counts:")
    print(data['regime'].value_counts())

    # 4. Visualizing the Regimes
    print("\n--- 4. Displaying Regime Scatter Plot ---")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data, x='vix', y='vvix', hue='regime', palette='viridis', s=50, alpha=0.7)
    plt.title('VIX vs. VVIX with Market Regimes', fontsize=16)
    plt.xlabel('VIX', fontsize=12)
    plt.ylabel('VVIX', fontsize=12)
    plt.legend(title='Regime')
    plt.savefig('regime_scatterplot.png')
    plt.show()

    # 5. Analyzing Forward Returns per Regime
    print("\n--- 5. Analyzing Forward Returns per Regime ---")
    fwd_return_cols = [f'spy_fwd_return_{d}d' for d in [1, 5, 10, 20]]
    regime_analysis = data.groupby('regime')[fwd_return_cols].mean() * 100

    print("Average Forward Returns per Regime (%):")
    print(regime_analysis)

    fig2, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    fig2.suptitle('Distribution of Forward Returns per Regime', fontsize=18)

    for i, days in enumerate([1, 5, 10, 20]):
        ax = axes[i//2, i%2]
        col = f'spy_fwd_return_{days}d'
        sns.boxplot(data=data, x='regime', y=col, ax=ax, palette='Set2')
        ax.set_title(f'{days}-Day Forward Returns')
        ax.set_xlabel('')
        ax.set_ylabel('Return')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('forward_returns_distribution.png')
    plt.show()

    # 6. Cumulative Returns Simulation
    print("\n--- 6. Simulating and Plotting Cumulative Returns ---")
    plt.figure(figsize=(14, 8))

    for regime in data['regime'].unique():
        if regime == 'Neutral':
            continue
        regime_returns = data[data['regime'] == regime]['spy_return'].dropna()
        cumulative_returns = (1 + regime_returns).cumprod()
        plt.plot(cumulative_returns, label=f'Hold during {regime}')

    benchmark_returns = (1 + data['spy_return']).cumprod()
    plt.plot(benchmark_returns, label='Buy and Hold SPY', color='black', linestyle='--')

    plt.title('Simulated Cumulative Returns by Regime', fontsize=16)
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.savefig('cumulative_returns_simulation.png')
    plt.show()

if __name__ == "__main__":
    main()

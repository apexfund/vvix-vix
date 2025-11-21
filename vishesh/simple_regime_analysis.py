#!/usr/bin/env python3
"""
Simplified Regime Change Analysis for VVIX/VIX Ratio Strategy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def main():
    print("VVIX/VIX Ratio Regime Change Analysis")
    print("=" * 60)
    
    # Data collection
    print("Collecting market data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    tickers = ['^VIX', '^VVIX', 'SPY']
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    print(f"Data collected: {data.shape[0]} days")
    
    # Extract and align data
    vix_close = data['Close']['^VIX'].dropna()
    vvix_close = data['Close']['^VVIX'].dropna()
    spy_close = data['Close']['SPY'].dropna()
    
    # Align all series to common dates
    common_dates = vix_close.index.intersection(vvix_close.index).intersection(spy_close.index)
    
    vix_aligned = vix_close.loc[common_dates]
    vvix_aligned = vvix_close.loc[common_dates]
    spy_aligned = spy_close.loc[common_dates]
    
    # Calculate ratio and returns
    ratio = vvix_aligned / vix_aligned
    spy_returns = spy_aligned.pct_change().dropna()
    
    print(f"Analyzing {len(ratio)} data points...")
    
    # 1. Basic Regime Detection using Rolling Statistics
    print("\n" + "="*60)
    print("1. REGIME DETECTION USING ROLLING STATISTICS")
    print("="*60)
    
    # Calculate rolling statistics
    window = 30
    ratio_rolling_mean = ratio.rolling(window).mean()
    ratio_rolling_std = ratio.rolling(window).std()
    
    # Z-score for regime detection
    ratio_z_scores = (ratio - ratio_rolling_mean) / ratio_rolling_std
    
    # Identify regime changes (when z-score exceeds 2 standard deviations)
    regime_changes = abs(ratio_z_scores) > 2.0
    
    print(f"Number of ratio regime changes detected: {regime_changes.sum()}")
    print(f"Percentage of time in regime change: {regime_changes.sum() / len(regime_changes) * 100:.2f}%")
    
    # 2. Volatility Regime Analysis
    print("\n" + "="*60)
    print("2. VOLATILITY REGIME ANALYSIS")
    print("="*60)
    
    # Calculate rolling volatility
    spy_rolling_vol = spy_returns.rolling(window).std()
    
    # Define volatility regimes based on percentiles
    low_vol_threshold = spy_rolling_vol.quantile(0.33)
    high_vol_threshold = spy_rolling_vol.quantile(0.67)
    
    # Create regime labels
    regimes = pd.Series('Normal', index=spy_returns.index)
    regimes[spy_rolling_vol <= low_vol_threshold] = 'Low Vol'
    regimes[spy_rolling_vol >= high_vol_threshold] = 'High Vol'
    
    print("\nVolatility Regimes:")
    regime_counts = regimes.value_counts()
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(regimes)*100:.1f}%)")
    
    # Analyze ratio behavior in different regimes
    print("\nRatio behavior by volatility regime:")
    for regime in regimes.unique():
        regime_mask = regimes == regime
        # Align indices properly
        common_idx = ratio.index.intersection(regimes.index)
        regime_mask_aligned = regime_mask.loc[common_idx]
        regime_ratio = ratio.loc[common_idx][regime_mask_aligned]
        regime_returns = spy_returns.loc[common_idx][regime_mask_aligned]
        
        print(f"\n{regime} Regime:")
        print(f"  Ratio mean: {regime_ratio.mean():.4f}")
        print(f"  Ratio std: {regime_ratio.std():.4f}")
        print(f"  Returns mean: {regime_returns.mean():.4f} ({regime_returns.mean()*100:.2f}%)")
        print(f"  Returns std: {regime_returns.std():.4f} ({regime_returns.std()*100:.2f}%)")
        print(f"  Correlation: {regime_ratio.corr(regime_returns):.4f}")
        print(f"  Sample size: {len(regime_ratio)}")
    
    # 3. Ratio Level Regime Analysis
    print("\n" + "="*60)
    print("3. RATIO LEVEL REGIME ANALYSIS")
    print("="*60)
    
    # Define ratio regimes based on percentiles
    ratio_low_threshold = ratio.quantile(0.33)
    ratio_high_threshold = ratio.quantile(0.67)
    
    ratio_regimes = pd.Series('Normal', index=ratio.index)
    ratio_regimes[ratio <= ratio_low_threshold] = 'Low Ratio'
    ratio_regimes[ratio >= ratio_high_threshold] = 'High Ratio'
    
    print("\nRatio Level Regimes:")
    ratio_regime_counts = ratio_regimes.value_counts()
    for regime, count in ratio_regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(ratio_regimes)*100:.1f}%)")
    
    # Analyze market behavior in different ratio regimes
    print("\nMarket behavior by ratio regime:")
    for regime in ratio_regimes.unique():
        regime_mask = ratio_regimes == regime
        # Align indices properly
        common_idx = spy_returns.index.intersection(ratio_regimes.index)
        regime_mask_aligned = regime_mask.loc[common_idx]
        regime_returns = spy_returns.loc[common_idx][regime_mask_aligned]
        
        print(f"\n{regime} Regime:")
        print(f"  Returns mean: {regime_returns.mean():.4f} ({regime_returns.mean()*100:.2f}%)")
        print(f"  Returns std: {regime_returns.std():.4f} ({regime_returns.std()*100:.2f}%)")
        print(f"  Sample size: {len(regime_returns)}")
    
    # 4. Regime Transition Analysis
    print("\n" + "="*60)
    print("4. REGIME TRANSITION ANALYSIS")
    print("="*60)
    
    # Find regime transitions
    regime_transitions = regimes != regimes.shift(1)
    transition_dates = regime_transitions[regime_transitions].index
    
    print(f"Number of volatility regime transitions: {len(transition_dates)}")
    
    # Analyze transitions
    transition_analysis = []
    for i, transition_date in enumerate(transition_dates[1:], 1):  # Skip first transition
        prev_regime = regimes.iloc[regimes.index.get_loc(transition_date) - 1]
        curr_regime = regimes.iloc[regimes.index.get_loc(transition_date)]
        
        # Look at ratio behavior around transition
        pre_period_start = transition_date - pd.Timedelta(days=5)
        post_period_end = transition_date + pd.Timedelta(days=5)
        
        pre_ratio = ratio[pre_period_start:transition_date].mean()
        post_ratio = ratio[transition_date:post_period_end].mean()
        
        transition_analysis.append({
            'date': transition_date,
            'from_regime': prev_regime,
            'to_regime': curr_regime,
            'pre_ratio': pre_ratio,
            'post_ratio': post_ratio,
            'ratio_change': post_ratio - pre_ratio
        })
    
    # Summarize transitions
    transition_df = pd.DataFrame(transition_analysis)
    if not transition_df.empty:
        print("\nTransition Summary:")
        transition_summary = transition_df.groupby(['from_regime', 'to_regime']).agg({
            'ratio_change': ['count', 'mean', 'std']
        }).round(4)
        print(transition_summary)
    
    # 5. Create Visualizations
    print("\n" + "="*60)
    print("5. GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create comprehensive regime analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VVIX/VIX Ratio Regime Change Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Ratio with regime changes
    axes[0, 0].plot(ratio.index, ratio.values, alpha=0.7, linewidth=1, label='VVIX/VIX Ratio')
    axes[0, 0].plot(ratio_rolling_mean.index, ratio_rolling_mean.values, color='red', linewidth=2, label='30-day Mean')
    axes[0, 0].fill_between(ratio_rolling_mean.index, 
                           ratio_rolling_mean - 2*ratio_rolling_std, 
                           ratio_rolling_mean + 2*ratio_rolling_std, 
                           alpha=0.2, color='red', label='±2σ Band')
    
    # Highlight regime changes
    regime_change_dates = ratio.index[regime_changes]
    axes[0, 0].scatter(regime_change_dates, ratio[regime_change_dates], 
                      color='red', s=30, alpha=0.8, label='Regime Changes', zorder=5)
    
    axes[0, 0].set_title('VVIX/VIX Ratio with Regime Changes')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Z-scores
    axes[0, 1].plot(ratio_z_scores.index, ratio_z_scores.values, alpha=0.7, linewidth=1)
    axes[0, 1].axhline(y=2, color='red', linestyle='--', alpha=0.8, label='+2σ Threshold')
    axes[0, 1].axhline(y=-2, color='red', linestyle='--', alpha=0.8, label='-2σ Threshold')
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_title('Ratio Z-Scores for Regime Detection')
    axes[0, 1].set_ylabel('Z-Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Volatility regimes
    regime_colors = {'Low Vol': 'green', 'Normal': 'blue', 'High Vol': 'red'}
    for regime in regimes.unique():
        regime_mask = regimes == regime
        # Align indices properly for plotting
        common_idx = ratio.index.intersection(regimes.index)
        regime_mask_aligned = regime_mask.loc[common_idx]
        axes[0, 2].scatter(ratio.loc[common_idx][regime_mask_aligned].index, 
                          ratio.loc[common_idx][regime_mask_aligned].values, 
                          c=regime_colors[regime], alpha=0.6, s=10, label=regime)
    
    axes[0, 2].set_title('Ratio by Volatility Regime')
    axes[0, 2].set_ylabel('VVIX/VIX Ratio')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Ratio level regimes
    ratio_regime_colors = {'Low Ratio': 'green', 'Normal': 'blue', 'High Ratio': 'red'}
    for regime in ratio_regimes.unique():
        regime_mask = ratio_regimes == regime
        # Align indices properly for plotting
        common_idx = spy_returns.index.intersection(ratio_regimes.index)
        regime_mask_aligned = regime_mask.loc[common_idx]
        axes[1, 0].scatter(ratio.loc[common_idx][regime_mask_aligned].index, 
                          spy_returns.loc[common_idx][regime_mask_aligned].values, 
                          c=ratio_regime_colors[regime], alpha=0.6, s=10, label=regime)
    
    axes[1, 0].set_title('Market Returns by Ratio Regime')
    axes[1, 0].set_ylabel('SPY Returns')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Regime transition frequency
    if not transition_df.empty:
        transition_counts = transition_df.groupby(['from_regime', 'to_regime']).size()
        transition_counts.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Regime Transition Frequency')
        axes[1, 1].set_ylabel('Number of Transitions')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Regime duration analysis
    regime_durations = []
    current_regime = regimes.iloc[0]
    current_duration = 1
    
    for i in range(1, len(regimes)):
        if regimes.iloc[i] == current_regime:
            current_duration += 1
        else:
            regime_durations.append(current_duration)
            current_regime = regimes.iloc[i]
            current_duration = 1
    regime_durations.append(current_duration)
    
    axes[1, 2].hist(regime_durations, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Regime Duration Distribution')
    axes[1, 2].set_xlabel('Duration (days)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Summary Statistics
    print("\n" + "="*60)
    print("6. SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nRegime Change Detection:")
    print(f"  Ratio regime changes: {regime_changes.sum()} ({regime_changes.sum()/len(regime_changes)*100:.1f}%)")
    
    print(f"\nVolatility Regimes:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(regimes)*100:.1f}%)")
    
    print(f"\nRatio Level Regimes:")
    for regime, count in ratio_regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(ratio_regimes)*100:.1f}%)")
    
    print(f"\nRegime Transitions:")
    print(f"  Total transitions: {len(transition_dates)}")
    
    if not transition_df.empty:
        print(f"  Average ratio change during transitions: {transition_df['ratio_change'].mean():.4f}")
        print(f"  Std of ratio changes: {transition_df['ratio_change'].std():.4f}")
    
    print("\nRegime change analysis complete! Check the generated plots for visual insights.")

if __name__ == "__main__":
    main()

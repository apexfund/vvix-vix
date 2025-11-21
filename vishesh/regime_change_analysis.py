#!/usr/bin/env python3
"""
Regime Change Analysis for VVIX/VIX Ratio Strategy
Detects market regime changes and analyzes their impact on the VVIX/VIX ratio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def detect_regime_changes(data, window=30, threshold=2.0):
    """
    Detect regime changes using rolling statistics
    """
    # Calculate rolling statistics
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    
    # Z-score for regime change detection
    z_scores = (data - rolling_mean) / rolling_std
    
    # Identify regime changes (when z-score exceeds threshold)
    regime_changes = abs(z_scores) > threshold
    
    return regime_changes, z_scores, rolling_mean, rolling_std

def identify_volatility_regimes(returns, window=30):
    """
    Identify different volatility regimes
    """
    rolling_vol = returns.rolling(window).std()
    
    # Define regimes based on volatility percentiles
    low_vol_threshold = rolling_vol.quantile(0.33)
    high_vol_threshold = rolling_vol.quantile(0.67)
    
    regimes = pd.Series('Normal', index=returns.index)
    regimes[rolling_vol <= low_vol_threshold] = 'Low Vol'
    regimes[rolling_vol >= high_vol_threshold] = 'High Vol'
    
    return regimes, rolling_vol

def analyze_regime_transitions(ratio, returns, regimes):
    """
    Analyze how ratio behaves during regime transitions
    """
    regime_changes = regimes != regimes.shift(1)
    transition_periods = regime_changes[regime_changes].index
    
    results = {}
    
    for i, transition_date in enumerate(transition_periods):
        if i == 0:
            continue
            
        prev_regime = regimes[transition_date - pd.Timedelta(days=1)]
        curr_regime = regimes[transition_date]
        
        # Look at ratio behavior around transition
        pre_period = ratio[transition_date - pd.Timedelta(days=10):transition_date]
        post_period = ratio[transition_date:transition_date + pd.Timedelta(days=10)]
        
        if len(pre_period) > 0 and len(post_period) > 0:
            transition_key = f"{prev_regime} -> {curr_regime}"
            if transition_key not in results:
                results[transition_key] = []
            
            results[transition_key].append({
                'date': transition_date,
                'pre_mean': pre_period.mean(),
                'post_mean': post_period.mean(),
                'ratio_change': post_period.mean() - pre_period.mean(),
                'return_around_transition': returns[transition_date - pd.Timedelta(days=5):transition_date + pd.Timedelta(days=5)].mean()
            })
    
    return results

def main():
    print("VVIX/VIX Ratio Regime Change Analysis")
    print("=" * 60)
    
    # Data collection
    print("Collecting market data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    tickers = ['^VIX', '^VVIX', 'SPY', '^GSPC']
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    print(f"Data collected: {data.shape[0]} days")
    
    # Extract and align data
    vix_close = data['Close']['^VIX'].dropna()
    vvix_close = data['Close']['^VVIX'].dropna()
    spy_close = data['Close']['SPY'].dropna()
    spx_close = data['Close']['^GSPC'].dropna()
    
    common_dates = (vix_close.index.intersection(vvix_close.index)
                   .intersection(spy_close.index)
                   .intersection(spx_close.index))
    
    vix_aligned = vix_close.loc[common_dates]
    vvix_aligned = vvix_close.loc[common_dates]
    spy_aligned = spy_close.loc[common_dates]
    spx_aligned = spx_close.loc[common_dates]
    
    # Calculate ratio and returns
    ratio = vvix_aligned / vix_aligned
    spy_returns = spy_aligned.pct_change().dropna()
    spx_returns = spx_aligned.pct_change().dropna()
    
    print(f"Analyzing {len(ratio)} data points...")
    
    # 1. Regime Change Detection
    print("\n" + "="*60)
    print("1. REGIME CHANGE DETECTION")
    print("="*60)
    
    # Detect regime changes in ratio
    ratio_regime_changes, ratio_z_scores, ratio_rolling_mean, ratio_rolling_std = detect_regime_changes(ratio, window=30, threshold=2.0)
    
    print(f"Number of ratio regime changes detected: {ratio_regime_changes.sum()}")
    print(f"Percentage of time in regime change: {ratio_regime_changes.sum() / len(ratio_regime_changes) * 100:.2f}%")
    
    # Detect regime changes in market returns
    market_regime_changes, market_z_scores, market_rolling_mean, market_rolling_std = detect_regime_changes(spy_returns, window=30, threshold=2.0)
    
    print(f"Number of market regime changes detected: {market_regime_changes.sum()}")
    print(f"Percentage of time in market regime change: {market_regime_changes.sum() / len(market_regime_changes) * 100:.2f}%")
    
    # 2. Volatility Regime Analysis
    print("\n" + "="*60)
    print("2. VOLATILITY REGIME ANALYSIS")
    print("="*60)
    
    # Identify volatility regimes
    spy_regimes, spy_rolling_vol = identify_volatility_regimes(spy_returns, window=30)
    spx_regimes, spx_rolling_vol = identify_volatility_regimes(spx_returns, window=30)
    
    print("\nSPY Volatility Regimes:")
    regime_counts = spy_regimes.value_counts()
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(spy_regimes)*100:.1f}%)")
    
    # Analyze ratio behavior in different regimes
    print("\nRatio behavior by volatility regime:")
    for regime in spy_regimes.unique():
        regime_mask = spy_regimes == regime
        # Align indices properly
        common_idx = ratio.index.intersection(spy_regimes.index)
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
    
    # 3. Regime Transition Analysis
    print("\n" + "="*60)
    print("3. REGIME TRANSITION ANALYSIS")
    print("="*60)
    
    # Analyze transitions between regimes
    transition_results = analyze_regime_transitions(ratio, spy_returns, spy_regimes)
    
    print("\nRegime Transition Analysis:")
    for transition_type, transitions in transition_results.items():
        if len(transitions) > 0:
            ratio_changes = [t['ratio_change'] for t in transitions]
            return_changes = [t['return_around_transition'] for t in transitions]
            
            print(f"\n{transition_type} ({len(transitions)} transitions):")
            print(f"  Average ratio change: {np.mean(ratio_changes):.4f}")
            print(f"  Average return around transition: {np.mean(return_changes):.4f} ({np.mean(return_changes)*100:.2f}%)")
            print(f"  Ratio change std: {np.std(ratio_changes):.4f}")
    
    # 4. Clustering Analysis
    print("\n" + "="*60)
    print("4. CLUSTERING ANALYSIS")
    print("="*60)
    
    # Prepare data for clustering - align all series first
    common_cluster_idx = ratio.index.intersection(spy_returns.index).intersection(spy_rolling_vol.index).intersection(vix_aligned.index).intersection(vvix_aligned.index)
    
    cluster_data = pd.DataFrame({
        'ratio': ratio.loc[common_cluster_idx],
        'ratio_change': ratio.loc[common_cluster_idx].pct_change(),
        'spy_return': spy_returns.loc[common_cluster_idx],
        'spy_volatility': spy_rolling_vol.loc[common_cluster_idx],
        'vix': vix_aligned.loc[common_cluster_idx],
        'vvix': vvix_aligned.loc[common_cluster_idx]
    }).dropna()
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cluster_data[['ratio', 'spy_return', 'spy_volatility', 'vix', 'vvix']])
    
    # Perform K-means clustering
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    cluster_data['cluster'] = cluster_labels
    
    print(f"Clustering into {n_clusters} regimes:")
    for i in range(n_clusters):
        cluster_mask = cluster_data['cluster'] == i
        cluster_subset = cluster_data[cluster_mask]
        
        print(f"\nCluster {i} ({cluster_mask.sum()} observations):")
        print(f"  Ratio mean: {cluster_subset['ratio'].mean():.4f}")
        print(f"  SPY return mean: {cluster_subset['spy_return'].mean():.4f} ({cluster_subset['spy_return'].mean()*100:.2f}%)")
        print(f"  SPY volatility mean: {cluster_subset['spy_volatility'].mean():.4f}")
        print(f"  VIX mean: {cluster_subset['vix'].mean():.4f}")
        print(f"  VVIX mean: {cluster_subset['vvix'].mean():.4f}")
    
    # 5. Create Visualizations
    print("\n" + "="*60)
    print("5. GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create comprehensive regime analysis plots
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('VVIX/VIX Ratio Regime Change Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Ratio with regime changes highlighted
    axes[0, 0].plot(ratio.index, ratio.values, alpha=0.7, linewidth=1, label='VVIX/VIX Ratio')
    axes[0, 0].plot(ratio_rolling_mean.index, ratio_rolling_mean.values, color='red', linewidth=2, label='30-day Mean')
    axes[0, 0].fill_between(ratio_rolling_mean.index, 
                           ratio_rolling_mean - 2*ratio_rolling_std, 
                           ratio_rolling_mean + 2*ratio_rolling_std, 
                           alpha=0.2, color='red', label='±2σ Band')
    
    # Highlight regime changes
    regime_change_dates = ratio.index[ratio_regime_changes]
    axes[0, 0].scatter(regime_change_dates, ratio[regime_change_dates], 
                      color='red', s=50, alpha=0.8, label='Regime Changes', zorder=5)
    
    axes[0, 0].set_title('VVIX/VIX Ratio with Regime Changes')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Z-scores for regime detection
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
    for regime in spy_regimes.unique():
        regime_mask = spy_regimes == regime
        axes[1, 0].scatter(ratio[regime_mask].index, ratio[regime_mask].values, 
                          c=regime_colors[regime], alpha=0.6, s=10, label=regime)
    
    axes[1, 0].set_title('Ratio by Volatility Regime')
    axes[1, 0].set_ylabel('VVIX/VIX Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Regime transition analysis
    transition_types = list(transition_results.keys())
    if transition_types:
        transition_counts = [len(transition_results[t]) for t in transition_types]
        axes[1, 1].bar(transition_types, transition_counts, alpha=0.7)
        axes[1, 1].set_title('Regime Transition Frequency')
        axes[1, 1].set_ylabel('Number of Transitions')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Clustering results
    scatter = axes[2, 0].scatter(cluster_data['ratio'], cluster_data['spy_return'], 
                                c=cluster_data['cluster'], cmap='viridis', alpha=0.6)
    axes[2, 0].set_title('Clustering Results: Ratio vs SPY Returns')
    axes[2, 0].set_xlabel('VVIX/VIX Ratio')
    axes[2, 0].set_ylabel('SPY Returns')
    axes[2, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[2, 0], label='Cluster')
    
    # Plot 6: Regime persistence analysis
    regime_durations = []
    current_regime = spy_regimes.iloc[0]
    current_duration = 1
    
    for i in range(1, len(spy_regimes)):
        if spy_regimes.iloc[i] == current_regime:
            current_duration += 1
        else:
            regime_durations.append(current_duration)
            current_regime = spy_regimes.iloc[i]
            current_duration = 1
    regime_durations.append(current_duration)
    
    axes[2, 1].hist(regime_durations, bins=20, alpha=0.7, edgecolor='black')
    axes[2, 1].set_title('Regime Duration Distribution')
    axes[2, 1].set_xlabel('Duration (days)')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 6. Summary Statistics
    print("\n" + "="*60)
    print("6. SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nRegime Change Detection:")
    print(f"  Ratio regime changes: {ratio_regime_changes.sum()} ({ratio_regime_changes.sum()/len(ratio_regime_changes)*100:.1f}%)")
    print(f"  Market regime changes: {market_regime_changes.sum()} ({market_regime_changes.sum()/len(market_regime_changes)*100:.1f}%)")
    
    print(f"\nVolatility Regimes:")
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(spy_regimes)*100:.1f}%)")
    
    print(f"\nRegime Transitions:")
    total_transitions = sum(len(transitions) for transitions in transition_results.values())
    print(f"  Total transitions analyzed: {total_transitions}")
    
    print(f"\nClustering Results:")
    print(f"  Number of clusters: {n_clusters}")
    for i in range(n_clusters):
        cluster_size = (cluster_data['cluster'] == i).sum()
        print(f"  Cluster {i}: {cluster_size} observations ({cluster_size/len(cluster_data)*100:.1f}%)")
    
    print("\nRegime change analysis complete! Check the generated plots for visual insights.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Basic VVIX/VIX Ratio + Interest Rate Analysis
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
    print("VVIX/VIX Ratio + Interest Rate Analysis")
    print("=" * 60)
    
    # Data collection
    print("Collecting market and interest rate data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    # Download data
    market_data = yf.download(['^VIX', '^VVIX', 'SPY'], start=start_date, end=end_date, progress=False)
    rate_data = yf.download(['^TNX', '^FVX', '^IRX', '^TYX'], start=start_date, end=end_date, progress=False)
    
    print(f"Market data collected: {market_data.shape[0]} days")
    print(f"Interest rate data collected: {rate_data.shape[0]} days")
    
    # Extract data
    vix = market_data['Close']['^VIX'].dropna()
    vvix = market_data['Close']['^VVIX'].dropna()
    spy = market_data['Close']['SPY'].dropna()
    
    rates_10y = rate_data['Close']['^TNX'].dropna()
    rates_5y = rate_data['Close']['^FVX'].dropna()
    rates_3m = rate_data['Close']['^IRX'].dropna()
    rates_30y = rate_data['Close']['^TYX'].dropna()
    
    # Calculate ratio and returns
    ratio = (vvix / vix).dropna()
    spy_returns = spy.pct_change().dropna()
    
    # Calculate yield curve metrics
    yield_slope = (rates_10y - rates_3m).dropna()
    yield_steepness = (rates_30y - rates_10y).dropna()
    
    # Align all data to common dates
    all_data = pd.DataFrame({
        'ratio': ratio,
        'spy_return': spy_returns,
        'rate_10y': rates_10y,
        'rate_5y': rates_5y,
        'rate_3m': rates_3m,
        'rate_30y': rates_30y,
        'yield_slope': yield_slope,
        'yield_steepness': yield_steepness
    }).dropna()
    
    print(f"Analyzing {len(all_data)} data points with complete data...")
    
    # 1. Basic Statistics
    print("\n" + "="*60)
    print("1. BASIC STATISTICS")
    print("="*60)
    
    print("\nVVIX/VIX Ratio Statistics:")
    print(f"  Mean: {all_data['ratio'].mean():.4f}")
    print(f"  Std: {all_data['ratio'].std():.4f}")
    print(f"  Min: {all_data['ratio'].min():.4f}")
    print(f"  Max: {all_data['ratio'].max():.4f}")
    
    print("\nInterest Rate Statistics:")
    print(f"  10Y Rate - Mean: {all_data['rate_10y'].mean():.2f}%, Std: {all_data['rate_10y'].std():.2f}%")
    print(f"  5Y Rate - Mean: {all_data['rate_5y'].mean():.2f}%, Std: {all_data['rate_5y'].std():.2f}%")
    print(f"  3M Rate - Mean: {all_data['rate_3m'].mean():.2f}%, Std: {all_data['rate_3m'].std():.2f}%")
    print(f"  30Y Rate - Mean: {all_data['rate_30y'].mean():.2f}%, Std: {all_data['rate_30y'].std():.2f}%")
    
    print("\nYield Curve Statistics:")
    print(f"  Slope (10Y-3M) - Mean: {all_data['yield_slope'].mean():.2f}%, Std: {all_data['yield_slope'].std():.2f}%")
    print(f"  Steepness (30Y-10Y) - Mean: {all_data['yield_steepness'].mean():.2f}%, Std: {all_data['yield_steepness'].std():.2f}%")
    print(f"  Inversion periods: {(all_data['yield_slope'] < 0).sum()} days ({(all_data['yield_slope'] < 0).sum()/len(all_data)*100:.1f}%)")
    
    # 2. Correlation Analysis
    print("\n" + "="*60)
    print("2. CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlations
    corr_matrix = all_data.corr()
    
    print("\nVVIX/VIX Ratio Correlations:")
    ratio_correlations = corr_matrix['ratio'].drop('ratio')
    for idx, val in ratio_correlations.items():
        print(f"  {idx}: {val:.4f}")
    
    print("\nSPY Return Correlations:")
    spy_correlations = corr_matrix['spy_return'].drop('spy_return')
    for idx, val in spy_correlations.items():
        print(f"  {idx}: {val:.4f}")
    
    # 3. Interest Rate Regime Analysis
    print("\n" + "="*60)
    print("3. INTEREST RATE REGIME ANALYSIS")
    print("="*60)
    
    # Define rate regimes
    rate_low = all_data['rate_10y'].quantile(0.33)
    rate_high = all_data['rate_10y'].quantile(0.67)
    
    all_data['rate_regime'] = 'Normal'
    all_data.loc[all_data['rate_10y'] <= rate_low, 'rate_regime'] = 'Low Rates'
    all_data.loc[all_data['rate_10y'] >= rate_high, 'rate_regime'] = 'High Rates'
    
    print("\nInterest Rate Regimes:")
    regime_counts = all_data['rate_regime'].value_counts()
    for regime, count in regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(all_data)*100:.1f}%)")
    
    # Analyze by regime
    print("\nRatio behavior by interest rate regime:")
    for regime in all_data['rate_regime'].unique():
        regime_data = all_data[all_data['rate_regime'] == regime]
        
        print(f"\n{regime} Regime:")
        print(f"  Ratio mean: {regime_data['ratio'].mean():.4f}")
        print(f"  Ratio std: {regime_data['ratio'].std():.4f}")
        print(f"  Returns mean: {regime_data['spy_return'].mean():.4f} ({regime_data['spy_return'].mean()*100:.2f}%)")
        print(f"  Returns std: {regime_data['spy_return'].std():.4f} ({regime_data['spy_return'].std()*100:.2f}%)")
        print(f"  Rate mean: {regime_data['rate_10y'].mean():.2f}%")
        print(f"  Correlation (ratio vs returns): {regime_data['ratio'].corr(regime_data['spy_return']):.4f}")
        print(f"  Sample size: {len(regime_data)}")
    
    # 4. Yield Curve Regime Analysis
    print("\n" + "="*60)
    print("4. YIELD CURVE REGIME ANALYSIS")
    print("="*60)
    
    # Define yield curve regimes
    slope_low = all_data['yield_slope'].quantile(0.33)
    slope_high = all_data['yield_slope'].quantile(0.67)
    
    all_data['slope_regime'] = 'Normal'
    all_data.loc[all_data['yield_slope'] <= slope_low, 'slope_regime'] = 'Flat/Inverted'
    all_data.loc[all_data['yield_slope'] >= slope_high, 'slope_regime'] = 'Steep'
    
    print("\nYield Curve Slope Regimes:")
    slope_regime_counts = all_data['slope_regime'].value_counts()
    for regime, count in slope_regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(all_data)*100:.1f}%)")
    
    # Analyze by yield curve regime
    print("\nRatio behavior by yield curve regime:")
    for regime in all_data['slope_regime'].unique():
        regime_data = all_data[all_data['slope_regime'] == regime]
        
        print(f"\n{regime} Regime:")
        print(f"  Ratio mean: {regime_data['ratio'].mean():.4f}")
        print(f"  Returns mean: {regime_data['spy_return'].mean():.4f} ({regime_data['spy_return'].mean()*100:.2f}%)")
        print(f"  Slope mean: {regime_data['yield_slope'].mean():.2f}%")
        print(f"  Correlation (ratio vs returns): {regime_data['ratio'].corr(regime_data['spy_return']):.4f}")
        print(f"  Correlation (ratio vs slope): {regime_data['ratio'].corr(regime_data['yield_slope']):.4f}")
        print(f"  Sample size: {len(regime_data)}")
    
    # 5. Statistical Tests
    print("\n" + "="*60)
    print("5. STATISTICAL TESTS")
    print("="*60)
    
    # Test key correlations
    def test_correlation(x, y, name):
        corr, p_value = stats.pearsonr(x, y)
        print(f"\n{name}:")
        print(f"  Correlation: {corr:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        return corr, p_value
    
    test_correlation(all_data['ratio'], all_data['rate_10y'], "Ratio vs 10Y Rate")
    test_correlation(all_data['ratio'], all_data['yield_slope'], "Ratio vs Yield Curve Slope")
    test_correlation(all_data['spy_return'], all_data['rate_10y'], "SPY Returns vs 10Y Rate")
    test_correlation(all_data['spy_return'], all_data['yield_slope'], "SPY Returns vs Yield Curve Slope")
    
    # 6. Create Visualizations
    print("\n" + "="*60)
    print("6. GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VVIX/VIX Ratio + Interest Rate Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Ratio vs 10Y rates
    axes[0, 0].scatter(all_data['rate_10y'], all_data['ratio'], alpha=0.6, s=10)
    axes[0, 0].set_xlabel('10Y Treasury Rate (%)')
    axes[0, 0].set_ylabel('VVIX/VIX Ratio')
    axes[0, 0].set_title(f'Ratio vs 10Y Rates (r={all_data["ratio"].corr(all_data["rate_10y"]):.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Ratio vs yield curve slope
    axes[0, 1].scatter(all_data['yield_slope'], all_data['ratio'], alpha=0.6, s=10)
    axes[0, 1].set_xlabel('Yield Curve Slope (%)')
    axes[0, 1].set_ylabel('VVIX/VIX Ratio')
    axes[0, 1].set_title(f'Ratio vs Yield Curve Slope (r={all_data["ratio"].corr(all_data["yield_slope"]):.3f})')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Inversion Line')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Interest rates over time
    axes[0, 2].plot(all_data.index, all_data['rate_10y'], label='10Y', linewidth=2)
    axes[0, 2].plot(all_data.index, all_data['rate_5y'], label='5Y', linewidth=2)
    axes[0, 2].plot(all_data.index, all_data['rate_3m'], label='3M', linewidth=2)
    axes[0, 2].set_title('Interest Rates Over Time')
    axes[0, 2].set_ylabel('Rate (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Yield curve slope over time
    axes[1, 0].plot(all_data.index, all_data['yield_slope'], alpha=0.7, linewidth=1)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8, label='Inversion Line')
    axes[1, 0].set_title('Yield Curve Slope (10Y-3M)')
    axes[1, 0].set_ylabel('Slope (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Ratio by interest rate regime
    rate_colors = {'Low Rates': 'green', 'Normal': 'blue', 'High Rates': 'red'}
    for regime in all_data['rate_regime'].unique():
        regime_data = all_data[all_data['rate_regime'] == regime]
        axes[1, 1].scatter(regime_data.index, regime_data['ratio'], 
                          c=rate_colors[regime], alpha=0.6, s=10, label=regime)
    
    axes[1, 1].set_title('Ratio by Interest Rate Regime')
    axes[1, 1].set_ylabel('VVIX/VIX Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=axes[1, 2])
    axes[1, 2].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # 7. Summary
    print("\n" + "="*60)
    print("7. SUMMARY")
    print("="*60)
    
    print(f"\nKey Findings:")
    print(f"  Ratio vs 10Y Rate correlation: {all_data['ratio'].corr(all_data['rate_10y']):.4f}")
    print(f"  Ratio vs Yield Slope correlation: {all_data['ratio'].corr(all_data['yield_slope']):.4f}")
    print(f"  SPY Returns vs 10Y Rate correlation: {all_data['spy_return'].corr(all_data['rate_10y']):.4f}")
    print(f"  SPY Returns vs Yield Slope correlation: {all_data['spy_return'].corr(all_data['yield_slope']):.4f}")
    
    print(f"\nInterest Rate Environment:")
    print(f"  Average 10Y Rate: {all_data['rate_10y'].mean():.2f}%")
    print(f"  Average Yield Slope: {all_data['yield_slope'].mean():.2f}%")
    print(f"  Inversion periods: {(all_data['yield_slope'] < 0).sum()} days ({(all_data['yield_slope'] < 0).sum()/len(all_data)*100:.1f}%)")
    
    print("\nInterest rate analysis complete! Check the generated plots for visual insights.")

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Focused correlation analysis between VVIX/VIX ratio and market performance
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
    print("VVIX/VIX Ratio vs Market Correlation Analysis")
    print("=" * 60)
    
    # Data collection
    print("Collecting market data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    tickers = ['^VIX', '^VVIX', 'SPY', '^GSPC', 'QQQ', 'IWM']  # Multiple market indices
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    print(f"Data collected: {data.shape[0]} days")
    
    # Extract and align data
    vix_close = data['Close']['^VIX'].dropna()
    vvix_close = data['Close']['^VVIX'].dropna()
    spy_close = data['Close']['SPY'].dropna()
    spx_close = data['Close']['^GSPC'].dropna()
    qqq_close = data['Close']['QQQ'].dropna()
    iwm_close = data['Close']['IWM'].dropna()
    
    # Align all series
    common_dates = (vix_close.index.intersection(vvix_close.index)
                   .intersection(spy_close.index)
                   .intersection(spx_close.index)
                   .intersection(qqq_close.index)
                   .intersection(iwm_close.index))
    
    vix_aligned = vix_close.loc[common_dates]
    vvix_aligned = vvix_close.loc[common_dates]
    spy_aligned = spy_close.loc[common_dates]
    spx_aligned = spx_close.loc[common_dates]
    qqq_aligned = qqq_close.loc[common_dates]
    iwm_aligned = iwm_close.loc[common_dates]
    
    # Calculate ratio and returns
    ratio = vvix_aligned / vix_aligned
    
    # Calculate returns for different time horizons
    spy_returns = spy_aligned.pct_change().dropna()
    spx_returns = spx_aligned.pct_change().dropna()
    qqq_returns = qqq_aligned.pct_change().dropna()
    iwm_returns = iwm_aligned.pct_change().dropna()
    
    # Forward-looking returns
    spy_returns_1d = spy_aligned.pct_change(1).dropna()
    spy_returns_3d = spy_aligned.pct_change(3).dropna()
    spy_returns_5d = spy_aligned.pct_change(5).dropna()
    spy_returns_10d = spy_aligned.pct_change(10).dropna()
    spy_returns_20d = spy_aligned.pct_change(20).dropna()
    
    # Ratio changes
    ratio_change = ratio.pct_change().dropna()
    ratio_change_3d = ratio.pct_change(3).dropna()
    ratio_change_5d = ratio.pct_change(5).dropna()
    ratio_change_10d = ratio.pct_change(10).dropna()
    
    print(f"\nAnalyzing {len(ratio)} data points...")
    
    # 1. Basic Correlation Analysis
    print("\n" + "="*60)
    print("1. BASIC CORRELATION ANALYSIS")
    print("="*60)
    
    # Create correlation matrix
    correlation_data = pd.DataFrame({
        'VVIX/VIX_Ratio': ratio,
        'Ratio_Change': ratio_change,
        'SPY_Return': spy_returns,
        'SPX_Return': spx_returns,
        'QQQ_Return': qqq_returns,
        'IWM_Return': iwm_returns,
        'SPY_1d_Forward': spy_returns_1d,
        'SPY_3d_Forward': spy_returns_3d,
        'SPY_5d_Forward': spy_returns_5d,
        'SPY_10d_Forward': spy_returns_10d,
        'SPY_20d_Forward': spy_returns_20d
    }).dropna()
    
    # Calculate correlations
    corr_matrix = correlation_data.corr()
    
    print("\nCorrelation Matrix (VVIX/VIX Ratio vs Market Returns):")
    print("-" * 50)
    
    # Focus on ratio correlations
    ratio_correlations = corr_matrix['VVIX/VIX_Ratio'].drop('VVIX/VIX_Ratio')
    ratio_change_correlations = corr_matrix['Ratio_Change'].drop('Ratio_Change')
    
    print("\nVVIX/VIX Ratio Correlations:")
    for idx, val in ratio_correlations.items():
        print(f"  {idx}: {val:.4f}")
    
    print("\nRatio Change Correlations:")
    for idx, val in ratio_change_correlations.items():
        print(f"  {idx}: {val:.4f}")
    
    # 2. Rolling Correlation Analysis
    print("\n" + "="*60)
    print("2. ROLLING CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate rolling correlations
    rolling_windows = [30, 60, 90, 252]  # 1 month, 2 months, 3 months, 1 year
    
    for window in rolling_windows:
        rolling_corr = ratio.rolling(window).corr(spy_returns.rolling(window).mean())
        mean_corr = rolling_corr.mean()
        std_corr = rolling_corr.std()
        
        print(f"\n{window}-day rolling correlation with SPY:")
        print(f"  Mean: {mean_corr:.4f}")
        print(f"  Std:  {std_corr:.4f}")
        print(f"  Min:  {rolling_corr.min():.4f}")
        print(f"  Max:  {rolling_corr.max():.4f}")
    
    # 3. Market Regime Analysis
    print("\n" + "="*60)
    print("3. MARKET REGIME ANALYSIS")
    print("="*60)
    
    # Define market regimes based on volatility
    spy_volatility = spy_returns.rolling(30).std()
    high_vol_periods = spy_volatility > spy_volatility.quantile(0.8)
    low_vol_periods = spy_volatility < spy_volatility.quantile(0.2)
    
    print("\nHigh Volatility Periods (top 20%):")
    high_vol_corr = ratio[high_vol_periods].corr(spy_returns[high_vol_periods])
    print(f"  Correlation: {high_vol_corr:.4f}")
    print(f"  Sample size: {high_vol_periods.sum()}")
    
    print("\nLow Volatility Periods (bottom 20%):")
    low_vol_corr = ratio[low_vol_periods].corr(spy_returns[low_vol_periods])
    print(f"  Correlation: {low_vol_corr:.4f}")
    print(f"  Sample size: {low_vol_periods.sum()}")
    
    # 4. Statistical Significance Testing
    print("\n" + "="*60)
    print("4. STATISTICAL SIGNIFICANCE TESTING")
    print("="*60)
    
    # Test correlation significance
    def test_correlation_significance(x, y, name):
        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]
        
        if len(x_clean) < 3:
            return None
            
        # Pearson correlation
        corr_coef, p_value = stats.pearsonr(x_clean, y_clean)
        
        # Spearman correlation (non-parametric)
        spearman_corr, spearman_p = stats.spearmanr(x_clean, y_clean)
        
        print(f"\n{name}:")
        print(f"  Pearson correlation: {corr_coef:.4f} (p-value: {p_value:.4f})")
        print(f"  Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4f})")
        print(f"  Significant (p<0.05): {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Sample size: {len(x_clean)}")
        
        return {
            'pearson_corr': corr_coef,
            'pearson_p': p_value,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'significant': p_value < 0.05
        }
    
    # Test various correlations
    test_correlation_significance(ratio, spy_returns, "Ratio vs SPY Returns")
    test_correlation_significance(ratio_change, spy_returns, "Ratio Change vs SPY Returns")
    test_correlation_significance(ratio, spy_returns_5d, "Ratio vs SPY 5-day Forward Returns")
    test_correlation_significance(ratio, spy_returns_10d, "Ratio vs SPY 10-day Forward Returns")
    test_correlation_significance(ratio, spy_returns_20d, "Ratio vs SPY 20-day Forward Returns")
    
    # 5. Create Visualizations
    print("\n" + "="*60)
    print("5. GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create comprehensive correlation plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VVIX/VIX Ratio vs Market Correlation Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=axes[0, 0])
    axes[0, 0].set_title('Correlation Matrix')
    
    # Plot 2: Ratio vs SPY returns scatter
    axes[0, 1].scatter(ratio, spy_returns, alpha=0.5, s=1)
    axes[0, 1].set_xlabel('VVIX/VIX Ratio')
    axes[0, 1].set_ylabel('SPY Daily Returns')
    axes[0, 1].set_title(f'Ratio vs SPY Returns (r={ratio.corr(spy_returns):.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Ratio vs 5-day forward returns
    axes[0, 2].scatter(ratio, spy_returns_5d, alpha=0.5, s=1)
    axes[0, 2].set_xlabel('VVIX/VIX Ratio')
    axes[0, 2].set_ylabel('SPY 5-day Forward Returns')
    axes[0, 2].set_title(f'Ratio vs SPY 5d Forward (r={ratio.corr(spy_returns_5d):.3f})')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Rolling correlation over time
    rolling_corr_30 = ratio.rolling(30).corr(spy_returns.rolling(30).mean())
    rolling_corr_90 = ratio.rolling(90).corr(spy_returns.rolling(90).mean())
    
    axes[1, 0].plot(rolling_corr_30.index, rolling_corr_30.values, label='30-day', alpha=0.8)
    axes[1, 0].plot(rolling_corr_90.index, rolling_corr_90.values, label='90-day', alpha=0.8)
    axes[1, 0].set_title('Rolling Correlation Over Time')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Ratio distribution by market performance
    # Split data into quintiles based on SPY returns
    spy_quintiles = pd.qcut(spy_returns, 5, labels=['Q1 (Worst)', 'Q2', 'Q3', 'Q4', 'Q5 (Best)'])
    
    ratio_by_quintile = []
    labels = []
    for quintile in spy_quintiles.cat.categories:
        mask = spy_quintiles == quintile
        if mask.sum() > 0:
            ratio_by_quintile.append(ratio[mask].dropna())
            labels.append(quintile)
    
    axes[1, 1].boxplot(ratio_by_quintile, labels=labels)
    axes[1, 1].set_title('Ratio Distribution by Market Performance Quintiles')
    axes[1, 1].set_ylabel('VVIX/VIX Ratio')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Ratio vs different market indices
    market_returns = [spy_returns, spx_returns, qqq_returns, iwm_returns]
    market_names = ['SPY', 'SPX', 'QQQ', 'IWM']
    correlations = [ratio.corr(ret) for ret in market_returns]
    
    bars = axes[1, 2].bar(market_names, correlations, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    axes[1, 2].set_title('Ratio Correlation with Different Market Indices')
    axes[1, 2].set_ylabel('Correlation Coefficient')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{corr:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Summary Statistics
    print("\n" + "="*60)
    print("6. SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nVVIX/VIX Ratio Statistics:")
    print(f"  Mean: {ratio.mean():.4f}")
    print(f"  Median: {ratio.median():.4f}")
    print(f"  Std: {ratio.std():.4f}")
    print(f"  Min: {ratio.min():.4f}")
    print(f"  Max: {ratio.max():.4f}")
    
    print(f"\nMarket Returns Statistics:")
    print(f"  SPY Mean Daily Return: {spy_returns.mean():.4f} ({spy_returns.mean()*100:.2f}%)")
    print(f"  SPY Volatility: {spy_returns.std():.4f} ({spy_returns.std()*100:.2f}%)")
    print(f"  SPY Sharpe Ratio: {spy_returns.mean()/spy_returns.std()*np.sqrt(252):.4f}")
    
    print(f"\nKey Correlations:")
    print(f"  Ratio vs SPY (1d): {ratio.corr(spy_returns):.4f}")
    print(f"  Ratio vs SPY (5d): {ratio.corr(spy_returns_5d):.4f}")
    print(f"  Ratio vs SPY (10d): {ratio.corr(spy_returns_10d):.4f}")
    print(f"  Ratio vs SPY (20d): {ratio.corr(spy_returns_20d):.4f}")
    
    print(f"\nRatio Change Correlations:")
    print(f"  Ratio Change vs SPY (1d): {ratio_change.corr(spy_returns):.4f}")
    print(f"  Ratio Change vs SPY (5d): {ratio_change.corr(spy_returns_5d):.4f}")
    
    print("\nAnalysis complete! Check the generated plots for visual insights.")

if __name__ == "__main__":
    main()


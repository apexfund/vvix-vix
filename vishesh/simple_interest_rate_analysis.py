#!/usr/bin/env python3
"""
Simplified VVIX/VIX Ratio + Interest Rate Analysis
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
    
    # Market data
    market_tickers = ['^VIX', '^VVIX', 'SPY']
    
    # Interest rate data (using available tickers)
    rate_tickers = ['^TNX', '^FVX', '^IRX', '^TYX']  # 10Y, 5Y, 3M, 30Y
    
    # Download data
    market_data = yf.download(market_tickers, start=start_date, end=end_date, progress=False)
    rate_data = yf.download(rate_tickers, start=start_date, end=end_date, progress=False)
    
    print(f"Market data collected: {market_data.shape[0]} days")
    print(f"Interest rate data collected: {rate_data.shape[0]} days")
    
    # Extract and align data
    vix_close = market_data['Close']['^VIX'].dropna()
    vvix_close = market_data['Close']['^VVIX'].dropna()
    spy_close = market_data['Close']['SPY'].dropna()
    
    rates_10y = rate_data['Close']['^TNX'].dropna()
    rates_5y = rate_data['Close']['^FVX'].dropna()
    rates_3m = rate_data['Close']['^IRX'].dropna()
    rates_30y = rate_data['Close']['^TYX'].dropna()
    
    # Align all series to common dates
    common_dates = (vix_close.index.intersection(vvix_close.index)
                   .intersection(spy_close.index)
                   .intersection(rates_10y.index)
                   .intersection(rates_5y.index)
                   .intersection(rates_3m.index)
                   .intersection(rates_30y.index))
    
    # Align all data
    vix_aligned = vix_close.loc[common_dates]
    vvix_aligned = vvix_close.loc[common_dates]
    spy_aligned = spy_close.loc[common_dates]
    rates_10y_aligned = rates_10y.loc[common_dates]
    rates_5y_aligned = rates_5y.loc[common_dates]
    rates_3m_aligned = rates_3m.loc[common_dates]
    rates_30y_aligned = rates_30y.loc[common_dates]
    
    # Calculate ratio and returns
    ratio = vvix_aligned / vix_aligned
    spy_returns = spy_aligned.pct_change().dropna()
    
    # Calculate yield curve metrics
    yield_curve_slope = rates_10y_aligned - rates_3m_aligned  # 10Y-3M spread
    yield_curve_steepness = rates_30y_aligned - rates_10y_aligned  # 30Y-10Y spread
    
    print(f"Analyzing {len(ratio)} data points with interest rate data...")
    
    # 1. Interest Rate Regime Analysis
    print("\n" + "="*60)
    print("1. INTEREST RATE REGIME ANALYSIS")
    print("="*60)
    
    # Define interest rate regimes based on 10Y yield
    rate_low_threshold = rates_10y_aligned.quantile(0.33)
    rate_high_threshold = rates_10y_aligned.quantile(0.67)
    
    rate_regimes = pd.Series('Normal', index=rates_10y_aligned.index)
    rate_regimes[rates_10y_aligned <= rate_low_threshold] = 'Low Rates'
    rate_regimes[rates_10y_aligned >= rate_high_threshold] = 'High Rates'
    
    print("\nInterest Rate Regimes (10Y Treasury):")
    rate_regime_counts = rate_regimes.value_counts()
    for regime, count in rate_regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(rate_regimes)*100:.1f}%)")
    
    # Analyze ratio behavior by interest rate regime
    print("\nRatio behavior by interest rate regime:")
    for regime in rate_regimes.unique():
        regime_mask = rate_regimes == regime
        # Align indices properly
        common_idx = ratio.index.intersection(rate_regimes.index)
        regime_mask_aligned = regime_mask.loc[common_idx]
        regime_ratio = ratio.loc[common_idx][regime_mask_aligned]
        regime_returns = spy_returns.loc[common_idx][regime_mask_aligned]
        regime_rates = rates_10y_aligned.loc[common_idx][regime_mask_aligned]
        
        print(f"\n{regime} Regime:")
        print(f"  Ratio mean: {regime_ratio.mean():.4f}")
        print(f"  Ratio std: {regime_ratio.std():.4f}")
        print(f"  Returns mean: {regime_returns.mean():.4f} ({regime_returns.mean()*100:.2f}%)")
        print(f"  Returns std: {regime_returns.std():.4f} ({regime_returns.std()*100:.2f}%)")
        print(f"  Rate mean: {regime_rates.mean():.4f}%")
        print(f"  Correlation (ratio vs returns): {regime_ratio.corr(regime_returns):.4f}")
        print(f"  Correlation (ratio vs rates): {regime_ratio.corr(regime_rates):.4f}")
        print(f"  Sample size: {len(regime_ratio)}")
    
    # 2. Yield Curve Analysis
    print("\n" + "="*60)
    print("2. YIELD CURVE ANALYSIS")
    print("="*60)
    
    # Analyze yield curve slope regimes
    slope_low_threshold = yield_curve_slope.quantile(0.33)
    slope_high_threshold = yield_curve_slope.quantile(0.67)
    
    slope_regimes = pd.Series('Normal', index=yield_curve_slope.index)
    slope_regimes[yield_curve_slope <= slope_low_threshold] = 'Flat/Inverted'
    slope_regimes[yield_curve_slope >= slope_high_threshold] = 'Steep'
    
    print("\nYield Curve Slope Regimes:")
    slope_regime_counts = slope_regimes.value_counts()
    for regime, count in slope_regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(slope_regimes)*100:.1f}%)")
    
    # Analyze ratio behavior by yield curve regime
    print("\nRatio behavior by yield curve regime:")
    for regime in slope_regimes.unique():
        regime_mask = slope_regimes == regime
        # Align indices properly
        common_idx = ratio.index.intersection(slope_regimes.index)
        regime_mask_aligned = regime_mask.loc[common_idx]
        regime_ratio = ratio.loc[common_idx][regime_mask_aligned]
        regime_returns = spy_returns.loc[common_idx][regime_mask_aligned]
        regime_slope = yield_curve_slope.loc[common_idx][regime_mask_aligned]
        
        print(f"\n{regime} Regime:")
        print(f"  Ratio mean: {regime_ratio.mean():.4f}")
        print(f"  Returns mean: {regime_returns.mean():.4f} ({regime_returns.mean()*100:.2f}%)")
        print(f"  Slope mean: {regime_slope.mean():.4f}%")
        print(f"  Correlation (ratio vs returns): {regime_ratio.corr(regime_returns):.4f}")
        print(f"  Correlation (ratio vs slope): {regime_ratio.corr(regime_slope):.4f}")
        print(f"  Sample size: {len(regime_ratio)}")
    
    # 3. Correlation Analysis
    print("\n" + "="*60)
    print("3. CORRELATION ANALYSIS")
    print("="*60)
    
    # Create correlation matrix
    correlation_data = pd.DataFrame({
        'VVIX/VIX_Ratio': ratio,
        'SPY_Return': spy_returns,
        '10Y_Rate': rates_10y_aligned,
        '5Y_Rate': rates_5y_aligned,
        '3M_Rate': rates_3m_aligned,
        '30Y_Rate': rates_30y_aligned,
        'Yield_Slope': yield_curve_slope,
        'Yield_Steepness': yield_curve_steepness
    }).dropna()
    
    corr_matrix = correlation_data.corr()
    
    print("\nCorrelation Matrix:")
    print("Ratio correlations with interest rates:")
    ratio_correlations = corr_matrix['VVIX/VIX_Ratio'].drop('VVIX/VIX_Ratio')
    for idx, val in ratio_correlations.items():
        print(f"  {idx}: {val:.4f}")
    
    print("\nSPY Return correlations with interest rates:")
    spy_correlations = corr_matrix['SPY_Return'].drop('SPY_Return')
    for idx, val in spy_correlations.items():
        print(f"  {idx}: {val:.4f}")
    
    # 4. Statistical Tests
    print("\n" + "="*60)
    print("4. STATISTICAL TESTS")
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
        
        print(f"\n{name}:")
        print(f"  Correlation: {corr_coef:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Sample size: {len(x_clean)}")
        
        return corr_coef, p_value
    
    # Test key correlations
    test_correlation_significance(ratio, rates_10y_aligned, "Ratio vs 10Y Rate")
    test_correlation_significance(ratio, yield_curve_slope, "Ratio vs Yield Curve Slope")
    test_correlation_significance(spy_returns, rates_10y_aligned, "SPY Returns vs 10Y Rate")
    test_correlation_significance(spy_returns, yield_curve_slope, "SPY Returns vs Yield Curve Slope")
    
    # 5. Create Visualizations
    print("\n" + "="*60)
    print("5. GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Create comprehensive interest rate analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('VVIX/VIX Ratio + Interest Rate Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Ratio vs 10Y rates scatter
    axes[0, 0].scatter(rates_10y_aligned, ratio, alpha=0.6, s=10)
    axes[0, 0].set_xlabel('10Y Treasury Rate (%)')
    axes[0, 0].set_ylabel('VVIX/VIX Ratio')
    axes[0, 0].set_title(f'Ratio vs 10Y Rates (r={ratio.corr(rates_10y_aligned):.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Ratio vs yield curve slope
    axes[0, 1].scatter(yield_curve_slope, ratio, alpha=0.6, s=10)
    axes[0, 1].set_xlabel('Yield Curve Slope (%)')
    axes[0, 1].set_ylabel('VVIX/VIX Ratio')
    axes[0, 1].set_title(f'Ratio vs Yield Curve Slope (r={ratio.corr(yield_curve_slope):.3f})')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Inversion Line')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Interest rates over time
    axes[0, 2].plot(rates_10y_aligned.index, rates_10y_aligned.values, label='10Y', linewidth=2)
    axes[0, 2].plot(rates_5y_aligned.index, rates_5y_aligned.values, label='5Y', linewidth=2)
    axes[0, 2].plot(rates_3m_aligned.index, rates_3m_aligned.values, label='3M', linewidth=2)
    axes[0, 2].set_title('Interest Rates Over Time')
    axes[0, 2].set_ylabel('Rate (%)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Yield curve slope over time
    axes[1, 0].plot(yield_curve_slope.index, yield_curve_slope.values, alpha=0.7, linewidth=1)
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8, label='Inversion Line')
    axes[1, 0].set_title('Yield Curve Slope (10Y-3M)')
    axes[1, 0].set_ylabel('Slope (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Ratio by interest rate regime
    rate_regime_colors = {'Low Rates': 'green', 'Normal': 'blue', 'High Rates': 'red'}
    for regime in rate_regimes.unique():
        regime_mask = rate_regimes == regime
        common_idx = ratio.index.intersection(rate_regimes.index)
        regime_mask_aligned = regime_mask.loc[common_idx]
        axes[1, 1].scatter(ratio.loc[common_idx][regime_mask_aligned].index, 
                          ratio.loc[common_idx][regime_mask_aligned].values, 
                          c=rate_regime_colors[regime], alpha=0.6, s=10, label=regime)
    
    axes[1, 1].set_title('Ratio by Interest Rate Regime')
    axes[1, 1].set_ylabel('VVIX/VIX Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, ax=axes[1, 2])
    axes[1, 2].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    # 6. Summary Statistics
    print("\n" + "="*60)
    print("6. SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nInterest Rate Regimes:")
    for regime, count in rate_regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(rate_regimes)*100:.1f}%)")
    
    print(f"\nYield Curve Regimes:")
    for regime, count in slope_regime_counts.items():
        print(f"  {regime}: {count} days ({count/len(slope_regimes)*100:.1f}%)")
    
    print(f"\nKey Correlations:")
    print(f"  Ratio vs 10Y Rate: {ratio.corr(rates_10y_aligned):.4f}")
    print(f"  Ratio vs Yield Slope: {ratio.corr(yield_curve_slope):.4f}")
    print(f"  SPY Returns vs 10Y Rate: {spy_returns.corr(rates_10y_aligned):.4f}")
    print(f"  SPY Returns vs Yield Slope: {spy_returns.corr(yield_curve_slope):.4f}")
    
    print(f"\nInterest Rate Statistics:")
    print(f"  10Y Rate - Mean: {rates_10y_aligned.mean():.2f}%, Std: {rates_10y_aligned.std():.2f}%")
    print(f"  Yield Slope - Mean: {yield_curve_slope.mean():.2f}%, Std: {yield_curve_slope.std():.2f}%")
    print(f"  Inversion periods: {(yield_curve_slope < 0).sum()} days ({(yield_curve_slope < 0).sum()/len(yield_curve_slope)*100:.1f}%)")
    
    print("\nInterest rate analysis complete! Check the generated plots for visual insights.")

if __name__ == "__main__":
    main()


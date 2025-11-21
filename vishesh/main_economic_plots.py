#!/usr/bin/env python3
"""
Main Economic Visualization: VVIX/VIX Ratio and Returns in Respect to Economy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Creating Main Economic Visualization Plots...")
    print("=" * 60)
    
    # Data collection
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    print("Collecting market and economic data...")
    market_data = yf.download(['^VIX', '^VVIX', 'SPY'], start=start_date, end=end_date, progress=False)
    rate_data = yf.download(['^TNX', '^FVX', '^IRX', '^TYX'], start=start_date, end=end_date, progress=False)
    
    # Extract and align data
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
    
    # Align all data
    all_data = pd.DataFrame({
        'ratio': ratio,
        'spy_return': spy_returns,
        'rate_10y': rates_10y,
        'rate_3m': rates_3m,
        'rate_5y': rates_5y,
        'rate_30y': rates_30y,
        'yield_slope': yield_slope
    }).dropna()
    
    print(f"Data collected: {len(all_data)} data points")
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle('VVIX/VIX Ratio and Returns in Respect to Economy', 
                 fontsize=18, fontweight='bold', y=1.02)
    
    # Plot 1: VVIX/VIX Ratio Over Time with Interest Rates
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    
    ax1.plot(all_data.index, all_data['ratio'], color='blue', linewidth=2, alpha=0.8, label='VVIX/VIX Ratio')
    ax1.set_ylabel('VVIX/VIX Ratio', color='blue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    ax1_twin.plot(all_data.index, all_data['rate_10y'], color='red', linewidth=2, alpha=0.7, label='10Y Rate')
    ax1_twin.set_ylabel('10Y Treasury Rate (%)', color='red', fontsize=12, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title('VVIX/VIX Ratio vs 10Y Interest Rates Over Time', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    
    # Plot 2: Ratio vs Yield Curve Slope (Scatter)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(all_data['yield_slope'], all_data['ratio'], 
                         c=all_data['spy_return'], cmap='RdYlGn', 
                         s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Inversion Line')
    ax2.set_xlabel('Yield Curve Slope (10Y - 3M) (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('VVIX/VIX Ratio', fontsize=12, fontweight='bold')
    ax2.set_title(f'Ratio vs Yield Curve Slope\n(r={all_data["ratio"].corr(all_data["yield_slope"]):.3f})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label='SPY Returns')
    
    # Plot 3: Returns vs Yield Curve Slope
    ax3 = axes[0, 2]
    scatter3 = ax3.scatter(all_data['yield_slope'], all_data['spy_return']*100, 
                          c=all_data['ratio'], cmap='viridis',
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Inversion Line')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Yield Curve Slope (10Y - 3M) (%)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('SPY Returns (%)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Market Returns vs Yield Curve Slope\n(r={all_data["spy_return"].corr(all_data["yield_slope"]):.3f})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    plt.colorbar(scatter3, ax=ax3, label='VVIX/VIX Ratio')
    
    # Plot 4: Ratio vs 10Y Rate (Scatter)
    ax4 = axes[1, 0]
    scatter4 = ax4.scatter(all_data['rate_10y'], all_data['ratio'], 
                          c=all_data['spy_return'], cmap='RdYlGn',
                          s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('10Y Treasury Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('VVIX/VIX Ratio', fontsize=12, fontweight='bold')
    ax4.set_title(f'Ratio vs 10Y Interest Rate\n(r={all_data["ratio"].corr(all_data["rate_10y"]):.3f})', 
                  fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='SPY Returns')
    
    # Plot 5: Yield Curve Over Time
    ax5 = axes[1, 1]
    ax5.plot(all_data.index, all_data['rate_30y'], label='30Y', linewidth=2.5, alpha=0.9)
    ax5.plot(all_data.index, all_data['rate_10y'], label='10Y', linewidth=2.5, alpha=0.9)
    ax5.plot(all_data.index, all_data['rate_5y'], label='5Y', linewidth=2.5, alpha=0.9)
    ax5.plot(all_data.index, all_data['rate_3m'], label='3M', linewidth=2.5, alpha=0.9)
    
    # Highlight inversion periods
    inversion_mask = all_data['yield_slope'] < 0
    ax5.fill_between(all_data.index, ax5.get_ylim()[0], ax5.get_ylim()[1], 
                     where=inversion_mask, alpha=0.2, color='red', label='Inverted')
    
    ax5.set_ylabel('Interest Rate (%)', fontsize=12, fontweight='bold')
    ax5.set_title('Treasury Yield Curve Over Time', fontsize=14, fontweight='bold', pad=15)
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Three-Panel Economic Overview
    ax6 = axes[1, 2]
    
    # Create 3 subplots in the last panel
    gs = ax6.get_gridspec()
    ax6.remove()
    ax6_1 = fig.add_subplot(gs[1, 2])
    
    # Ratio over time
    ax6_1.plot(all_data.index, all_data['ratio'], color='blue', linewidth=2.5, alpha=0.9)
    ax6_1.fill_between(all_data.index, all_data['ratio'], alpha=0.3, color='blue')
    ax6_1.set_ylabel('VVIX/VIX Ratio', fontsize=11, fontweight='bold', color='blue')
    ax6_1.tick_params(axis='y', labelcolor='blue')
    ax6_1.set_title('Economic Summary: Ratio, Yield Slope & Returns', fontsize=12, fontweight='bold', pad=15)
    ax6_1.grid(True, alpha=0.3)
    
    ax6_2 = ax6_1.twinx()
    ax6_2.plot(all_data.index, all_data['yield_slope'], color='red', linewidth=2.5, alpha=0.9)
    ax6_2.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax6_2.set_ylabel('Yield Slope (%)', fontsize=11, fontweight='bold', color='red')
    ax6_2.tick_params(axis='y', labelcolor='red')
    
    ax6_3 = ax6_1.twinx()
    ax6_3.spines['right'].set_position(('outward', 60))
    ax6_3.plot(all_data.index, all_data['spy_return']*100, color='green', linewidth=1.5, alpha=0.7)
    ax6_3.set_ylabel('SPY Returns (%)', fontsize=11, fontweight='bold', color='green')
    ax6_3.tick_params(axis='y', labelcolor='green')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('vvix_vix_main_economic_plots.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'vvix_vix_main_economic_plots.png'")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"\nVVIX/VIX Ratio:")
    print(f"  Mean: {all_data['ratio'].mean():.4f}")
    print(f"  Std: {all_data['ratio'].std():.4f}")
    
    print(f"\nInterest Rates:")
    print(f"  10Y Rate - Mean: {all_data['rate_10y'].mean():.2f}%")
    print(f"  Yield Slope - Mean: {all_data['yield_slope'].mean():.2f}%")
    print(f"  Inversion periods: {(all_data['yield_slope'] < 0).sum()} days ({(all_data['yield_slope'] < 0).sum()/len(all_data)*100:.1f}%)")
    
    print(f"\nMarket Returns:")
    print(f"  Mean: {all_data['spy_return'].mean():.4f} ({all_data['spy_return'].mean()*100:.2f}%)")
    print(f"  Std: {all_data['spy_return'].std():.4f} ({all_data['spy_return'].std()*100:.2f}%)")
    
    print(f"\nKey Correlations:")
    print(f"  Ratio vs Yield Slope: {all_data['ratio'].corr(all_data['yield_slope']):.4f}")
    print(f"  Ratio vs 10Y Rate: {all_data['ratio'].corr(all_data['rate_10y']):.4f}")
    print(f"  Returns vs Yield Slope: {all_data['spy_return'].corr(all_data['yield_slope']):.4f}")
    print(f"  Returns vs 10Y Rate: {all_data['spy_return'].corr(all_data['rate_10y']):.4f}")

if __name__ == "__main__":
    main()


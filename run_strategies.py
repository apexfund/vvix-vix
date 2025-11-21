"""
Run and compare Strategy 1 (Defensive Equity) and Strategy 3 (Adaptive Allocation)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy import VVIXVIXStrategy
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Create results directory
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Timestamp for this run
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

def run_and_compare_strategies():
    """Execute both strategies and provide comprehensive comparison"""
    
    print("="*80)
    print("VVIX/VIX TRADING STRATEGIES: COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Initialize strategy class
    strategy = VVIXVIXStrategy(start_date='2010-01-01')
    
    # Fetch market data
    strategy.fetch_data()
    
    print(f"\n📊 Data Overview:")
    print(f"  Period: {strategy.data.index[0].date()} to {strategy.data.index[-1].date()}")
    print(f"  Total days: {len(strategy.data)}")
    print(f"  VVIX/VIX Ratio - Mean: {strategy.data['VVIX_VIX_Ratio'].mean():.2f}, Std: {strategy.data['VVIX_VIX_Ratio'].std():.2f}")
    
    # ==================== STRATEGY 1: DEFENSIVE EQUITY ====================
    print("\n" + "="*80)
    print("🛡️  STRATEGY 1: DEFENSIVE EQUITY OVERLAY")
    print("="*80)
    print("\n📋 Logic:")
    print("  • Binary on/off signal based on thresholds")
    print("  • IF VVIX/VIX ratio > 90th percentile OR (VIX >= 15 AND VVIX >= 85)")
    print("  • THEN reduce equity exposure to 50%")
    print("  • ELSE maintain 100% equity exposure")
    
    signals_1 = strategy.strategy_1_defensive_overlay(
        ratio_threshold_percentile=90,
        vix_threshold=15,
        vvix_threshold=85,
        equity_reduction=0.5
    )
    
    # ==================== STRATEGY 3: ADAPTIVE ALLOCATION ====================
    print("\n" + "="*80)
    print("📈 STRATEGY 3: ADAPTIVE ASSET ALLOCATION")
    print("="*80)
    print("\n📋 Logic:")
    print("  • Continuous adjustment based on normalized VVIX/VIX ratio")
    print("  • Normalize ratio between 25th and 75th percentile (0 to 1)")
    print("  • Equity Exposure = 1.0 - (normalized_ratio × 0.7)")
    print("  • Smoothly scales from 100% (calm) to 30% (stressed)")
    
    signals_3 = strategy.strategy_3_adaptive_allocation(
        ratio_threshold_percentile=75,
        volatility_lookback=20
    )
    
    # ==================== COMPARISON TABLE ====================
    comparison = strategy.compare_strategies()
    
    print("\n" + "="*80)
    print("📊 PERFORMANCE COMPARISON TABLE")
    print("="*80)
    print(comparison.to_string(index=False))
    
    # ==================== DETAILED INSIGHTS ====================
    s1_metrics = strategy.strategy_results['Defensive_Overlay']['metrics']
    s3_metrics = strategy.strategy_results['Adaptive_Allocation']['metrics']
    
    print("\n" + "="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    
    print("\n🛡️  STRATEGY 1 (Defensive Equity):")
    print(f"  • Type: Binary (on/off)")
    print(f"  • Signal active: {s1_metrics['signal_days']} days ({s1_metrics['signal_days']/len(signals_1)*100:.1f}% of time)")
    print(f"  • When active: Reduces to 50% equity exposure")
    print(f"  • Annualized Return: {s1_metrics['annualized_return']*100:.2f}%")
    print(f"  • Sharpe Ratio: {s1_metrics['sharpe_ratio']:.2f}")
    print(f"  • Max Drawdown: {s1_metrics['max_drawdown']*100:.2f}%")
    
    print("\n📈 STRATEGY 3 (Adaptive Allocation):")
    print(f"  • Type: Continuous (smooth scaling)")
    print(f"  • Average equity exposure: {s3_metrics['avg_exposure']:.1%}")
    print(f"  • Exposure range: {signals_3['Equity_Exposure'].min():.1%} to {signals_3['Equity_Exposure'].max():.1%}")
    print(f"  • Annualized Return: {s3_metrics['annualized_return']*100:.2f}%")
    print(f"  • Sharpe Ratio: {s3_metrics['sharpe_ratio']:.2f}")
    print(f"  • Max Drawdown: {s3_metrics['max_drawdown']*100:.2f}%")
    
    print("\n🎯 HEAD-TO-HEAD:")
    return_diff = (s3_metrics['annualized_return'] - s1_metrics['annualized_return']) * 100
    sharpe_diff = s3_metrics['sharpe_ratio'] - s1_metrics['sharpe_ratio']
    dd_diff = (s3_metrics['max_drawdown'] - s1_metrics['max_drawdown']) * 100
    
    print(f"  • Return difference: {return_diff:+.2f}% (Strategy 3 vs Strategy 1)")
    print(f"  • Sharpe difference: {sharpe_diff:+.2f} (Strategy 3 vs Strategy 1)")
    print(f"  • Drawdown difference: {dd_diff:+.2f}% (Strategy 3 vs Strategy 1)")
    
    if s3_metrics['sharpe_ratio'] > s1_metrics['sharpe_ratio']:
        print("\n✅ Winner: Strategy 3 (Adaptive) - Better risk-adjusted returns")
    else:
        print("\n✅ Winner: Strategy 1 (Defensive) - Better risk-adjusted returns")
    
    # ==================== VISUALIZATIONS ====================
    print("\n" + "="*80)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Plot 1: Cumulative Returns Comparison
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(signals_1.index, signals_1['SPY_Cumulative'], 
            label='SPY (Buy & Hold)', linewidth=2.5, alpha=0.7, color='gray')
    ax.plot(signals_1.index, signals_1['Strategy_Cumulative'], 
            label='Strategy 1: Defensive Equity', linewidth=2, alpha=0.9, color='blue')
    ax.plot(signals_3.index, signals_3['Strategy_Cumulative'], 
            label='Strategy 3: Adaptive Allocation', linewidth=2, alpha=0.9, color='green')
    ax.set_title('Cumulative Returns: SPY vs Strategy 1 vs Strategy 3', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Growth of $1', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'1_cumulative_returns_{TIMESTAMP}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()
    
    # Plot 2: Equity Exposure Comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Strategy 1 exposure
    ax1.fill_between(signals_1.index, 0, signals_1['Equity_Exposure'], 
                      alpha=0.5, color='blue', label='Equity Exposure')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='50% Threshold')
    ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='100% (Full)')
    ax1.set_ylabel('Equity Exposure', fontsize=11)
    ax1.set_title('Strategy 1: Binary Exposure (50% or 100%)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Strategy 3 exposure
    ax2.fill_between(signals_3.index, 0, signals_3['Equity_Exposure'], 
                      alpha=0.5, color='green', label='Equity Exposure')
    ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='30% Min')
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='100% Max')
    ax2.set_ylabel('Equity Exposure', fontsize=11)
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_title('Strategy 3: Continuous Exposure (30% to 100%)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'2_equity_exposure_{TIMESTAMP}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()
    
    # Plot 3: Drawdown Analysis
    fig, ax = plt.subplots(figsize=(14, 7))
    
    spy_running_max = signals_1['SPY_Cumulative'].cummax()
    spy_dd = (signals_1['SPY_Cumulative'] - spy_running_max) / spy_running_max
    
    s1_running_max = signals_1['Strategy_Cumulative'].cummax()
    s1_dd = (signals_1['Strategy_Cumulative'] - s1_running_max) / s1_running_max
    
    s3_running_max = signals_3['Strategy_Cumulative'].cummax()
    s3_dd = (signals_3['Strategy_Cumulative'] - s3_running_max) / s3_running_max
    
    ax.fill_between(signals_1.index, spy_dd * 100, 0, alpha=0.3, label='SPY Drawdown', color='gray')
    ax.fill_between(signals_1.index, s1_dd * 100, 0, alpha=0.5, label='Strategy 1 Drawdown', color='blue')
    ax.fill_between(signals_3.index, s3_dd * 100, 0, alpha=0.5, label='Strategy 3 Drawdown', color='green')
    ax.set_title('Drawdown Analysis: Risk Comparison', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'3_drawdown_analysis_{TIMESTAMP}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()
    
    # Plot 4: VVIX/VIX Ratio with Signals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Strategy 1 signals
    ax1.plot(signals_1.index, signals_1['Ratio'], alpha=0.6, color='navy', linewidth=1)
    ax1.fill_between(signals_1.index, 0, 
                      signals_1['Signal'] * signals_1['Ratio'].max(), 
                      alpha=0.3, label='Signal Active (50% exposure)', color='red')
    ax1.set_ylabel('VVIX/VIX Ratio', fontsize=11)
    ax1.set_title('Strategy 1: Binary Signal Activation', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    # Strategy 3 continuous
    ax2.plot(signals_3.index, signals_3['Ratio'], alpha=0.6, color='darkgreen', 
             linewidth=1, label='VVIX/VIX Ratio')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(signals_3.index, signals_3['Equity_Exposure'], 
                   alpha=0.8, color='orange', linewidth=1.5, label='Equity Exposure')
    ax2.set_ylabel('VVIX/VIX Ratio', fontsize=11, color='darkgreen')
    ax2_twin.set_ylabel('Equity Exposure', fontsize=11, color='orange')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_title('Strategy 3: Continuous Adjustment', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2_twin.legend(loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'4_signals_over_time_{TIMESTAMP}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {save_path}")
    plt.close()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\n📁 All plots saved to: {RESULTS_DIR}/")
    print(f"   Timestamp: {TIMESTAMP}")
    
    return strategy

if __name__ == "__main__":
    strategy = run_and_compare_strategies()
    
    print("\n💡 Next steps:")
    print("  1. Run validation.py to check for overfitting")
    print("  2. Experiment with different thresholds")
    print("  3. Consider transaction costs and slippage")
    print("  4. Test on recent out-of-sample data")

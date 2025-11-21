"""
Test the improved Strategy 3 with higher base exposure
"""
from strategy import VVIXVIXStrategy
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESTING IMPROVED STRATEGY 3")
print("="*80)

strategy = VVIXVIXStrategy(start_date='2010-01-01')
strategy.fetch_data()

# Test original (corrected) version
print("\n" + "="*80)
print("1. CORRECTED STRATEGY 3 (50-100% exposure)")
print("="*80)
signals_3_corrected = strategy.strategy_3_adaptive_allocation(use_corrected_logic=True)

# Test improved version with 70% base
print("\n" + "="*80)
print("2. IMPROVED STRATEGY 3 (70-100% exposure)")
print("="*80)
signals_3_improved = strategy.strategy_3_improved(base_exposure_min=0.70)

# Test improved version with 80% base
print("\n" + "="*80)
print("3. AGGRESSIVE STRATEGY 3 (80-100% exposure)")
print("="*80)
signals_3_aggressive = strategy.strategy_3_improved(base_exposure_min=0.80)

# Also run Strategy 1 for comparison
print("\n" + "="*80)
print("4. STRATEGY 1 FOR COMPARISON")
print("="*80)
signals_1 = strategy.strategy_1_defensive_overlay()

# Final comparison
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)
comparison = strategy.compare_strategies()
print(comparison.to_string(index=False))

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("\n✅ Best Overall: Strategy 1 (Defensive Equity)")
print("   - Highest Sharpe ratio")
print("   - Lowest drawdown")
print("   - Simple and robust")

metrics = strategy.strategy_results

if 'Adaptive_Allocation_Improved' in metrics:
    improved = metrics['Adaptive_Allocation_Improved']['metrics']
    baseline = metrics['Adaptive_Allocation']['metrics']
    
    print(f"\n✅ Strategy 3 Improved vs Original:")
    print(f"   - Return improvement: {(improved['annualized_return'] - baseline['annualized_return'])*100:+.2f}%")
    print(f"   - Sharpe improvement: {(improved['sharpe_ratio'] - baseline['sharpe_ratio']):+.2f}")
    print(f"   - Avg exposure change: {(improved['avg_exposure'] - baseline['avg_exposure'])*100:+.1f}%")

print("\n💡 Use Strategy 3 Improved (70-100%) if:")
print("   - You want smoother transitions than Strategy 1")
print("   - You can tolerate slightly lower returns for less whipsaw")
print("   - You prefer continuous adjustment over binary signals")

print("\n⚠️  Still prefer Strategy 1 if:")
print("   - Maximum return and Sharpe are priority")
print("   - You can handle binary on/off switches")
print("   - Simplicity and interpretability matter")

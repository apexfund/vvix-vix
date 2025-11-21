"""
Diagnose Strategy 3 implementation to understand performance issues
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy import VVIXVIXStrategy

# Initialize and fetch data
strategy = VVIXVIXStrategy(start_date='2010-01-01')
strategy.fetch_data()
data = strategy.data

print("="*80)
print("STRATEGY 3 DIAGNOSTIC ANALYSIS")
print("="*80)

# Calculate what Strategy 3 does
ratio_25 = data['VVIX_VIX_Ratio'].quantile(0.25)
ratio_75 = data['VVIX_VIX_Ratio'].quantile(0.75)

print(f"\n📊 VVIX/VIX Ratio Distribution:")
print(f"  25th percentile: {ratio_25:.2f}")
print(f"  50th percentile (median): {data['VVIX_VIX_Ratio'].quantile(0.50):.2f}")
print(f"  75th percentile: {ratio_75:.2f}")
print(f"  Mean: {data['VVIX_VIX_Ratio'].mean():.2f}")
print(f"  Std: {data['VVIX_VIX_Ratio'].std():.2f}")

# Normalization
normalized_ratio = (data['VVIX_VIX_Ratio'] - ratio_25) / (ratio_75 - ratio_25)
normalized_ratio = normalized_ratio.clip(0, 1)

# Exposure calculation
equity_exposure = 1.0 - (normalized_ratio * 0.7)

print(f"\n📈 Normalized Ratio Statistics:")
print(f"  Mean: {normalized_ratio.mean():.2f}")
print(f"  Values at p25: {normalized_ratio.quantile(0.25):.2f}")
print(f"  Values at p50: {normalized_ratio.quantile(0.50):.2f}")
print(f"  Values at p75: {normalized_ratio.quantile(0.75):.2f}")

print(f"\n🎯 Equity Exposure Statistics:")
print(f"  Mean: {equity_exposure.mean():.1%}")
print(f"  Min: {equity_exposure.min():.1%}")
print(f"  Max: {equity_exposure.max():.1%}")
print(f"  Median: {equity_exposure.median():.1%}")
print(f"  Std: {equity_exposure.std():.2f}")

# Distribution of exposure
print(f"\n📊 Exposure Distribution:")
print(f"  < 40% exposure: {(equity_exposure < 0.4).sum()} days ({(equity_exposure < 0.4).mean()*100:.1f}%)")
print(f"  40-60% exposure: {((equity_exposure >= 0.4) & (equity_exposure < 0.6)).sum()} days ({((equity_exposure >= 0.4) & (equity_exposure < 0.6)).mean()*100:.1f}%)")
print(f"  60-80% exposure: {((equity_exposure >= 0.6) & (equity_exposure < 0.8)).sum()} days ({((equity_exposure >= 0.6) & (equity_exposure < 0.8)).mean()*100:.1f}%)")
print(f"  > 80% exposure: {(equity_exposure >= 0.8).sum()} days ({(equity_exposure >= 0.8).mean()*100:.1f}%)")

# Compare with Strategy 1
ratio_threshold_s1 = data['VVIX_VIX_Ratio'].quantile(0.90)
signal_s1 = (
    (data['VVIX_VIX_Ratio'] >= ratio_threshold_s1) |
    ((data['VIX'] >= 15) & (data['VVIX'] >= 85))
).astype(int)
exposure_s1 = np.where(signal_s1 == 1, 0.5, 1.0)

print(f"\n🔄 Comparison with Strategy 1:")
print(f"  Strategy 1 avg exposure: {exposure_s1.mean():.1%}")
print(f"  Strategy 3 avg exposure: {equity_exposure.mean():.1%}")
print(f"  Difference: {(exposure_s1.mean() - equity_exposure.mean())*100:.1f}%")

# Market regimes
print(f"\n📉 How does exposure respond to market conditions?")
# High VIX periods (stressed markets)
high_vix = data['VIX'] > 25
print(f"  During high VIX (>25): Avg exposure = {equity_exposure[high_vix].mean():.1%} ({high_vix.sum()} days)")

# Low VIX periods (calm markets)
low_vix = data['VIX'] < 15
print(f"  During low VIX (<15): Avg exposure = {equity_exposure[low_vix].mean():.1%} ({low_vix.sum()} days)")

# Bull market days (SPY > 0)
bull_days = data['SPY_Returns'] > 0
bear_days = data['SPY_Returns'] < 0
print(f"  On up days: Avg exposure = {equity_exposure[bull_days].mean():.1%}")
print(f"  On down days: Avg exposure = {equity_exposure[bear_days].mean():.1%}")

print(f"\n🚨 PROBLEM IDENTIFICATION:")
problem_identified = False

if equity_exposure.mean() < 0.75:
    print(f"  ⚠️  Average exposure ({equity_exposure.mean():.1%}) is TOO LOW!")
    print(f"      In a bull market (2010-2025), being under-invested hurts returns.")
    problem_identified = True

if equity_exposure[low_vix].mean() < 0.85:
    print(f"  ⚠️  Even in calm markets (VIX<15), exposure is only {equity_exposure[low_vix].mean():.1%}")
    print(f"      Should be closer to 100% when markets are calm.")
    problem_identified = True

if not problem_identified:
    print(f"  ✅ Exposure levels look reasonable")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Plot 1: Exposure over time
ax1 = axes[0]
ax1.plot(data.index, equity_exposure, alpha=0.7, color='green', linewidth=1)
ax1.axhline(y=equity_exposure.mean(), color='red', linestyle='--', label=f'Mean: {equity_exposure.mean():.1%}')
ax1.axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, label='Min: 30%')
ax1.axhline(y=1.0, color='blue', linestyle=':', alpha=0.5, label='Max: 100%')
ax1.set_title('Strategy 3: Equity Exposure Over Time', fontsize=12, fontweight='bold')
ax1.set_ylabel('Equity Exposure')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Exposure vs VVIX/VIX Ratio
ax2 = axes[1]
ax2.scatter(data['VVIX_VIX_Ratio'], equity_exposure, alpha=0.3, s=10)
ax2.axvline(x=ratio_25, color='red', linestyle='--', alpha=0.5, label=f'P25: {ratio_25:.2f}')
ax2.axvline(x=ratio_75, color='orange', linestyle='--', alpha=0.5, label=f'P75: {ratio_75:.2f}')
ax2.set_xlabel('VVIX/VIX Ratio')
ax2.set_ylabel('Equity Exposure')
ax2.set_title('Exposure vs VVIX/VIX Ratio Relationship', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Exposure distribution
ax3 = axes[2]
ax3.hist(equity_exposure, bins=50, alpha=0.7, color='green', edgecolor='black')
ax3.axvline(x=equity_exposure.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {equity_exposure.mean():.1%}')
ax3.axvline(x=equity_exposure.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {equity_exposure.median():.1%}')
ax3.set_xlabel('Equity Exposure')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Equity Exposure', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*80)

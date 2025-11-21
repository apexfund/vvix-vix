#!/usr/bin/env python3
"""
VVIX/VIX Ratio Strategy Analysis Runner

This script runs the complete analysis for the VVIX/VIX ratio strategy.
It can be executed directly or imported as a module.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from strategy_utils import (
    calculate_ratio_stats, identify_decline_periods, calculate_performance_metrics,
    perform_statistical_tests, create_strategy_signals, calculate_strategy_returns,
    plot_ratio_analysis, plot_performance_comparison, generate_summary_report
)

warnings.filterwarnings('ignore')

def main():
    """
    Main function to run the complete VVIX/VIX strategy analysis
    """
    print("Starting VVIX/VIX Ratio Strategy Analysis...")
    print("=" * 60)
    
    # Step 1: Data Collection
    print("Step 1: Collecting market data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    
    tickers = ['^VIX', '^VVIX', 'SPY']
    data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    
    print(f"Data collected: {data.shape[0]} days from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Step 2: Data Processing
    print("\nStep 2: Processing data and calculating ratios...")
    
    # Extract and align data
    vix_close = data['Close']['^VIX'].dropna()
    vvix_close = data['Close']['^VVIX'].dropna()
    spy_close = data['Close']['SPY'].dropna()
    
    common_dates = vix_close.index.intersection(vvix_close.index).intersection(spy_close.index)
    
    vix_aligned = vix_close.loc[common_dates]
    vvix_aligned = vvix_close.loc[common_dates]
    spy_aligned = spy_close.loc[common_dates]
    
    # Calculate ratio and returns
    ratio = vvix_aligned / vix_aligned
    spy_returns = spy_aligned.pct_change().dropna()
    spy_returns_5d = spy_aligned.pct_change(5).dropna()
    spy_returns_10d = spy_aligned.pct_change(10).dropna()
    spy_returns_20d = spy_aligned.pct_change(20).dropna()
    
    print(f"Ratio calculated for {len(ratio)} data points")
    
    # Step 3: Ratio Analysis
    print("\nStep 3: Analyzing VVIX/VIX ratio...")
    ratio_stats = calculate_ratio_stats(ratio)
    
    print("Ratio Statistics:")
    for key, value in ratio_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Step 4: Identify Decline Periods
    print("\nStep 4: Identifying decline periods...")
    decline_periods, threshold = identify_decline_periods(ratio, threshold_percentile=0.2)
    
    print(f"Decline threshold (20th percentile): {threshold:.4f}")
    print(f"Number of decline days: {decline_periods.sum()}")
    print(f"Percentage of total days: {decline_periods.sum() / len(decline_periods) * 100:.2f}%")
    
    # Step 5: Performance Analysis
    print("\nStep 5: Analyzing performance during different periods...")
    
    # Calculate returns for different scenarios
    normal_periods = ~decline_periods
    
    decline_returns = spy_returns[decline_periods].dropna()
    normal_returns = spy_returns[normal_periods].dropna()
    
    decline_metrics = calculate_performance_metrics(decline_returns)
    normal_metrics = calculate_performance_metrics(normal_returns)
    
    print("Performance during decline periods:")
    print(f"  Total Return: {decline_metrics['total_return']:.2%}")
    print(f"  Volatility: {decline_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {decline_metrics['sharpe_ratio']:.4f}")
    
    print("Performance during normal periods:")
    print(f"  Total Return: {normal_metrics['total_return']:.2%}")
    print(f"  Volatility: {normal_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {normal_metrics['sharpe_ratio']:.4f}")
    
    # Step 6: Statistical Testing
    print("\nStep 6: Performing statistical tests...")
    statistical_tests = perform_statistical_tests(decline_returns, normal_returns, "Decline vs Normal Periods")
    
    print("Statistical Test Results:")
    print(f"  T-statistic: {statistical_tests['t_statistic']:.4f}")
    print(f"  P-value: {statistical_tests['t_p_value']:.4f}")
    print(f"  Significant: {'Yes' if statistical_tests['significant_t'] else 'No'}")
    print(f"  Effect Size (Cohen's d): {statistical_tests['cohens_d']:.4f}")
    print(f"  Mean Difference: {statistical_tests['mean_diff_pct']:.2f}%")
    
    # Step 7: Strategy Implementation
    print("\nStep 7: Implementing trading strategy...")
    
    strategy_signals = create_strategy_signals(ratio, decline_periods, lookback_days=1)
    strategy_returns = calculate_strategy_returns(strategy_signals, spy_returns)
    
    strategy_metrics = calculate_performance_metrics(strategy_returns)
    benchmark_metrics = calculate_performance_metrics(spy_returns)
    
    print("Strategy Performance:")
    print(f"  Total Return: {strategy_metrics['total_return']:.2%}")
    print(f"  Volatility: {strategy_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {strategy_metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {strategy_metrics['max_drawdown']:.2%}")
    
    print("Benchmark Performance (Buy & Hold):")
    print(f"  Total Return: {benchmark_metrics['total_return']:.2%}")
    print(f"  Volatility: {benchmark_metrics['volatility']:.2%}")
    print(f"  Sharpe Ratio: {benchmark_metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {benchmark_metrics['max_drawdown']:.2%}")
    
    outperformance = strategy_metrics['total_return'] - benchmark_metrics['total_return']
    print(f"\nStrategy Outperformance: {outperformance:.2%}")
    
    # Step 8: Generate Visualizations
    print("\nStep 8: Generating visualizations...")
    
    # Plot ratio analysis
    plot_ratio_analysis(ratio, decline_periods, threshold, "VVIX/VIX Ratio Analysis")
    
    # Plot performance comparison
    plot_performance_comparison(strategy_returns, spy_returns, "Strategy vs Buy & Hold")
    
    # Step 9: Generate Summary Report
    print("\nStep 9: Generating summary report...")
    summary_report = generate_summary_report(ratio_stats, strategy_metrics, statistical_tests)
    
    # Save report to file
    with open('analysis_report.txt', 'w') as f:
        f.write(summary_report)
    
    print("Analysis complete! Summary report saved to 'analysis_report.txt'")
    print("\nKey Findings:")
    print(f"- VVIX/VIX ratio shows {ratio_stats['skewness']:.2f} skewness")
    print(f"- Strategy outperformed benchmark by {outperformance:.2%}")
    print(f"- Statistical significance: {'Yes' if statistical_tests['significant_t'] else 'No'}")
    print(f"- Effect size: {statistical_tests['cohens_d']:.4f}")

if __name__ == "__main__":
    main()


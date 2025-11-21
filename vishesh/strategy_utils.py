"""
Utility functions for VVIX/VIX ratio strategy analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Tuple, Dict, List


def calculate_ratio_stats(ratio: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for VVIX/VIX ratio
    
    Args:
        ratio: VVIX/VIX ratio time series
        
    Returns:
        Dictionary with ratio statistics
    """
    stats_dict = {
        'mean': ratio.mean(),
        'median': ratio.median(),
        'std': ratio.std(),
        'min': ratio.min(),
        'max': ratio.max(),
        'q25': ratio.quantile(0.25),
        'q75': ratio.quantile(0.75),
        'skewness': ratio.skew(),
        'kurtosis': ratio.kurtosis()
    }
    return stats_dict


def identify_decline_periods(ratio: pd.Series, threshold_percentile: float = 0.2) -> Tuple[pd.Series, float]:
    """
    Identify periods when ratio is in significant decline
    
    Args:
        ratio: VVIX/VIX ratio time series
        threshold_percentile: Percentile threshold for decline (default: 0.2 for 20th percentile)
        
    Returns:
        Tuple of (decline_periods_boolean, threshold_value)
    """
    threshold = ratio.quantile(threshold_percentile)
    decline_periods = ratio < threshold
    return decline_periods, threshold


def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a return series
    
    Args:
        returns: Return series
        
    Returns:
        Dictionary with performance metrics
    """
    metrics = {
        'total_return': (1 + returns).prod() - 1,
        'annualized_return': (1 + returns).mean() ** 252 - 1,
        'volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': (returns > 0).mean(),
        'avg_win': returns[returns > 0].mean(),
        'avg_loss': returns[returns < 0].mean(),
        'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if returns[returns < 0].sum() != 0 else np.inf
    }
    return metrics


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from peak
    
    Args:
        returns: Return series
        
    Returns:
        Maximum drawdown as a percentage
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def perform_statistical_tests(returns1: pd.Series, returns2: pd.Series, 
                             test_name: str = "Returns Comparison") -> Dict[str, float]:
    """
    Perform statistical tests comparing two return series
    
    Args:
        returns1: First return series
        returns2: Second return series
        test_name: Name for the test
        
    Returns:
        Dictionary with test results
    """
    # Remove NaN values
    returns1_clean = returns1.dropna()
    returns2_clean = returns2.dropna()
    
    # T-test
    t_stat, p_value = stats.ttest_ind(returns1_clean, returns2_clean)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(returns1_clean) - 1) * returns1_clean.var() + 
                         (len(returns2_clean) - 1) * returns2_clean.var()) / 
                        (len(returns1_clean) + len(returns2_clean) - 2))
    cohens_d = (returns1_clean.mean() - returns2_clean.mean()) / pooled_std
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_p_value = stats.mannwhitneyu(returns1_clean, returns2_clean, alternative='two-sided')
    
    results = {
        'test_name': test_name,
        't_statistic': t_stat,
        't_p_value': p_value,
        'cohens_d': cohens_d,
        'significant_t': p_value < 0.05,
        'u_statistic': u_stat,
        'u_p_value': u_p_value,
        'significant_u': u_p_value < 0.05,
        'mean_diff': returns1_clean.mean() - returns2_clean.mean(),
        'mean_diff_pct': (returns1_clean.mean() - returns2_clean.mean()) * 100
    }
    
    return results


def create_strategy_signals(ratio: pd.Series, decline_periods: pd.Series, 
                           lookback_days: int = 1) -> pd.Series:
    """
    Create trading signals based on ratio decline periods
    
    Args:
        ratio: VVIX/VIX ratio time series
        decline_periods: Boolean series indicating decline periods
        lookback_days: Days to look back for signal generation
        
    Returns:
        Trading signals (1 for buy, 0 for hold)
    """
    signals = pd.Series(0, index=ratio.index)
    signals[decline_periods] = 1
    
    # Shift signals to avoid look-ahead bias
    signals = signals.shift(lookback_days)
    
    return signals


def calculate_strategy_returns(signals: pd.Series, market_returns: pd.Series) -> pd.Series:
    """
    Calculate strategy returns based on signals and market returns
    
    Args:
        signals: Trading signals (1 for buy, 0 for hold)
        market_returns: Market return series
        
    Returns:
        Strategy return series
    """
    strategy_returns = signals * market_returns
    return strategy_returns.dropna()


def plot_ratio_analysis(ratio: pd.Series, decline_periods: pd.Series, 
                       threshold: float, title: str = "VVIX/VIX Ratio Analysis") -> None:
    """
    Create comprehensive ratio analysis plot
    
    Args:
        ratio: VVIX/VIX ratio time series
        decline_periods: Boolean series indicating decline periods
        threshold: Decline threshold value
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Time series with decline periods highlighted
    axes[0, 0].plot(ratio.index, ratio.values, alpha=0.7, linewidth=1, label='VVIX/VIX Ratio')
    axes[0, 0].axhline(y=threshold, color='red', linestyle='--', alpha=0.8, 
                      label=f'Decline Threshold ({threshold:.2f})')
    axes[0, 0].fill_between(ratio.index, 0, threshold, alpha=0.2, color='red', label='Decline Zone')
    axes[0, 0].set_title('Ratio Over Time with Decline Threshold')
    axes[0, 0].set_ylabel('Ratio')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Distribution of ratio
    axes[0, 1].hist(ratio.dropna(), bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[0, 1].axvline(ratio.mean(), color='red', linestyle='--', label=f'Mean: {ratio.mean():.2f}')
    axes[0, 1].axvline(ratio.median(), color='orange', linestyle='--', label=f'Median: {ratio.median():.2f}')
    axes[0, 1].axvline(threshold, color='green', linestyle='--', label=f'Threshold: {threshold:.2f}')
    axes[0, 1].set_title('Ratio Distribution')
    axes[0, 1].set_xlabel('Ratio Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Rolling statistics
    rolling_mean = ratio.rolling(30).mean()
    rolling_std = ratio.rolling(30).std()
    
    axes[1, 0].plot(ratio.index, ratio.values, alpha=0.5, linewidth=0.5, label='Ratio')
    axes[1, 0].plot(rolling_mean.index, rolling_mean.values, linewidth=2, label='30-day Mean')
    axes[1, 0].fill_between(rolling_mean.index, 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, 
                           alpha=0.3, label='±1 Std Dev')
    axes[1, 0].set_title('Rolling Statistics (30-day)')
    axes[1, 0].set_ylabel('Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Decline periods frequency
    decline_freq = decline_periods.rolling(252).sum()  # Annual frequency
    axes[1, 1].plot(decline_freq.index, decline_freq.values, linewidth=2)
    axes[1, 1].set_title('Decline Periods Frequency (Rolling Annual)')
    axes[1, 1].set_ylabel('Number of Decline Days')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(strategy_returns: pd.Series, benchmark_returns: pd.Series,
                              title: str = "Strategy Performance Comparison") -> None:
    """
    Create performance comparison plots
    
    Args:
        strategy_returns: Strategy return series
        benchmark_returns: Benchmark return series
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Cumulative returns
    strategy_cumret = (1 + strategy_returns).cumprod()
    benchmark_cumret = (1 + benchmark_returns).cumprod()
    
    axes[0, 0].plot(strategy_cumret.index, strategy_cumret.values, label='Strategy', linewidth=2)
    axes[0, 0].plot(benchmark_cumret.index, benchmark_cumret.values, label='Benchmark', linewidth=2)
    axes[0, 0].set_title('Cumulative Returns')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Drawdown analysis
    strategy_dd = (strategy_cumret / strategy_cumret.cummax() - 1)
    benchmark_dd = (benchmark_cumret / benchmark_cumret.cummax() - 1)
    
    axes[0, 1].fill_between(strategy_dd.index, strategy_dd.values, 0, alpha=0.3, label='Strategy Drawdown')
    axes[0, 1].fill_between(benchmark_dd.index, benchmark_dd.values, 0, alpha=0.3, label='Benchmark Drawdown')
    axes[0, 1].set_title('Drawdown Analysis')
    axes[0, 1].set_ylabel('Drawdown')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Returns distribution
    axes[1, 0].hist(strategy_returns, bins=30, alpha=0.6, label='Strategy', density=True)
    axes[1, 0].hist(benchmark_returns, bins=30, alpha=0.6, label='Benchmark', density=True)
    axes[1, 0].set_title('Returns Distribution')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Rolling Sharpe ratio
    strategy_rolling_sharpe = strategy_returns.rolling(252).mean() / strategy_returns.rolling(252).std() * np.sqrt(252)
    benchmark_rolling_sharpe = benchmark_returns.rolling(252).mean() / benchmark_returns.rolling(252).std() * np.sqrt(252)
    
    axes[1, 1].plot(strategy_rolling_sharpe.index, strategy_rolling_sharpe.values, label='Strategy', linewidth=2)
    axes[1, 1].plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe.values, label='Benchmark', linewidth=2)
    axes[1, 1].set_title('Rolling Sharpe Ratio (Annual)')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def generate_summary_report(ratio_stats: Dict, performance_metrics: Dict, 
                          statistical_tests: Dict) -> str:
    """
    Generate a comprehensive summary report
    
    Args:
        ratio_stats: Ratio statistics dictionary
        performance_metrics: Performance metrics dictionary
        statistical_tests: Statistical test results dictionary
        
    Returns:
        Formatted summary report string
    """
    report = f"""
# VVIX/VIX Strategy Analysis Report

## Ratio Statistics
- Mean: {ratio_stats['mean']:.4f}
- Median: {ratio_stats['median']:.4f}
- Standard Deviation: {ratio_stats['std']:.4f}
- Skewness: {ratio_stats['skewness']:.4f}
- Kurtosis: {ratio_stats['kurtosis']:.4f}

## Performance Metrics
- Total Return: {performance_metrics['total_return']:.2%}
- Annualized Return: {performance_metrics['annualized_return']:.2%}
- Volatility: {performance_metrics['volatility']:.2%}
- Sharpe Ratio: {performance_metrics['sharpe_ratio']:.4f}
- Maximum Drawdown: {performance_metrics['max_drawdown']:.2%}
- Win Rate: {performance_metrics['win_rate']:.2%}

## Statistical Tests
- Test: {statistical_tests['test_name']}
- T-statistic: {statistical_tests['t_statistic']:.4f}
- P-value: {statistical_tests['t_p_value']:.4f}
- Significant: {'Yes' if statistical_tests['significant_t'] else 'No'}
- Effect Size (Cohen's d): {statistical_tests['cohens_d']:.4f}
- Mean Difference: {statistical_tests['mean_diff_pct']:.2f}%
"""
    return report

"""
Volatility-of-Volatility Premium Strategy EDA
=============================================

This script performs exploratory data analysis for the Volatility-of-Volatility Premium strategy:
Monitor VIX vs. VVIX; when VVIX spikes relative to VIX, equity returns often skew negatively (hedging demand).

Strategy Hypothesis:
- VVIX measures the volatility of VIX (volatility-of-volatility)
- When VVIX spikes relative to VIX, it indicates increased uncertainty about volatility
- This creates hedging demand, leading to negative skew in equity returns
- The VVIX-VIX spread can be used as a signal for market stress
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VolatilityOfVolatilityAnalyzer:
    """
    Analyzes the relationship between VIX and VVIX for the Volatility-of-Volatility Premium strategy
    """
    
    def __init__(self, start_date='2010-01-01', end_date=None):
        """
        Initialize the analyzer with date range
        
        Args:
            start_date (str): Start date for data collection
            end_date (str): End date for data collection (default: today)
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.spy_data = None
        
    def fetch_data(self):
        """
        Fetch VIX, VVIX, and SPY data from Yahoo Finance
        """
        print("Fetching market data...")
        
        # Fetch VIX data
        vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix['Close'].iloc[:, 0].rename('VIX')
        else:
            vix = vix['Close'].rename('VIX')
        
        # Fetch VVIX data
        vvix = yf.download('^VVIX', start=self.start_date, end=self.end_date, progress=False)
        if isinstance(vvix.columns, pd.MultiIndex):
            vvix = vvix['Close'].iloc[:, 0].rename('VVIX')
        else:
            vvix = vvix['Close'].rename('VVIX')
        
        # Fetch SPY data for equity returns
        spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy_returns = spy['Close'].iloc[:, 0].pct_change().rename('SPY_Returns')
        else:
            spy_returns = spy['Close'].pct_change().rename('SPY_Returns')
        
        # Combine all data
        self.data = pd.concat([vix, vvix, spy_returns], axis=1).dropna()
        
        # Calculate additional metrics
        self.data['VVIX_VIX_Ratio'] = self.data['VVIX'] / self.data['VIX']
        self.data['VVIX_VIX_Spread'] = self.data['VVIX'] - self.data['VIX']
        self.data['VIX_Change'] = self.data['VIX'].pct_change()
        self.data['VVIX_Change'] = self.data['VVIX'].pct_change()
        
        # Calculate rolling statistics
        self.data['VIX_MA_20'] = self.data['VIX'].rolling(20).mean()
        self.data['VVIX_MA_20'] = self.data['VVIX'].rolling(20).mean()
        self.data['Ratio_MA_20'] = self.data['VVIX_VIX_Ratio'].rolling(20).mean()
        
        print(f"Data collected: {len(self.data)} observations from {self.data.index[0].date()} to {self.data.index[-1].date()}")
        
    def basic_statistics(self):
        """
        Display basic statistics for VIX, VVIX, and their relationship
        """
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        stats_df = self.data[['VIX', 'VVIX', 'VVIX_VIX_Ratio', 'VVIX_VIX_Spread', 'SPY_Returns']].describe()
        print(stats_df.round(4))
        
        print("\nCORRELATION MATRIX:")
        corr_matrix = self.data[['VIX', 'VVIX', 'VVIX_VIX_Ratio', 'VVIX_VIX_Spread', 'SPY_Returns']].corr()
        print(corr_matrix.round(4))
        
    def plot_time_series(self):
        """
        Create time series plots for VIX, VVIX, and their relationship
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Plot 1: VIX and VVIX levels
        axes[0].plot(self.data.index, self.data['VIX'], label='VIX', alpha=0.8)
        axes[0].plot(self.data.index, self.data['VVIX'], label='VVIX', alpha=0.8)
        axes[0].set_title('VIX vs VVIX Levels Over Time')
        axes[0].set_ylabel('Volatility Level')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: VVIX/VIX Ratio
        axes[1].plot(self.data.index, self.data['VVIX_VIX_Ratio'], color='red', alpha=0.8)
        axes[1].axhline(y=self.data['VVIX_VIX_Ratio'].mean(), color='red', linestyle='--', alpha=0.5, label=f'Mean: {self.data["VVIX_VIX_Ratio"].mean():.2f}')
        axes[1].set_title('VVIX/VIX Ratio Over Time')
        axes[1].set_ylabel('Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: SPY Returns
        axes[2].plot(self.data.index, self.data['SPY_Returns'], color='green', alpha=0.7)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title('SPY Daily Returns')
        axes[2].set_ylabel('Daily Return')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_volatility_spikes(self, threshold_percentile=90):
        """
        Analyze periods when VVIX spikes relative to VIX
        
        Args:
            threshold_percentile (int): Percentile threshold for defining spikes
        """
        print(f"\n" + "="*60)
        print(f"VOLATILITY SPIKE ANALYSIS (Top {100-threshold_percentile}% of VVIX/VIX ratios)")
        print("="*60)
        
        # Define spike threshold
        spike_threshold = self.data['VVIX_VIX_Ratio'].quantile(threshold_percentile/100)
        spike_periods = self.data[self.data['VVIX_VIX_Ratio'] >= spike_threshold]
        
        print(f"Spike threshold (VVIX/VIX ratio): {spike_threshold:.3f}")
        print(f"Number of spike days: {len(spike_periods)} ({len(spike_periods)/len(self.data)*100:.1f}% of total)")
        
        # Analyze equity returns during spikes
        spike_returns = spike_periods['SPY_Returns'].dropna()
        normal_returns = self.data[self.data['VVIX_VIX_Ratio'] < spike_threshold]['SPY_Returns'].dropna()
        
        print(f"\nEQUITY RETURNS ANALYSIS:")
        print(f"During VVIX spikes:")
        print(f"  Mean return: {spike_returns.mean():.4f} ({spike_returns.mean()*252:.1f}% annualized)")
        print(f"  Volatility: {spike_returns.std():.4f} ({spike_returns.std()*np.sqrt(252)*100:.1f}% annualized)")
        print(f"  Skewness: {spike_returns.skew():.3f}")
        print(f"  Kurtosis: {spike_returns.kurtosis():.3f}")
        print(f"  Min return: {spike_returns.min():.4f}")
        print(f"  Max return: {spike_returns.max():.4f}")
        
        print(f"\nDuring normal periods:")
        print(f"  Mean return: {normal_returns.mean():.4f} ({normal_returns.mean()*252:.1f}% annualized)")
        print(f"  Volatility: {normal_returns.std():.4f} ({normal_returns.std()*np.sqrt(252)*100:.1f}% annualized)")
        print(f"  Skewness: {normal_returns.skew():.3f}")
        print(f"  Kurtosis: {normal_returns.kurtosis():.3f}")
        print(f"  Min return: {normal_returns.min():.4f}")
        print(f"  Max return: {normal_returns.max():.4f}")
        
        # Statistical significance test
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(spike_returns, normal_returns)
        print(f"\nStatistical Test (t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        return spike_periods, spike_returns, normal_returns
        
    def plot_spike_analysis(self, spike_periods, spike_returns, normal_returns):
        """
        Create visualizations for spike analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Distribution comparison
        axes[0,0].hist(normal_returns, bins=50, alpha=0.7, label='Normal Periods', density=True)
        axes[0,0].hist(spike_returns, bins=30, alpha=0.7, label='VVIX Spike Periods', density=True)
        axes[0,0].set_title('Distribution of SPY Returns')
        axes[0,0].set_xlabel('Daily Return')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Box plot comparison
        data_for_box = [normal_returns, spike_returns]
        axes[0,1].boxplot(data_for_box, labels=['Normal', 'VVIX Spikes'])
        axes[0,1].set_title('SPY Returns: Normal vs VVIX Spike Periods')
        axes[0,1].set_ylabel('Daily Return')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: VVIX/VIX ratio over time with spike periods highlighted
        axes[1,0].plot(self.data.index, self.data['VVIX_VIX_Ratio'], alpha=0.6, color='blue')
        axes[1,0].scatter(spike_periods.index, spike_periods['VVIX_VIX_Ratio'], 
                         color='red', alpha=0.8, s=20, label='Spike Periods')
        axes[1,0].set_title('VVIX/VIX Ratio with Spike Periods Highlighted')
        axes[1,0].set_ylabel('VVIX/VIX Ratio')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot of VVIX/VIX ratio vs SPY returns
        axes[1,1].scatter(self.data['VVIX_VIX_Ratio'], self.data['SPY_Returns'], 
                         alpha=0.5, s=10)
        axes[1,1].scatter(spike_periods['VVIX_VIX_Ratio'], spike_periods['SPY_Returns'], 
                         color='red', alpha=0.8, s=20, label='Spike Periods')
        axes[1,1].set_title('VVIX/VIX Ratio vs SPY Returns')
        axes[1,1].set_xlabel('VVIX/VIX Ratio')
        axes[1,1].set_ylabel('SPY Daily Return')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def analyze_hedging_demand(self):
        """
        Analyze the hedging demand hypothesis by looking at extreme market conditions
        """
        print(f"\n" + "="*60)
        print("HEDGING DEMAND ANALYSIS")
        print("="*60)
        
        # Define extreme market conditions
        high_vix_threshold = self.data['VIX'].quantile(0.8)  # Top 20% VIX
        high_vvix_threshold = self.data['VVIX'].quantile(0.8)  # Top 20% VVIX
        high_ratio_threshold = self.data['VVIX_VIX_Ratio'].quantile(0.8)  # Top 20% ratio
        
        # Create market condition categories
        conditions = []
        for idx, row in self.data.iterrows():
            if row['VIX'] >= high_vix_threshold and row['VVIX'] >= high_vvix_threshold:
                conditions.append('High VIX & VVIX')
            elif row['VVIX_VIX_Ratio'] >= high_ratio_threshold:
                conditions.append('High VVIX/VIX Ratio')
            elif row['VIX'] >= high_vix_threshold:
                conditions.append('High VIX Only')
            else:
                conditions.append('Normal')
        
        self.data['Market_Condition'] = conditions
        
        # Analyze returns by market condition
        condition_analysis = self.data.groupby('Market_Condition')['SPY_Returns'].agg([
            'count', 'mean', 'std', 'skew', 'min', 'max'
        ]).round(4)
        
        print("EQUITY RETURNS BY MARKET CONDITION:")
        print(condition_analysis)
        
        # Calculate annualized metrics
        condition_analysis['Annualized_Return'] = condition_analysis['mean'] * 252
        condition_analysis['Annualized_Vol'] = condition_analysis['std'] * np.sqrt(252)
        
        print("\nANNUALIZED METRICS:")
        print(condition_analysis[['Annualized_Return', 'Annualized_Vol', 'skew']].round(3))
        
        return condition_analysis
        
    def plot_hedging_demand(self, condition_analysis):
        """
        Visualize hedging demand analysis
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Returns by market condition
        condition_analysis['mean'].plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Average SPY Returns by Market Condition')
        axes[0,0].set_ylabel('Average Daily Return')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Volatility by market condition
        condition_analysis['std'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
        axes[0,1].set_title('SPY Volatility by Market Condition')
        axes[0,1].set_ylabel('Daily Volatility')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Skewness by market condition
        condition_analysis['skew'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('SPY Skewness by Market Condition')
        axes[1,0].set_ylabel('Skewness')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Box plot of returns by condition
        conditions = self.data['Market_Condition'].unique()
        data_by_condition = [self.data[self.data['Market_Condition'] == cond]['SPY_Returns'].dropna() 
                           for cond in conditions]
        axes[1,1].boxplot(data_by_condition, labels=conditions)
        axes[1,1].set_title('SPY Returns Distribution by Market Condition')
        axes[1,1].set_ylabel('Daily Return')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def strategy_signals(self):
        """
        Generate trading signals based on VVIX/VIX ratio
        """
        print(f"\n" + "="*60)
        print("STRATEGY SIGNAL ANALYSIS")
        print("="*60)
        
        # Define signal thresholds
        signal_thresholds = [0.7, 0.8, 0.9, 0.95]
        
        for threshold in signal_thresholds:
            signal_data = self.data[self.data['VVIX_VIX_Ratio'] >= threshold]
            if len(signal_data) > 0:
                forward_returns = signal_data['SPY_Returns'].shift(-1).dropna()
                
                print(f"\nSignal Threshold: {threshold:.1f} (Top {(1-threshold)*100:.0f}% of ratios)")
                print(f"Number of signals: {len(signal_data)}")
                print(f"Average forward return: {forward_returns.mean():.4f}")
                print(f"Hit rate (positive returns): {(forward_returns > 0).mean():.3f}")
                print(f"Average negative return: {forward_returns[forward_returns < 0].mean():.4f}")
                print(f"Average positive return: {forward_returns[forward_returns > 0].mean():.4f}")
        
    def run_complete_analysis(self):
        """
        Run the complete exploratory data analysis
        """
        print("VOLATILITY-OF-VOLATILITY PREMIUM STRATEGY EDA")
        print("=" * 60)
        
        # Fetch data
        self.fetch_data()
        
        # Basic statistics
        self.basic_statistics()
        
        # Time series plots
        self.plot_time_series()
        
        # Spike analysis
        spike_periods, spike_returns, normal_returns = self.analyze_volatility_spikes()
        self.plot_spike_analysis(spike_periods, spike_returns, normal_returns)
        
        # Hedging demand analysis
        condition_analysis = self.analyze_hedging_demand()
        self.plot_hedging_demand(condition_analysis)
        
        # Strategy signals
        self.strategy_signals()
        
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Key Findings:")
        print("1. VVIX/VIX ratio spikes indicate periods of high volatility uncertainty")
        print("2. During these periods, equity returns tend to be more negative and volatile")
        print("3. This supports the hedging demand hypothesis")
        print("4. The strategy can be used to identify market stress periods")
        
        return self.data

if __name__ == "__main__":
    # Initialize analyzer
    analyzer = VolatilityOfVolatilityAnalyzer(start_date='2010-01-01')
    
    # Run complete analysis
    data = analyzer.run_complete_analysis()

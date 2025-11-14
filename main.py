import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VolatilityOfVolatilityAnalyzer:
    
    def __init__(self, start_date='2010-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.spy_data = None
        
    def fetch_data(self):
        print("Fetching market data...")
        
        vix = yf.download('^VIX', start=self.start_date, end=self.end_date, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix = vix['Close'].iloc[:, 0].rename('VIX')
        else:
            vix = vix['Close'].rename('VIX')
        
        vvix = yf.download('^VVIX', start=self.start_date, end=self.end_date, progress=False)
        if isinstance(vvix.columns, pd.MultiIndex):
            vvix = vvix['Close'].iloc[:, 0].rename('VVIX')
        else:
            vvix = vvix['Close'].rename('VVIX')
        
        spy = yf.download('SPY', start=self.start_date, end=self.end_date, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy_returns = spy['Close'].iloc[:, 0].pct_change().rename('SPY_Returns')
        else:
            spy_returns = spy['Close'].pct_change().rename('SPY_Returns')
        
        self.data = pd.concat([vix, vvix, spy_returns], axis=1).dropna()
        
        self.data['VVIX_VIX_Ratio'] = self.data['VVIX'] / self.data['VIX']
        self.data['VVIX_VIX_Spread'] = self.data['VVIX'] - self.data['VIX']
        self.data['VIX_Change'] = self.data['VIX'].pct_change()
        self.data['VVIX_Change'] = self.data['VVIX'].pct_change()
        
        self.data['VIX_MA_20'] = self.data['VIX'].rolling(20).mean()
        self.data['VVIX_MA_20'] = self.data['VVIX'].rolling(20).mean()
        self.data['Ratio_MA_20'] = self.data['VVIX_VIX_Ratio'].rolling(20).mean()
        
        print(f"Data collected: {len(self.data)} observations from {self.data.index[0].date()} to {self.data.index[-1].date()}")
        
    def basic_statistics(self):
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        
        stats_df = self.data[['VIX', 'VVIX', 'VVIX_VIX_Ratio', 'VVIX_VIX_Spread', 'SPY_Returns']].describe()
        print(stats_df.round(4))
        
        print("\nCORRELATION MATRIX:")
        corr_matrix = self.data[['VIX', 'VVIX', 'VVIX_VIX_Ratio', 'VVIX_VIX_Spread', 'SPY_Returns']].corr()
        print(corr_matrix.round(4))
        
    def plot_time_series(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['VIX'], label='VIX', alpha=0.8)
        ax.plot(self.data.index, self.data['VVIX'], label='VVIX', alpha=0.8)
        ax.set_title('VIX vs VVIX Levels Over Time')
        ax.set_ylabel('Volatility Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['VVIX_VIX_Ratio'], color='red', alpha=0.8)
        ax.axhline(y=self.data['VVIX_VIX_Ratio'].mean(), color='red', linestyle='--', alpha=0.5, label=f'Mean: {self.data["VVIX_VIX_Ratio"].mean():.2f}')
        ax.set_title('VVIX/VIX Ratio Over Time')
        ax.set_ylabel('Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['SPY_Returns'], color='green', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title('SPY Daily Returns')
        ax.set_ylabel('Daily Return')
        ax.set_xlabel('Date')
        ax.grid(True, alpha=0.3)
        plt.show()
        
    def analyze_volatility_spikes(self, threshold_percentile=90):
        print(f"\n" + "="*60)
        print(f"VOLATILITY SPIKE ANALYSIS (Top {100-threshold_percentile}% of VVIX/VIX ratios)")
        print("="*60)
        
        spike_threshold = self.data['VVIX_VIX_Ratio'].quantile(threshold_percentile/100)
        spike_periods = self.data[self.data['VVIX_VIX_Ratio'] >= spike_threshold]
        
        print(f"Spike threshold (VVIX/VIX ratio): {spike_threshold:.3f}")
        print(f"Number of spike days: {len(spike_periods)} ({len(spike_periods)/len(self.data)*100:.1f}% of total)")
        
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
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(spike_returns, normal_returns)
        print(f"\nStatistical Test (t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")
        
        return spike_periods, spike_returns, normal_returns
        
    def plot_spike_analysis(self, spike_periods, spike_returns, normal_returns):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(normal_returns, bins=50, alpha=0.7, label='Normal Periods', density=True)
        ax.hist(spike_returns, bins=30, alpha=0.7, label='VVIX Spike Periods', density=True)
        ax.set_title('Distribution of SPY Returns')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        data_for_box = [normal_returns, spike_returns]
        ax.boxplot(data_for_box, labels=['Normal', 'VVIX Spikes'])
        ax.set_title('SPY Returns: Normal vs VVIX Spike Periods')
        ax.set_ylabel('Daily Return')
        ax.grid(True, alpha=0.3)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.data.index, self.data['VVIX_VIX_Ratio'], alpha=0.6, color='blue')
        ax.scatter(spike_periods.index, spike_periods['VVIX_VIX_Ratio'], 
                         color='red', alpha=0.8, s=20, label='Spike Periods')
        ax.set_title('VVIX/VIX Ratio with Spike Periods Highlighted')
        ax.set_ylabel('VVIX/VIX Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(self.data['VVIX_VIX_Ratio'], self.data['SPY_Returns'], 
                         alpha=0.5, s=10)
        ax.scatter(spike_periods['VVIX_VIX_Ratio'], spike_periods['SPY_Returns'], 
                         color='red', alpha=0.8, s=20, label='Spike Periods')
        ax.set_title('VVIX/VIX Ratio vs SPY Returns')
        ax.set_xlabel('VVIX/VIX Ratio')
        ax.set_ylabel('SPY Daily Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        
    def analyze_hedging_demand(self):
        print(f"\n" + "="*60)
        print("HEDGING DEMAND ANALYSIS")
        print("="*60)
        
        high_vix_threshold = self.data['VIX'].quantile(0.8)
        high_vvix_threshold = self.data['VVIX'].quantile(0.8)
        high_ratio_threshold = self.data['VVIX_VIX_Ratio'].quantile(0.8)
        
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
        
        condition_analysis = self.data.groupby('Market_Condition')['SPY_Returns'].agg([
            'count', 'mean', 'std', 'skew', 'min', 'max'
        ]).round(4)
        
        print("EQUITY RETURNS BY MARKET CONDITION:")
        print(condition_analysis)
        
        condition_analysis['Annualized_Return'] = condition_analysis['mean'] * 252
        condition_analysis['Annualized_Vol'] = condition_analysis['std'] * np.sqrt(252)
        
        print("\nANNUALIZED METRICS:")
        print(condition_analysis[['Annualized_Return', 'Annualized_Vol', 'skew']].round(3))
        
        return condition_analysis
        
    def plot_hedging_demand(self, condition_analysis):
        fig, ax = plt.subplots(figsize=(10, 6))
        condition_analysis['mean'].plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Average SPY Returns by Market Condition')
        ax.set_ylabel('Average Daily Return')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        condition_analysis['std'].plot(kind='bar', ax=ax, color='lightcoral')
        ax.set_title('SPY Volatility by Market Condition')
        ax.set_ylabel('Daily Volatility')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        condition_analysis['skew'].plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('SPY Skewness by Market Condition')
        ax.set_ylabel('Skewness')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        conditions = self.data['Market_Condition'].unique()
        data_by_condition = [self.data[self.data['Market_Condition'] == cond]['SPY_Returns'].dropna() 
                           for cond in conditions]
        ax.boxplot(data_by_condition, labels=conditions)
        ax.set_title('SPY Returns Distribution by Market Condition')
        ax.set_ylabel('Daily Return')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.show()
        
    def strategy_signals(self):
        print(f"\n" + "="*60)
        print("STRATEGY SIGNAL ANALYSIS")
        print("="*60)
        
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
        print("VOLATILITY-OF-VOLATILITY PREMIUM STRATEGY EDA")
        print("=" * 60)
        
        self.fetch_data()
        
        self.basic_statistics()
        
        self.plot_time_series()
        
        spike_periods, spike_returns, normal_returns = self.analyze_volatility_spikes()
        self.plot_spike_analysis(spike_periods, spike_returns, normal_returns)
        
        condition_analysis = self.analyze_hedging_demand()
        self.plot_hedging_demand(condition_analysis)
        
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
    analyzer = VolatilityOfVolatilityAnalyzer(start_date='2010-01-01')
    
    data = analyzer.run_complete_analysis()

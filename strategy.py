import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class VVIXVIXStrategy:
    
    def __init__(self, start_date='2010-01-01', end_date=None, initial_capital=100000):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital
        self.data = None
        self.strategy_results = {}
        
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
            spy_price = spy['Close'].iloc[:, 0].rename('SPY_Price')
            spy_returns = spy['Close'].iloc[:, 0].pct_change().rename('SPY_Returns')
        else:
            spy_price = spy['Close'].rename('SPY_Price')
            spy_returns = spy['Close'].pct_change().rename('SPY_Returns')
        
        self.data = pd.concat([vix, vvix, spy_price, spy_returns], axis=1).dropna()
        
        self.data['VVIX_VIX_Ratio'] = self.data['VVIX'] / self.data['VIX']
        self.data['VVIX_VIX_Spread'] = self.data['VVIX'] - self.data['VIX']
        
        self.data['Ratio_MA_20'] = self.data['VVIX_VIX_Ratio'].rolling(20).mean()
        self.data['Ratio_MA_60'] = self.data['VVIX_VIX_Ratio'].rolling(60).mean()
        self.data['Ratio_Std_20'] = self.data['VVIX_VIX_Ratio'].rolling(20).std()
        self.data['VIX_MA_20'] = self.data['VIX'].rolling(20).mean()
        self.data['VVIX_MA_20'] = self.data['VVIX'].rolling(20).mean()
        
        print(f"Data collected: {len(self.data)} observations")
        
    def strategy_1_defensive_overlay(self, 
                                     ratio_threshold_percentile=90,
                                     vix_threshold=15,
                                     vvix_threshold=85,
                                     equity_reduction=0.5):
        print("\n" + "="*60)
        print("STRATEGY 1: DEFENSIVE EQUITY OVERLAY")
        print("="*60)
        
        ratio_threshold = self.data['VVIX_VIX_Ratio'].quantile(ratio_threshold_percentile/100)
        
        print(f"Thresholds: Ratio >= {ratio_threshold:.2f}, VIX >= {vix_threshold}, VVIX >= {vvix_threshold}")
        
        signals = pd.DataFrame(index=self.data.index)
        signals['SPY_Returns'] = self.data['SPY_Returns']
        signals['VIX'] = self.data['VIX']
        signals['VVIX'] = self.data['VVIX']
        signals['Ratio'] = self.data['VVIX_VIX_Ratio']
        
        signals['Signal'] = (
            (self.data['VVIX_VIX_Ratio'] >= ratio_threshold) |
            ((self.data['VIX'] >= vix_threshold) & (self.data['VVIX'] >= vvix_threshold))
        ).astype(int)
        
        signals['Equity_Exposure'] = np.where(signals['Signal'] == 1, 
                                              1 - equity_reduction, 
                                              1.0)
        
        signals['Strategy_Returns'] = signals['SPY_Returns'] * signals['Equity_Exposure']
        
        signals['SPY_Cumulative'] = (1 + signals['SPY_Returns']).cumprod()
        signals['Strategy_Cumulative'] = (1 + signals['Strategy_Returns']).cumprod()
        
        spy_total_return = signals['SPY_Cumulative'].iloc[-1] - 1
        strategy_total_return = signals['Strategy_Cumulative'].iloc[-1] - 1
        
        spy_annualized = (1 + spy_total_return) ** (252 / len(signals)) - 1
        strategy_annualized = (1 + strategy_total_return) ** (252 / len(signals)) - 1
        
        spy_sharpe = signals['SPY_Returns'].mean() / signals['SPY_Returns'].std() * np.sqrt(252)
        strategy_sharpe = signals['Strategy_Returns'].mean() / signals['Strategy_Returns'].std() * np.sqrt(252)
        
        spy_max_dd = self.calculate_max_drawdown(signals['SPY_Cumulative'])
        strategy_max_dd = self.calculate_max_drawdown(signals['Strategy_Cumulative'])
        
        print(f"\nSignal Statistics:")
        print(f"  Signal days: {signals['Signal'].sum()} ({signals['Signal'].sum()/len(signals)*100:.1f}% of time)")
        print(f"  Average signal duration: {self.calculate_avg_duration(signals['Signal']):.1f} days")
        
        print(f"\nPerformance Metrics:")
        print(f"  SPY Total Return: {spy_total_return*100:.2f}%")
        print(f"  Strategy Total Return: {strategy_total_return*100:.2f}%")
        print(f"  SPY Annualized Return: {spy_annualized*100:.2f}%")
        print(f"  Strategy Annualized Return: {strategy_annualized*100:.2f}%")
        print(f"  SPY Sharpe Ratio: {spy_sharpe:.2f}")
        print(f"  Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"  SPY Max Drawdown: {spy_max_dd*100:.2f}%")
        print(f"  Strategy Max Drawdown: {strategy_max_dd*100:.2f}%")
        
        self.strategy_results['Defensive_Overlay'] = {
            'signals': signals,
            'metrics': {
                'total_return': strategy_total_return,
                'annualized_return': strategy_annualized,
                'sharpe_ratio': strategy_sharpe,
                'max_drawdown': strategy_max_dd,
                'signal_days': signals['Signal'].sum()
            }
        }
        
        return signals
    
    def strategy_2_hedging_with_vix(self,
                                   ratio_threshold_percentile=90,
                                   vix_hedge_threshold=18):
        print("\n" + "="*60)
        print("STRATEGY 2: VIX HEDGING STRATEGY")
        print("="*60)
        
        ratio_threshold = self.data['VVIX_VIX_Ratio'].quantile(ratio_threshold_percentile/100)
        
        print(f"Thresholds: Ratio >= {ratio_threshold:.2f}, VIX >= {vix_hedge_threshold}")
        
        signals = pd.DataFrame(index=self.data.index)
        signals['SPY_Returns'] = self.data['SPY_Returns']
        signals['VIX'] = self.data['VIX']
        signals['VVIX'] = self.data['VVIX']
        signals['Ratio'] = self.data['VVIX_VIX_Ratio']
        
        signals['Hedge_Signal'] = (
            (self.data['VVIX_VIX_Ratio'] >= ratio_threshold) &
            (self.data['VIX'] >= vix_hedge_threshold)
        ).astype(int)
        
        hedge_allocation = 0.1
        vix_beta = -0.3
        
        signals['SPY_Allocation'] = 1.0 - (signals['Hedge_Signal'] * hedge_allocation)
        signals['VIX_Hedge_Returns'] = signals['SPY_Returns'] * vix_beta * signals['Hedge_Signal']
        
        signals['Strategy_Returns'] = (
            signals['SPY_Returns'] * signals['SPY_Allocation'] + 
            signals['VIX_Hedge_Returns'] * hedge_allocation
        )
        
        signals['SPY_Cumulative'] = (1 + signals['SPY_Returns']).cumprod()
        signals['Strategy_Cumulative'] = (1 + signals['Strategy_Returns']).cumprod()
        
        spy_total_return = signals['SPY_Cumulative'].iloc[-1] - 1
        strategy_total_return = signals['Strategy_Cumulative'].iloc[-1] - 1
        
        spy_annualized = (1 + spy_total_return) ** (252 / len(signals)) - 1
        strategy_annualized = (1 + strategy_total_return) ** (252 / len(signals)) - 1
        
        spy_sharpe = signals['SPY_Returns'].mean() / signals['SPY_Returns'].std() * np.sqrt(252)
        strategy_sharpe = signals['Strategy_Returns'].mean() / signals['Strategy_Returns'].std() * np.sqrt(252)
        
        spy_max_dd = self.calculate_max_drawdown(signals['SPY_Cumulative'])
        strategy_max_dd = self.calculate_max_drawdown(signals['Strategy_Cumulative'])
        
        print(f"\nSignal Statistics:")
        print(f"  Hedge signal days: {signals['Hedge_Signal'].sum()} ({signals['Hedge_Signal'].sum()/len(signals)*100:.1f}%)")
        print(f"  Average hedge duration: {self.calculate_avg_duration(signals['Hedge_Signal']):.1f} days")
        
        print(f"\nPerformance Metrics:")
        print(f"  SPY Total Return: {spy_total_return*100:.2f}%")
        print(f"  Strategy Total Return: {strategy_total_return*100:.2f}%")
        print(f"  SPY Annualized Return: {spy_annualized*100:.2f}%")
        print(f"  Strategy Annualized Return: {strategy_annualized*100:.2f}%")
        print(f"  SPY Sharpe Ratio: {spy_sharpe:.2f}")
        print(f"  Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"  SPY Max Drawdown: {spy_max_dd*100:.2f}%")
        print(f"  Strategy Max Drawdown: {strategy_max_dd*100:.2f}%")
        
        self.strategy_results['VIX_Hedging'] = {
            'signals': signals,
            'metrics': {
                'total_return': strategy_total_return,
                'annualized_return': strategy_annualized,
                'sharpe_ratio': strategy_sharpe,
                'max_drawdown': strategy_max_dd,
                'hedge_days': signals['Hedge_Signal'].sum()
            }
        }
        
        return signals
    
    def strategy_3_adaptive_allocation(self,
                                      ratio_threshold_percentile=75,
                                      volatility_lookback=20):
        print("\n" + "="*60)
        print("STRATEGY 3: ADAPTIVE ASSET ALLOCATION")
        print("="*60)
        
        ratio_25 = self.data['VVIX_VIX_Ratio'].quantile(0.25)
        ratio_75 = self.data['VVIX_VIX_Ratio'].quantile(0.75)
        
        normalized_ratio = (self.data['VVIX_VIX_Ratio'] - ratio_25) / (ratio_75 - ratio_25)
        normalized_ratio = normalized_ratio.clip(0, 1)
        
        signals = pd.DataFrame(index=self.data.index)
        signals['SPY_Returns'] = self.data['SPY_Returns']
        signals['Ratio'] = self.data['VVIX_VIX_Ratio']
        signals['Normalized_Ratio'] = normalized_ratio
        
        signals['Equity_Exposure'] = 1.0 - (normalized_ratio * 0.7)
        
        signals['Strategy_Returns'] = signals['SPY_Returns'] * signals['Equity_Exposure']
        
        signals['SPY_Cumulative'] = (1 + signals['SPY_Returns']).cumprod()
        signals['Strategy_Cumulative'] = (1 + signals['Strategy_Returns']).cumprod()
        
        spy_total_return = signals['SPY_Cumulative'].iloc[-1] - 1
        strategy_total_return = signals['Strategy_Cumulative'].iloc[-1] - 1
        
        spy_annualized = (1 + spy_total_return) ** (252 / len(signals)) - 1
        strategy_annualized = (1 + strategy_total_return) ** (252 / len(signals)) - 1
        
        spy_sharpe = signals['SPY_Returns'].mean() / signals['SPY_Returns'].std() * np.sqrt(252)
        strategy_sharpe = signals['Strategy_Returns'].mean() / signals['Strategy_Returns'].std() * np.sqrt(252)
        
        spy_max_dd = self.calculate_max_drawdown(signals['SPY_Cumulative'])
        strategy_max_dd = self.calculate_max_drawdown(signals['Strategy_Cumulative'])
        
        print(f"\nExposure Statistics:")
        print(f"  Average equity exposure: {signals['Equity_Exposure'].mean():.1%}")
        print(f"  Min equity exposure: {signals['Equity_Exposure'].min():.1%}")
        print(f"  Max equity exposure: {signals['Equity_Exposure'].max():.1%}")
        
        print(f"\nPerformance Metrics:")
        print(f"  SPY Total Return: {spy_total_return*100:.2f}%")
        print(f"  Strategy Total Return: {strategy_total_return*100:.2f}%")
        print(f"  SPY Annualized Return: {spy_annualized*100:.2f}%")
        print(f"  Strategy Annualized Return: {strategy_annualized*100:.2f}%")
        print(f"  SPY Sharpe Ratio: {spy_sharpe:.2f}")
        print(f"  Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"  SPY Max Drawdown: {spy_max_dd*100:.2f}%")
        print(f"  Strategy Max Drawdown: {strategy_max_dd*100:.2f}%")
        
        self.strategy_results['Adaptive_Allocation'] = {
            'signals': signals,
            'metrics': {
                'total_return': strategy_total_return,
                'annualized_return': strategy_annualized,
                'sharpe_ratio': strategy_sharpe,
                'max_drawdown': strategy_max_dd,
                'avg_exposure': signals['Equity_Exposure'].mean()
            }
        }
        
        return signals
    
    def calculate_max_drawdown(self, cumulative_returns):
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def calculate_avg_duration(self, signal_series):
        signal_changes = signal_series.diff().fillna(0)
        durations = []
        current_duration = 0
        
        for change in signal_changes:
            if change != 0:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 1
            else:
                current_duration += 1
        
        if durations:
            return np.mean(durations)
        return 0
    
    def plot_strategy_performance(self, strategy_name='Defensive_Overlay'):
        if strategy_name not in self.strategy_results:
            print(f"Strategy {strategy_name} not found")
            return
        
        signals = self.strategy_results[strategy_name]['signals']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(signals.index, signals['SPY_Cumulative'], label='SPY', alpha=0.8)
        ax.plot(signals.index, signals['Strategy_Cumulative'], label='Strategy', alpha=0.8)
        ax.set_title(f'{strategy_name} - Cumulative Returns')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        
        if 'Signal' in signals.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(signals.index, signals['Ratio'], alpha=0.6, label='VVIX/VIX Ratio')
            ax.fill_between(signals.index, 0, signals['Signal'] * signals['Ratio'].max(), 
                                alpha=0.3, label='Signal Active', color='red')
            ax.set_title('Signal Visualization')
            ax.set_ylabel('VVIX/VIX Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.show()
        elif 'Hedge_Signal' in signals.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(signals.index, signals['Ratio'], alpha=0.6, label='VVIX/VIX Ratio')
            ax.fill_between(signals.index, 0, signals['Hedge_Signal'] * signals['Ratio'].max(), 
                                alpha=0.3, label='Hedge Active', color='orange')
            ax.set_title('Hedge Signal Visualization')
            ax.set_ylabel('VVIX/VIX Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.show()
        elif 'Equity_Exposure' in signals.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(signals.index, signals['Equity_Exposure'], alpha=0.8, color='green')
            ax.set_title('Equity Exposure Over Time')
            ax.set_ylabel('Exposure')
            ax.grid(True, alpha=0.3)
            plt.show()
        
        spy_running_max = signals['SPY_Cumulative'].cummax()
        spy_dd = (signals['SPY_Cumulative'] - spy_running_max) / spy_running_max
        
        strategy_running_max = signals['Strategy_Cumulative'].cummax()
        strategy_dd = (signals['Strategy_Cumulative'] - strategy_running_max) / strategy_running_max
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.fill_between(signals.index, spy_dd, 0, alpha=0.5, label='SPY Drawdown')
        ax.fill_between(signals.index, strategy_dd, 0, alpha=0.5, label='Strategy Drawdown')
        ax.set_title('Drawdown Analysis')
        ax.set_ylabel('Drawdown')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
    
    def compare_strategies(self):
        print("\n" + "="*60)
        print("STRATEGY COMPARISON")
        print("="*60)
        
        comparison = pd.DataFrame({
            'Strategy': [],
            'Total Return': [],
            'Annualized Return': [],
            'Sharpe Ratio': [],
            'Max Drawdown': []
        })
        
        for name, results in self.strategy_results.items():
            metrics = results['metrics']
            comparison = pd.concat([comparison, pd.DataFrame({
                'Strategy': [name],
                'Total Return': [metrics.get('total_return', 0) * 100],
                'Annualized Return': [metrics.get('annualized_return', 0) * 100],
                'Sharpe Ratio': [metrics.get('sharpe_ratio', 0)],
                'Max Drawdown': [metrics.get('max_drawdown', 0) * 100]
            })], ignore_index=True)
        
        print(comparison.to_string(index=False))
        
        return comparison
    
    def strategy_1_optimized(self,
                            ratio_threshold_percentile=95,
                            vix_threshold=18,
                            vvix_threshold=100,
                            equity_reduction=0.5,
                            min_signal_duration=3):
        print("\n" + "="*60)
        print("STRATEGY 1 OPTIMIZED: SELECTIVE DEFENSIVE OVERLAY")
        print("="*60)
        
        ratio_threshold = self.data['VVIX_VIX_Ratio'].quantile(ratio_threshold_percentile/100)
        
        print(f"Thresholds: Ratio >= {ratio_threshold:.2f}, VIX >= {vix_threshold}, VVIX >= {vvix_threshold}")
        print(f"Minimum signal duration: {min_signal_duration} days")
        
        signals = pd.DataFrame(index=self.data.index)
        signals['SPY_Returns'] = self.data['SPY_Returns']
        signals['VIX'] = self.data['VIX']
        signals['VVIX'] = self.data['VVIX']
        signals['Ratio'] = self.data['VVIX_VIX_Ratio']
        
        initial_signal = (
            (self.data['VVIX_VIX_Ratio'] >= ratio_threshold) &
            (self.data['VIX'] >= vix_threshold) &
            (self.data['VVIX'] >= vvix_threshold)
        ).astype(int)
        
        signals['Signal'] = 0
        rolling_sum = initial_signal.rolling(window=min_signal_duration, min_periods=min_signal_duration).sum()
        signals['Signal'] = (rolling_sum >= min_signal_duration).astype(int)
        
        signals['Equity_Exposure'] = np.where(signals['Signal'] == 1, 
                                              1 - equity_reduction, 
                                              1.0)
        
        signals['Strategy_Returns'] = signals['SPY_Returns'] * signals['Equity_Exposure']
        signals['SPY_Cumulative'] = (1 + signals['SPY_Returns']).cumprod()
        signals['Strategy_Cumulative'] = (1 + signals['Strategy_Returns']).cumprod()
        
        spy_total_return = signals['SPY_Cumulative'].iloc[-1] - 1
        strategy_total_return = signals['Strategy_Cumulative'].iloc[-1] - 1
        
        spy_annualized = (1 + spy_total_return) ** (252 / len(signals)) - 1
        strategy_annualized = (1 + strategy_total_return) ** (252 / len(signals)) - 1
        
        spy_sharpe = signals['SPY_Returns'].mean() / signals['SPY_Returns'].std() * np.sqrt(252)
        strategy_sharpe = signals['Strategy_Returns'].mean() / signals['Strategy_Returns'].std() * np.sqrt(252)
        
        spy_max_dd = self.calculate_max_drawdown(signals['SPY_Cumulative'])
        strategy_max_dd = self.calculate_max_drawdown(signals['Strategy_Cumulative'])
        
        print(f"\nSignal Statistics:")
        print(f"  Signal days: {signals['Signal'].sum()} ({signals['Signal'].sum()/len(signals)*100:.1f}% of time)")
        print(f"  Average signal duration: {self.calculate_avg_duration(signals['Signal']):.1f} days")
        
        print(f"\nPerformance Metrics:")
        print(f"  SPY Total Return: {spy_total_return*100:.2f}%")
        print(f"  Strategy Total Return: {strategy_total_return*100:.2f}%")
        print(f"  SPY Annualized Return: {spy_annualized*100:.2f}%")
        print(f"  Strategy Annualized Return: {strategy_annualized*100:.2f}%")
        print(f"  SPY Sharpe Ratio: {spy_sharpe:.2f}")
        print(f"  Strategy Sharpe Ratio: {strategy_sharpe:.2f}")
        print(f"  SPY Max Drawdown: {spy_max_dd*100:.2f}%")
        print(f"  Strategy Max Drawdown: {strategy_max_dd*100:.2f}%")
        
        self.strategy_results['Defensive_Overlay_Optimized'] = {
            'signals': signals,
            'metrics': {
                'total_return': strategy_total_return,
                'annualized_return': strategy_annualized,
                'sharpe_ratio': strategy_sharpe,
                'max_drawdown': strategy_max_dd,
                'signal_days': signals['Signal'].sum()
            }
        }
        
        return signals
    
    def run_all_strategies(self):
        self.fetch_data()
        
        self.strategy_1_defensive_overlay()
        self.strategy_1_optimized()
        self.strategy_2_hedging_with_vix()
        self.strategy_3_adaptive_allocation()
        
        comparison = self.compare_strategies()
        
        return comparison

if __name__ == "__main__":
    strategy = VVIXVIXStrategy(start_date='2010-01-01')
    
    comparison = strategy.run_all_strategies()
    
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE CHARTS")
    print("="*60)
    
    if 'Defensive_Overlay' in strategy.strategy_results:
        strategy.plot_strategy_performance('Defensive_Overlay')
    if 'VIX_Hedging' in strategy.strategy_results:
        strategy.plot_strategy_performance('VIX_Hedging')

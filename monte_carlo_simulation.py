import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

from strategy import VVIXVIXStrategy

class DefensiveEquityMonteCarlo:
    
    def __init__(self, start_date='2010-01-01', end_date=None, n_simulations=1000):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.n_simulations = n_simulations
        self.strategy = VVIXVIXStrategy(start_date, end_date)
        self.actual_data = None
        self.simulation_results = []
        
    def fetch_actual_data(self):
        print("Fetching actual market data...")
        self.strategy.fetch_data()
        self.actual_data = self.strategy.data.copy()
        print(f"Data collected: {len(self.actual_data)} observations")
        
    def run_defensive_strategy(self, data, ratio_threshold_percentile=90,
                              vix_threshold=15, vvix_threshold=85, equity_reduction=0.5):
        ratio_threshold = data['VVIX_VIX_Ratio'].quantile(ratio_threshold_percentile/100)
        
        signals = pd.DataFrame(index=data.index)
        signals['SPY_Returns'] = data['SPY_Returns']
        signals['VIX'] = data['VIX']
        signals['VVIX'] = data['VVIX']
        signals['Ratio'] = data['VVIX_VIX_Ratio']
        
        signals['Signal'] = (
            (data['VVIX_VIX_Ratio'] >= ratio_threshold) |
            ((data['VIX'] >= vix_threshold) & (data['VVIX'] >= vvix_threshold))
        ).astype(int)
        
        signals['Equity_Exposure'] = np.where(signals['Signal'] == 1, 
                                              1 - equity_reduction, 
                                              1.0)
        
        signals['Strategy_Returns'] = signals['SPY_Returns'] * signals['Equity_Exposure']
        signals['SPY_Cumulative'] = (1 + signals['SPY_Returns']).cumprod()
        signals['Strategy_Cumulative'] = (1 + signals['Strategy_Returns']).cumprod()
        
        spy_total_return = signals['SPY_Cumulative'].iloc[-1] - 1
        strategy_total_return = signals['Strategy_Cumulative'].iloc[-1] - 1
        
        n_days = len(signals)
        spy_annualized = (1 + spy_total_return) ** (252 / n_days) - 1
        strategy_annualized = (1 + strategy_total_return) ** (252 / n_days) - 1
        
        spy_sharpe = signals['SPY_Returns'].mean() / signals['SPY_Returns'].std() * np.sqrt(252) if signals['SPY_Returns'].std() > 0 else 0
        strategy_sharpe = signals['Strategy_Returns'].mean() / signals['Strategy_Returns'].std() * np.sqrt(252) if signals['Strategy_Returns'].std() > 0 else 0
        
        spy_max_dd = self.calculate_max_drawdown(signals['SPY_Cumulative'])
        strategy_max_dd = self.calculate_max_drawdown(signals['Strategy_Cumulative'])
        
        return {
            'spy_annualized_return': spy_annualized,
            'strategy_annualized_return': strategy_annualized,
            'spy_sharpe': spy_sharpe,
            'strategy_sharpe': strategy_sharpe,
            'spy_max_drawdown': spy_max_dd,
            'strategy_max_drawdown': strategy_max_dd,
            'signal_frequency': signals['Signal'].mean(),
            'excess_return': strategy_annualized - spy_annualized,
            'excess_sharpe': strategy_sharpe - spy_sharpe,
            'dd_improvement': spy_max_dd - strategy_max_dd,
            'signals': signals
        }
    
    def calculate_max_drawdown(self, cumulative_returns):
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()
    
    def generate_random_returns(self, method='bootstrap', volatility_scale=1.0):
        actual_returns = self.actual_data['SPY_Returns'].dropna()
        mean_return = actual_returns.mean()
        std_return = actual_returns.std()
        n_days = len(self.actual_data)
        
        if method == 'bootstrap':
            random_returns = np.random.choice(actual_returns.values, size=n_days, replace=True)
            
        elif method == 'gaussian':
            random_returns = np.random.normal(mean_return, std_return * volatility_scale, n_days)
            
        elif method == 'garch':
            random_returns = np.zeros(n_days)
            current_vol = std_return
            
            for i in range(n_days):
                if i > 0:
                    recent_return = abs(random_returns[i-1])
                    current_vol = 0.95 * current_vol + 0.05 * recent_return * 10
                
                random_returns[i] = np.random.normal(mean_return, current_vol * volatility_scale)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return pd.Series(random_returns, index=self.actual_data.index)
    
    def generate_random_volatility_indicators(self, method='shuffle', correlation_preserve=True):
        actual_vix = self.actual_data['VIX'].values.copy()
        actual_vvix = self.actual_data['VVIX'].values.copy()
        n_days = len(self.actual_data)
        
        if method == 'shuffle':
            random_vix = actual_vix.copy()
            random_vvix = actual_vvix.copy()
            np.random.shuffle(random_vix)
            np.random.shuffle(random_vvix)
            
        elif method == 'independent':
            vix_mean, vix_std = actual_vix.mean(), actual_vix.std()
            vvix_mean, vvix_std = actual_vvix.mean(), actual_vvix.std()
            
            random_vix = np.random.normal(vix_mean, vix_std, n_days)
            random_vvix = np.random.normal(vvix_mean, vvix_std, n_days)
            
            random_vix = np.maximum(random_vix, 5)
            random_vvix = np.maximum(random_vvix, 50)
            
        elif method == 'correlated':
            correlation = np.corrcoef(actual_vix, actual_vvix)[0, 1]
            
            random_vix = np.random.normal(actual_vix.mean(), actual_vix.std(), n_days)
            random_vvix = correlation * random_vix + np.sqrt(1 - correlation**2) * np.random.normal(
                actual_vvix.mean(), actual_vvix.std(), n_days)
            
            random_vix = (random_vix - random_vix.mean()) / random_vix.std() * actual_vix.std() + actual_vix.mean()
            random_vvix = (random_vvix - random_vvix.mean()) / random_vvix.std() * actual_vvix.std() + actual_vvix.mean()
            
            random_vix = np.maximum(random_vix, 5)
            random_vvix = np.maximum(random_vvix, 50)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return pd.Series(random_vix, index=self.actual_data.index), \
               pd.Series(random_vvix, index=self.actual_data.index)
    
    def run_single_simulation(self, return_method='bootstrap', volatility_method='shuffle',
                             randomize_returns=True, randomize_volatility=True):
        if randomize_returns:
            random_returns = self.generate_random_returns(method=return_method)
        else:
            random_returns = self.actual_data['SPY_Returns']
        
        if randomize_volatility:
            random_vix, random_vvix = self.generate_random_volatility_indicators(method=volatility_method)
        else:
            random_vix = self.actual_data['VIX']
            random_vvix = self.actual_data['VVIX']
        
        synthetic_data = pd.DataFrame(index=self.actual_data.index)
        synthetic_data['SPY_Returns'] = random_returns
        synthetic_data['VIX'] = random_vix
        synthetic_data['VVIX'] = random_vvix
        synthetic_data['VVIX_VIX_Ratio'] = random_vvix / random_vix
        
        synthetic_data['Ratio_MA_20'] = synthetic_data['VVIX_VIX_Ratio'].rolling(20).mean()
        synthetic_data['Ratio_MA_60'] = synthetic_data['VVIX_VIX_Ratio'].rolling(60).mean()
        synthetic_data['Ratio_Std_20'] = synthetic_data['VVIX_VIX_Ratio'].rolling(20).std()
        synthetic_data['VIX_MA_20'] = synthetic_data['VIX'].rolling(20).mean()
        synthetic_data['VVIX_MA_20'] = synthetic_data['VVIX'].rolling(20).mean()
        
        synthetic_data = synthetic_data.dropna()
        
        if len(synthetic_data) < 100:
            return None
        
        result = self.run_defensive_strategy(synthetic_data)
        
        return result
    
    def run_monte_carlo(self, return_method='bootstrap', volatility_method='shuffle',
                       randomize_returns=True, randomize_volatility=True):
        np.random.seed(int(time.time() * 1000000) % 2**32)
        
        print("\n" + "="*60)
        print("MONTE CARLO SIMULATION: DEFENSIVE EQUITY STRATEGY")
        print("="*60)
        print(f"Number of simulations: {self.n_simulations}")
        print(f"Return randomization: {return_method if randomize_returns else 'None'}")
        print(f"Volatility randomization: {volatility_method if randomize_volatility else 'None'}")
        print("\nRunning simulations...")
        
        self.fetch_actual_data()
        
        print("\nRunning actual strategy on real data...")
        actual_result = self.run_defensive_strategy(self.actual_data)
        
        successful_sims = 0
        for i in range(self.n_simulations):
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{self.n_simulations} simulations completed...")
            
            result = self.run_single_simulation(
                return_method=return_method,
                volatility_method=volatility_method,
                randomize_returns=randomize_returns,
                randomize_volatility=randomize_volatility
            )
            
            if result is not None:
                self.simulation_results.append(result)
                successful_sims += 1
        
        print(f"\nCompleted {successful_sims} successful simulations out of {self.n_simulations}")
        
        self.analyze_results(actual_result)
        
        return actual_result, self.simulation_results
    
    def analyze_results(self, actual_result):
        if len(self.simulation_results) == 0:
            return
        
        sim_returns = [r['strategy_annualized_return'] for r in self.simulation_results]
        sim_sharpes = [r['strategy_sharpe'] for r in self.simulation_results]
        sim_max_dds = [r['strategy_max_drawdown'] for r in self.simulation_results]
        sim_excess_returns = [r['excess_return'] for r in self.simulation_results]
        sim_excess_sharpes = [r['excess_sharpe'] for r in self.simulation_results]
        sim_dd_improvements = [r['dd_improvement'] for r in self.simulation_results]
        
        sim_returns = np.array(sim_returns)
        sim_sharpes = np.array(sim_sharpes)
        sim_max_dds = np.array(sim_max_dds)
        sim_excess_returns = np.array(sim_excess_returns)
        sim_excess_sharpes = np.array(sim_excess_sharpes)
        sim_dd_improvements = np.array(sim_dd_improvements)
        
        self.plot_results(actual_result, sim_returns, sim_sharpes, sim_max_dds,
                         sim_excess_returns, sim_excess_sharpes, sim_dd_improvements)
    
    def plot_results(self, actual_result, sim_returns, sim_sharpes, sim_max_dds,
                    sim_excess_returns, sim_excess_sharpes, sim_dd_improvements):
        
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.hist(sim_returns * 100, bins=50, alpha=0.6, label='Random Simulations')
        ax1.axvline(x=actual_result['strategy_annualized_return'] * 100, 
                   color='r', linestyle='--', linewidth=2, label='Actual Strategy')
        ax1.set_title('Annualized Return Distribution')
        ax1.set_xlabel('Annualized Return (%)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.hist(sim_sharpes, bins=50, alpha=0.6, label='Random Simulations')
        ax2.axvline(x=actual_result['strategy_sharpe'], 
                   color='r', linestyle='--', linewidth=2, label='Actual Strategy')
        ax2.set_title('Sharpe Ratio Distribution')
        ax2.set_xlabel('Sharpe Ratio')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.hist(sim_max_dds * 100, bins=50, alpha=0.6, label='Random Simulations')
        ax3.axvline(x=actual_result['strategy_max_drawdown'] * 100, 
                   color='r', linestyle='--', linewidth=2, label='Actual Strategy')
        ax3.set_title('Max Drawdown Distribution')
        ax3.set_xlabel('Max Drawdown (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        ax4.hist(sim_excess_returns * 100, bins=50, alpha=0.6, label='Random Simulations')
        ax4.axvline(x=actual_result['excess_return'] * 100, 
                   color='r', linestyle='--', linewidth=2, label='Actual Strategy')
        ax4.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
        ax4.set_title('Excess Return vs SPY')
        ax4.set_xlabel('Excess Return (%)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.hist(sim_excess_sharpes, bins=50, alpha=0.6, label='Random Simulations')
        ax5.axvline(x=actual_result['excess_sharpe'], 
                   color='r', linestyle='--', linewidth=2, label='Actual Strategy')
        ax5.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
        ax5.set_title('Excess Sharpe vs SPY')
        ax5.set_xlabel('Excess Sharpe')
        ax5.set_ylabel('Frequency')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        ax6.hist(sim_dd_improvements * 100, bins=50, alpha=0.6, label='Random Simulations')
        ax6.axvline(x=actual_result['dd_improvement'] * 100, 
                   color='r', linestyle='--', linewidth=2, label='Actual Strategy')
        ax6.axvline(x=0, color='k', linestyle='-', linewidth=1, alpha=0.3)
        ax6.set_title('Drawdown Improvement vs SPY')
        ax6.set_xlabel('DD Improvement (%)')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig7, ax7 = plt.subplots(figsize=(8, 5))
        ax7.scatter(sim_returns * 100, sim_sharpes, alpha=0.4, s=8, label='Random Simulations')
        ax7.scatter(actual_result['strategy_annualized_return'] * 100, 
                   actual_result['strategy_sharpe'], 
                   color='r', s=80, marker='*', label='Actual Strategy', zorder=5)
        ax7.set_title('Return vs Sharpe Ratio')
        ax7.set_xlabel('Annualized Return (%)')
        ax7.set_ylabel('Sharpe Ratio')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig8, ax8 = plt.subplots(figsize=(8, 5))
        ax8.scatter(sim_returns * 100, sim_max_dds * 100, alpha=0.4, s=8, label='Random Simulations')
        ax8.scatter(actual_result['strategy_annualized_return'] * 100, 
                   actual_result['strategy_max_drawdown'] * 100, 
                   color='r', s=80, marker='*', label='Actual Strategy', zorder=5)
        ax8.set_title('Return vs Max Drawdown')
        ax8.set_xlabel('Annualized Return (%)')
        ax8.set_ylabel('Max Drawdown (%)')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        plt.tight_layout()
        
        fig9, ax9 = plt.subplots(figsize=(8, 5))
        sorted_returns = np.sort(sim_returns * 100)
        percentiles = np.linspace(0, 100, len(sorted_returns))
        ax9.plot(sorted_returns, percentiles, label='Random Simulations CDF', linewidth=2)
        actual_percentile = (sim_returns < actual_result['strategy_annualized_return']).mean() * 100
        ax9.axvline(x=actual_result['strategy_annualized_return'] * 100, 
                   color='r', linestyle='--', linewidth=2, label=f'Actual ({actual_percentile:.1f}th percentile)')
        ax9.set_title('Cumulative Distribution: Returns')
        ax9.set_xlabel('Annualized Return (%)')
        ax9.set_ylabel('Percentile')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.show()

if __name__ == "__main__":
    mc = DefensiveEquityMonteCarlo(start_date='2010-01-01', n_simulations=1000)
    
    print("\n" + "="*60)
    print("TESTING WITH BOOTSTRAP RETURNS AND SHUFFLED VOLATILITY")
    print("="*60)
    actual_result, sim_results = mc.run_monte_carlo(
        return_method='bootstrap',
        volatility_method='shuffle',
        randomize_returns=True,
        randomize_volatility=True
    )

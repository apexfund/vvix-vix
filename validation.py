import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from strategy import VVIXVIXStrategy

class StrategyValidator:
    
    def __init__(self, start_date='2010-01-01', end_date=None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.strategy = VVIXVIXStrategy(start_date, end_date)
        
    def walk_forward_analysis(self, train_years=3, test_years=1, min_periods=2):
        print("\n" + "="*60)
        print("WALK-FORWARD ANALYSIS")
        print("="*60)
        print(f"Training period: {train_years} years, Testing period: {test_years} years")
        
        self.strategy.fetch_data()
        data = self.strategy.data
        
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        
        results = []
        current_date = start_date
        
        while current_date < end_date:
            train_end = current_date + pd.DateOffset(years=train_years)
            test_start = train_end
            test_end = test_start + pd.DateOffset(years=test_years)
            
            if test_end > end_date:
                break
            
            train_data = data[(data.index >= current_date) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]
            
            if len(train_data) < 252 * min_periods or len(test_data) < 252:
                current_date += pd.DateOffset(years=1)
                continue
            
            ratio_threshold_train = train_data['VVIX_VIX_Ratio'].quantile(0.90)
            
            test_signals = (
                (test_data['VVIX_VIX_Ratio'] >= ratio_threshold_train) |
                ((test_data['VIX'] >= 15) & (test_data['VVIX'] >= 85))
            ).astype(int)
            
            test_equity_exposure = np.where(test_signals == 1, 0.5, 1.0)
            test_strategy_returns = test_data['SPY_Returns'] * test_equity_exposure
            
            train_ratio_threshold = train_data['VVIX_VIX_Ratio'].quantile(0.90)
            train_signals = (
                (train_data['VVIX_VIX_Ratio'] >= train_ratio_threshold) |
                ((train_data['VIX'] >= 15) & (train_data['VVIX'] >= 85))
            ).astype(int)
            train_equity_exposure = np.where(train_signals == 1, 0.5, 1.0)
            train_strategy_returns = train_data['SPY_Returns'] * train_equity_exposure
            
            train_annualized = (1 + train_strategy_returns.mean()) ** 252 - 1
            test_annualized = (1 + test_strategy_returns.mean()) ** 252 - 1
            train_sharpe = train_strategy_returns.mean() / train_strategy_returns.std() * np.sqrt(252)
            test_sharpe = test_strategy_returns.mean() / test_strategy_returns.std() * np.sqrt(252)
            
            results.append({
                'Train_Start': current_date,
                'Train_End': train_end,
                'Test_Start': test_start,
                'Test_End': test_end,
                'Train_Return': train_annualized,
                'Test_Return': test_annualized,
                'Train_Sharpe': train_sharpe,
                'Test_Sharpe': test_sharpe,
                'Ratio_Threshold': ratio_threshold_train
            })
            
            current_date += pd.DateOffset(years=1)
        
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            print(f"\nNumber of periods tested: {len(results_df)}")
            print(f"\nAverage Training Return: {results_df['Train_Return'].mean()*100:.2f}%")
            print(f"Average Testing Return: {results_df['Test_Return'].mean()*100:.2f}%")
            print(f"Return Degradation: {(results_df['Train_Return'].mean() - results_df['Test_Return'].mean())*100:.2f}%")
            print(f"\nAverage Training Sharpe: {results_df['Train_Sharpe'].mean():.2f}")
            print(f"Average Testing Sharpe: {results_df['Test_Sharpe'].mean():.2f}")
            print(f"Sharpe Degradation: {results_df['Train_Sharpe'].mean() - results_df['Test_Sharpe'].mean():.2f}")
            
            consistency = (results_df['Test_Return'] > 0).mean()
            print(f"\nPositive Return Consistency: {consistency*100:.1f}% of periods")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(results_df['Test_Return'], label='Test Period Returns', marker='o')
            ax.axhline(y=results_df['Test_Return'].mean(), color='r', linestyle='--', 
                      label=f'Mean: {results_df["Test_Return"].mean()*100:.2f}%')
            ax.set_title('Walk-Forward Analysis: Out-of-Sample Returns')
            ax.set_ylabel('Annualized Return')
            ax.set_xlabel('Period')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.show()
            
        return results_df
    
    def parameter_stability_test(self):
        print("\n" + "="*60)
        print("PARAMETER STABILITY TEST")
        print("="*60)
        
        self.strategy.fetch_data()
        
        ratio_thresholds = [85, 87.5, 90, 92.5, 95]
        vix_thresholds = [12, 15, 18, 20, 22]
        vvix_thresholds = [80, 85, 90, 95, 100]
        
        results = []
        
        for ratio_pct in ratio_thresholds:
            for vix_thresh in vix_thresholds:
                for vvix_thresh in vvix_thresholds:
                    ratio_threshold = self.strategy.data['VVIX_VIX_Ratio'].quantile(ratio_pct/100)
                    
                    signals = (
                        (self.strategy.data['VVIX_VIX_Ratio'] >= ratio_threshold) |
                        ((self.strategy.data['VIX'] >= vix_thresh) & 
                         (self.strategy.data['VVIX'] >= vvix_thresh))
                    ).astype(int)
                    
                    equity_exposure = np.where(signals == 1, 0.5, 1.0)
                    strategy_returns = self.strategy.data['SPY_Returns'] * equity_exposure
                    
                    total_return = (1 + strategy_returns).cumprod().iloc[-1] - 1
                    annualized = (1 + total_return) ** (252 / len(strategy_returns)) - 1
                    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
                    
                    results.append({
                        'Ratio_Percentile': ratio_pct,
                        'VIX_Threshold': vix_thresh,
                        'VVIX_Threshold': vvix_thresh,
                        'Annualized_Return': annualized,
                        'Sharpe_Ratio': sharpe,
                        'Signal_Frequency': signals.mean()
                    })
        
        results_df = pd.DataFrame(results)
        
        best_params = results_df.loc[results_df['Annualized_Return'].idxmax()]
        base_params = results_df[
            (results_df['Ratio_Percentile'] == 90) & 
            (results_df['VIX_Threshold'] == 15) & 
            (results_df['VVIX_Threshold'] == 85)
        ].iloc[0]
        
        print(f"\nBase Parameters (Current):")
        print(f"  Return: {base_params['Annualized_Return']*100:.2f}%")
        print(f"  Sharpe: {base_params['Sharpe_Ratio']:.2f}")
        
        print(f"\nBest Parameters (Optimized):")
        print(f"  Ratio Percentile: {best_params['Ratio_Percentile']}")
        print(f"  VIX Threshold: {best_params['VIX_Threshold']}")
        print(f"  VVIX Threshold: {best_params['VVIX_Threshold']}")
        print(f"  Return: {best_params['Annualized_Return']*100:.2f}%")
        print(f"  Sharpe: {best_params['Sharpe_Ratio']:.2f}")
        
        improvement = (best_params['Annualized_Return'] - base_params['Annualized_Return']) * 100
        print(f"\nImprovement over base: {improvement:.2f}%")
        
        if improvement > 5:
            print("WARNING: Large improvement suggests potential overfitting!")
        
        return results_df
    
    def monte_carlo_validation(self, n_simulations=1000):
        print("\n" + "="*60)
        print("MONTE CARLO VALIDATION")
        print("="*60)
        print(f"Running {n_simulations} simulations...")
        
        self.strategy.fetch_data()
        data = self.strategy.data
        
        ratio_threshold = data['VVIX_VIX_Ratio'].quantile(0.90)
        
        actual_signals = (
            (data['VVIX_VIX_Ratio'] >= ratio_threshold) |
            ((data['VIX'] >= 15) & (data['VVIX'] >= 85))
        ).astype(int)
        
        actual_equity_exposure = np.where(actual_signals == 1, 0.5, 1.0)
        actual_strategy_returns = data['SPY_Returns'] * actual_equity_exposure
        actual_total_return = (1 + actual_strategy_returns).cumprod().iloc[-1] - 1
        actual_annualized = (1 + actual_total_return) ** (252 / len(actual_strategy_returns)) - 1
        
        random_returns = []
        random_sharpes = []
        
        for _ in range(n_simulations):
            random_signals = np.random.randint(0, 2, len(data))
            random_equity_exposure = np.where(random_signals == 1, 0.5, 1.0)
            random_strategy_returns = data['SPY_Returns'] * random_equity_exposure
            
            random_total_return = (1 + random_strategy_returns).cumprod().iloc[-1] - 1
            random_annualized = (1 + random_total_return) ** (252 / len(random_strategy_returns)) - 1
            random_sharpe = random_strategy_returns.mean() / random_strategy_returns.std() * np.sqrt(252)
            
            random_returns.append(random_annualized)
            random_sharpes.append(random_sharpe)
        
        random_returns = np.array(random_returns)
        random_sharpes = np.array(random_sharpes)
        
        percentile_return = (random_returns < actual_annualized).mean() * 100
        percentile_sharpe = (random_sharpes < actual_strategy_returns.mean() / actual_strategy_returns.std() * np.sqrt(252)).mean() * 100
        
        print(f"\nActual Strategy Performance:")
        print(f"  Annualized Return: {actual_annualized*100:.2f}%")
        print(f"  Sharpe Ratio: {actual_strategy_returns.mean() / actual_strategy_returns.std() * np.sqrt(252):.2f}")
        
        print(f"\nRandom Strategy Comparison:")
        print(f"  Mean Random Return: {random_returns.mean()*100:.2f}%")
        print(f"  Std Random Return: {random_returns.std()*100:.2f}%")
        print(f"  Actual Return Percentile: {percentile_return:.1f}%")
        print(f"  Actual Sharpe Percentile: {percentile_sharpe:.1f}%")
        
        if percentile_return < 5 or percentile_return > 95:
            print("\nWARNING: Strategy performance is in extreme tail of random distribution!")
            print("This suggests the strategy may be overfit or genuinely exceptional.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(random_returns * 100, bins=50, alpha=0.7, label='Random Strategies', density=True)
        ax.axvline(x=actual_annualized*100, color='r', linestyle='--', linewidth=2, 
                  label=f'Actual Strategy: {actual_annualized*100:.2f}%')
        ax.set_title('Monte Carlo Validation: Strategy vs Random')
        ax.set_xlabel('Annualized Return (%)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.show()
        
        return {
            'actual_return': actual_annualized,
            'random_returns': random_returns,
            'percentile': percentile_return
        }
    
    def out_of_sample_test(self, split_date='2020-01-01'):
        print("\n" + "="*60)
        print("OUT-OF-SAMPLE TEST")
        print("="*60)
        
        self.strategy.fetch_data()
        data = self.strategy.data
        
        split = pd.to_datetime(split_date)
        
        in_sample = data[data.index < split]
        out_of_sample = data[data.index >= split]
        
        if len(in_sample) == 0 or len(out_of_sample) == 0:
            print("Insufficient data for split")
            return None
        
        ratio_threshold_is = in_sample['VVIX_VIX_Ratio'].quantile(0.90)
        
        is_signals = (
            (in_sample['VVIX_VIX_Ratio'] >= ratio_threshold_is) |
            ((in_sample['VIX'] >= 15) & (in_sample['VVIX'] >= 85))
        ).astype(int)
        
        oos_signals = (
            (out_of_sample['VVIX_VIX_Ratio'] >= ratio_threshold_is) |
            ((out_of_sample['VIX'] >= 15) & (out_of_sample['VVIX'] >= 85))
        ).astype(int)
        
        is_exposure = np.where(is_signals == 1, 0.5, 1.0)
        oos_exposure = np.where(oos_signals == 1, 0.5, 1.0)
        
        is_returns = in_sample['SPY_Returns'] * is_exposure
        oos_returns = out_of_sample['SPY_Returns'] * oos_exposure
        
        is_total = (1 + is_returns).cumprod().iloc[-1] - 1
        oos_total = (1 + oos_returns).cumprod().iloc[-1] - 1
        
        is_annualized = (1 + is_total) ** (252 / len(is_returns)) - 1
        oos_annualized = (1 + oos_total) ** (252 / len(oos_returns)) - 1
        
        is_sharpe = is_returns.mean() / is_returns.std() * np.sqrt(252)
        oos_sharpe = oos_returns.mean() / oos_returns.std() * np.sqrt(252)
        
        print(f"In-Sample Period: {in_sample.index[0].date()} to {in_sample.index[-1].date()}")
        print(f"Out-of-Sample Period: {out_of_sample.index[0].date()} to {out_of_sample.index[-1].date()}")
        print(f"\nIn-Sample Performance:")
        print(f"  Annualized Return: {is_annualized*100:.2f}%")
        print(f"  Sharpe Ratio: {is_sharpe:.2f}")
        print(f"\nOut-of-Sample Performance:")
        print(f"  Annualized Return: {oos_annualized*100:.2f}%")
        print(f"  Sharpe Ratio: {oos_sharpe:.2f}")
        print(f"\nPerformance Degradation:")
        print(f"  Return: {(is_annualized - oos_annualized)*100:.2f}%")
        print(f"  Sharpe: {is_sharpe - oos_sharpe:.2f}")
        
        degradation_ratio = oos_annualized / is_annualized if is_annualized > 0 else 0
        if degradation_ratio < 0.5:
            print("\nWARNING: Significant performance degradation in out-of-sample period!")
            print("This may indicate overfitting.")
        
        return {
            'in_sample': {'return': is_annualized, 'sharpe': is_sharpe},
            'out_of_sample': {'return': oos_annualized, 'sharpe': oos_sharpe}
        }
    
    def run_all_validations(self):
        print("="*60)
        print("STRATEGY OVERFITTING VALIDATION")
        print("="*60)
        
        oos_result = self.out_of_sample_test()
        wf_result = self.walk_forward_analysis()
        param_result = self.parameter_stability_test()
        mc_result = self.monte_carlo_validation()
        
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        if oos_result:
            degradation = abs(oos_result['in_sample']['return'] - oos_result['out_of_sample']['return']) / abs(oos_result['in_sample']['return'])
            if degradation > 0.5:
                print("OVERFITTING RISK: High")
            elif degradation > 0.3:
                print("OVERFITTING RISK: Medium")
            else:
                print("OVERFITTING RISK: Low")
        
        return {
            'out_of_sample': oos_result,
            'walk_forward': wf_result,
            'parameter_stability': param_result,
            'monte_carlo': mc_result
        }

if __name__ == "__main__":
    validator = StrategyValidator(start_date='2010-01-01')
    results = validator.run_all_validations()


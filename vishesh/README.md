# VVIX/VIX Ratio Strategy

This folder contains a quantitative strategy that investigates the relationship between the VVIX/VIX ratio and market returns, specifically examining how declining ratios correlate with increased market returns.

## Strategy Overview

The VVIX/VIX ratio strategy is based on the hypothesis that when the VVIX/VIX ratio decreases significantly, it may signal reduced volatility expectations and increased market confidence, potentially leading to higher market returns.

### Key Components

- **VVIX (CBOE VVIX Index)**: Measures the volatility of VIX itself
- **VIX (CBOE Volatility Index)**: Measures market volatility expectations
- **Ratio Analysis**: VVIX/VIX ratio to identify market sentiment shifts
- **Market Returns**: SPY (S&P 500 ETF) for market performance analysis

## Files Structure

```
vvix_vix_strategy/
├── vvix_vix_analysis.ipynb    # Main analysis notebook
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── data/                     # Data storage (created when running)
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the Jupyter notebook:
```bash
jupyter notebook vvix_vix_analysis.ipynb
```

## Analysis Components

### 1. Data Collection
- Downloads VIX, VVIX, and SPY data from Yahoo Finance
- Covers 5 years of historical data for comprehensive analysis

### 2. Ratio Calculation
- Calculates VVIX/VIX ratio
- Analyzes ratio distribution and behavior over time

### 3. Correlation Analysis
- Examines correlations between ratio and market returns
- Tests different time horizons (1d, 5d, 10d, 20d)

### 4. Strategy Implementation
- Identifies periods of significant ratio decline (bottom 20th percentile)
- Compares returns during decline periods vs normal periods
- Implements simple buy signal when ratio is low

### 5. Performance Analysis
- Statistical significance testing
- Performance metrics comparison
- Drawdown analysis
- Sharpe ratio calculations

## Key Visualizations

1. **Time Series Analysis**: VVIX/VIX ratio over time with decline thresholds
2. **Correlation Heatmaps**: Relationship between ratio and market returns
3. **Distribution Analysis**: Returns distribution during different market conditions
4. **Strategy Performance**: Cumulative returns and drawdown analysis
5. **Statistical Testing**: T-tests and effect size calculations

## Strategy Logic

1. **Signal Generation**: Buy SPY when VVIX/VIX ratio falls below 20th percentile
2. **Risk Management**: Simple binary signal (buy/hold)
3. **Performance Measurement**: Compare against buy-and-hold strategy

## Usage

1. Open the Jupyter notebook
2. Run all cells to execute the complete analysis
3. Review the generated plots and statistical results
4. Modify parameters as needed for different analysis periods

## Key Metrics

- **Correlation Analysis**: Measures relationship strength between ratio and returns
- **Statistical Significance**: T-tests to validate strategy effectiveness
- **Performance Metrics**: Total return, volatility, Sharpe ratio
- **Outperformance**: Strategy returns vs buy-and-hold benchmark

## Future Enhancements

- More sophisticated entry/exit rules
- Risk management features
- Different market regime analysis
- Transaction cost considerations
- Portfolio optimization integration

## Notes

- This is a research strategy for educational purposes
- Past performance does not guarantee future results
- Consider transaction costs and market impact in live trading
- Regular rebalancing and monitoring recommended

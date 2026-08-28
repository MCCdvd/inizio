# Model Comparison Framework

A comprehensive framework for comparing RL trading agents (DQN, PPO, A3C) with identical parameters.

## Features

- **Unified Testing**: Run all 3 algorithms with identical configuration
- **Comprehensive Metrics**: Collects 12+ performance metrics for each model
- **Multiple Reports**: CSV and HTML comparison reports
- **Reproducible**: Uses seed for deterministic results
- **Easy to Use**: Simple command-line interface

## Usage

### Basic Comparison (1000 episodes on AAPL)
```bash
cd trading-agent
python compare_models.py --stock AAPL --episodes 1000
```

### Using Shell Script Wrapper
```bash
./compare_models.sh --stock AAPL --episodes 1000 --flat-fee 4.0
```

### Custom Parameters
```bash
python compare_models.py \
  --stock AAPL \
  --episodes 500 \
  --flat-fee 4.0 \
  --seed 42 \
  --output-dir ./my_comparison_results
```

## Output

The framework generates:
1. **CSV Report** (`comparison_report.csv`) - Tab-separated comparison table
2. **HTML Report** (`comparison_report.html`) - Visual report for web viewing
3. **Individual Model Results** - Organized by algorithm in separate directories
   - `dqn_AAPL_1000ep/` - DQN results
   - `ppo_AAPL_1000ep/` - PPO results
   - `a3c_AAPL_1000ep/` - A3C results

## Metrics Compared

| Metric | Description |
|--------|-------------|
| **Final Portfolio** | End-of-training portfolio value |
| **Return %** | Total return percentage |
| **Sharpe Ratio** | Risk-adjusted return |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Total Trades** | Number of trades executed |
| **Win Rate %** | Percentage of profitable trades |
| **Profit Factor** | Ratio of gains to losses |
| **Avg Win %** | Average winning trade size |
| **Avg Loss %** | Average losing trade size |
| **Trades/Day** | Trading frequency |

## Example Output

```
======================================================================
MODEL COMPARISON SUMMARY
======================================================================
     Model Final Portfolio  Return % Sharpe Ratio  Max Drawdown Total Trades Win Rate % Profit Factor
       DQN      $10,250.50    +2.51%       0.1234      -0.0892        150      62.00%         1.85x
       PPO      $10,188.14    +1.88%       0.0587      -0.0682        217      57.14%         1.43x
       A3C      $10,320.75    +3.21%       0.1567      -0.0765        198      59.60%         1.92x

======================================================================
BEST PERFORMING MODELS
======================================================================
🏆 Best Return: A3C (+3.21%)
🏆 Best Sharpe Ratio: A3C (0.1567)
🏆 Best Drawdown Control: PPO (-0.0682)
```

## API Usage

```python
from compare_models import ModelComparator

# Create comparator
comparator = ModelComparator(
    output_base='./results',
    stock='AAPL',
    episodes=1000,
    flat_fee=4.0,
    seed=42
)

# Run all models
results = comparator.run_all_models()

# Generate reports
comparator.print_comparison_summary()
df = comparator.generate_comparison_report()
comparator.save_comparison_report(df)
comparator.generate_html_report()
```

## Command Reference

```
optional arguments:
  -h, --help              show this help message and exit
  --stock STOCK           Stock symbol (default: AAPL)
  --episodes EPISODES     Number of training episodes (default: 1000)
  --flat-fee FLAT_FEE     Trading fee per trade (default: 4.0)
  --seed SEED             Random seed for reproducibility (default: 42)
  --output-dir OUTPUT_DIR Output directory for results (default: ./model_comparison)
```

## Best Practices

1. **Use same seed** - Ensures reproducible comparison across runs
2. **Run overnight** - Each model takes time to train (1000 episodes)
3. **Compare metrics** - Look at Sharpe/Sortino, not just return
4. **Monitor convergence** - Check if models are still improving at 1000 episodes
5. **Validate results** - Run on multiple stocks/periods for robust conclusions

## Notes

- Training can take 10-30 minutes per model depending on episode count
- Results are saved locally in organized directories
- HTML report provides visual comparison dashboard
- Each model is trained independently with fresh initialization

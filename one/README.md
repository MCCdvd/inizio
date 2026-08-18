# one: Reinforcement Learning Trading Agent

Enhanced `one/` implementation with DQN + PPO training, reward shaping, and backtesting.

## Structure

```
one/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── dqn_agent.py
│   └── ppo_agent.py
├── env/
│   └── trading_env.py
├── features/
│   └── indicators.py
├── utils/
│   ├── logger.py
│   ├── metrics.py
│   └── plot.py
├── train.py
├── backtest.py
└── config.yaml
```

## Key Improvements

- Built-in logging via `utils/logger.py` (no external logging module dependency)
- State includes normalized OHLCV + indicators + portfolio context
- Reward shaping includes return, drawdown penalty, trade efficiency, and risk-adjusted signal
- DQN upgraded to Double-DQN style target network and checkpointing
- New PPO implementation with clipped PPO objective + GAE
- Backtesting with return, Sharpe, max drawdown, win rate, trade count, and average trade profit
- Metrics exported to CSV and optional plots for rewards/equity/actions

## Setup

```bash
pip install tensorflow pandas numpy matplotlib yfinance pyyaml
```

## Train

```bash
python one/train.py --agent dqn --ticker AAPL
python one/train.py --agent ppo --ticker AAPL
```

## Backtest

```bash
python one/backtest.py --agent dqn --ticker AAPL --model-path one/models/dqn/best_model.keras
python one/backtest.py --agent ppo --ticker AAPL --model-path one/models/ppo/best_model
python one/backtest.py --agent compare --ticker AAPL
```

## Outputs

- Training log file: `one/logs/train.log`
- Training metrics CSV: `one/training_metrics.csv`
- Backtest comparison CSV: `one/backtest_report.csv`
- Model checkpoints in `one/models/dqn` and `one/models/ppo`

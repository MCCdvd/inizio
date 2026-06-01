# one: Improved RL Trading Agent Folder

This folder contains a modern implementation of a deep RL trading agent using DQN (with hooks for PPO/A3C), advanced technical indicators, improved reward shaping, and strong code modularity.

## Structure

```
one/
├── agents/
│   ├── dqn_agent.py         # DQN agent implementation
├── env/
│   ├── trading_env.py       # Trading environment with risk and reward shaping
├── features/
│   ├── indicators.py        # Adds technical features to state
├── utils/
│   ├── plot.py              # Plotting for rewards/actions
├── train.py                 # Training loop (DQN example)
├── config.yaml              # Hyperparameters/config
└── README.md                # This file
```

## Setup

1. Install requirements (TensorFlow, pandas, numpy, matplotlib, yfinance)
2. Market data is automatically fetched from Yahoo Finance during training
3. Run training:

```bash
python one/train.py
```

The training script will download OHLCV data for the specified ticker from Yahoo Finance. Edit `one/config.yaml` for hyperparameters, data source configuration, or to expand and include PPO/A3C agents.

---

This codebase gives you a strong starting point to build, tune, and experiment with state-of-the-art RL trading agents.

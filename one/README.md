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

1. Place your OHLCV market data as `data/AAPL.csv` (daily ohlcv, with cols: 'close', 'poc', 'vah', 'val', etc)
2. Install requirements (TensorFlow, pandas, numpy, matplotlib)
3. Run training:

```bash
python one/train.py
```

Edit `one/config.yaml` for hyperparameters, or expand to include PPO/A3C agents.

---

This codebase gives you a strong starting point to build, tune, and experiment with state-of-the-art RL trading agents.

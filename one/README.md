# one: Improved RL Trading Agent Folder

This folder contains a modern implementation of a deep RL trading agent using DQN (with hooks for PPO/A3C), advanced technical indicators, improved reward shaping, and strong code modularity.

## Structure

```
one/
├── agents/
│   ├── dqn_agent.py         # DQN agent implementation with GPU support
├── env/
│   ├── trading_env.py       # Trading environment with risk and reward shaping
├── features/
│   ├── indicators.py        # Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
├── utils/
│   ├── plot.py              # Plotting utilities for rewards/actions
├── train.py                 # Training loop with GPU/CPU auto-detection
├── config.yaml              # Hyperparameters and configuration
└── README.md                # This file
```

## Setup

1. **Install requirements:**
```bash
pip install tensorflow pandas numpy matplotlib yfinance pyyaml
```

2. **Run training:**
```bash
python one/train.py
```

Market data is automatically fetched from Yahoo Finance (AAPL stock by default).

## Features

- **Automatic GPU Detection**: Uses GPU if available, falls back to CPU
- **Technical Indicators**: SMA, EMA, RSI, MACD, Bollinger Bands
- **DQN Agent**: Deep Q-Network with experience replay and epsilon-greedy exploration
- **Trading Environment**: Realistic buy/hold/sell actions with portfolio tracking
- **Configurable Training**: Adjust episodes, batch size, and learning parameters in `config.yaml`

## Performance Tips

### For CPU-only machines:
- Reduce `episodes` in `config.yaml` (default: 50, try 20-30)
- Reduce `batch_size` (default: 32, try 16)
- Agent will still learn effectively with smaller settings

### For GPU machines:
- Increase `episodes` for longer training
- Increase `batch_size` for better GPU utilization
- Training will be 5-10x faster than CPU

## Configuration

Edit `one/config.yaml`:
```yaml
episodes: 50          # Number of training episodes
batch_size: 32        # Batch size for experience replay
```

## Training Output

The script will display:
- GPU/CPU device being used
- Data date range
- Episode rewards and epsilon decay
- Best model saves when performance improves

## Next Steps

- Tune hyperparameters in `config.yaml`
- Modify `one/env/trading_env.py` to add more complex reward shaping
- Implement PPO or A3C agents (hooks available in agents folder)
- Test on different stock tickers

---

This codebase gives you a strong starting point to build, tune, and experiment with state-of-the-art RL trading agents.

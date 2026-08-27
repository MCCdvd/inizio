# Trading Agent with AI - Volume Profile & Reinforcement Learning

Advanced reinforcement learning trading agent using volume profile analysis to define support/resistance targets.

## 🎯 Features

✅ **Volume Profile Analysis**
- Point of Control (POC) - highest volume price
- Value Area High (VAH) - 70% volume upper resistance
- Value Area Low (VAL) - 70% volume lower support
- Smart entry/exit targeting at key levels

✅ **Advanced RL Algorithms**
- **DQN** (Deep Q-Network) - Value-based learning with experience replay
- **PPO** (Proximal Policy Optimization) - Policy gradient with clipping
- **A3C** (Asynchronous Advantage Actor-Critic) - Parallel actor-critic learning
- **Adaptive Selector** - Meta-strategy that routes between RL and deterministic policies by market regime

✅ **Interactive Visualization**
- Volume profile distribution with price action
- Trading signals (buy/sell markers)
- Training performance metrics
- Multi-stock algorithm comparison
- Episode rewards and returns tracking

✅ **Multi-Stock Support**
- Trade any publicly available stock via yfinance
- Portfolio analysis across multiple symbols
- Comparative algorithm performance

✅ **Risk Management**
- Position sizing based on available capital
- Incentive-based reward system
- Stop tracking and profit calculations

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/MCCdvd/inizio.git
cd inizio/trading-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 📋 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
yfinance>=0.1.70
tensorflow>=2.8.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
```

## 🚀 Usage

### Train DQN Agent (Value-based)

```bash
python src/train.py --algorithm dqn --stock AAPL --episodes 50
```

### Train PPO Agent (Policy Gradient)

```bash
python src/train.py --algorithm ppo --stock AAPL --episodes 50
```

### Train A3C Agent (Actor-Critic)

```bash
python src/train.py --algorithm a3c --stock AAPL --episodes 50
```

### Compare All Algorithms

```bash
python src/train.py --algorithm all --stock AAPL --episodes 50 --compare
```

### Train Adaptive Selector

```bash
python src/train.py --algorithm adaptive --stock AAPL --episodes 20
```

### Train on Multiple Stocks

```bash
python src/train.py --algorithm dqn --stock TSLA --episodes 50
python src/train.py --algorithm dqn --stock MSFT --episodes 50
python src/train.py --algorithm dqn --stock GOOGL --episodes 50
```

### Run Backtesting

```bash
python src/backtest.py --stock AAPL --algorithm dqn --start-date 2023-01-01 --end-date 2024-01-01
```

### Run Adaptive Backtesting

```bash
python src/backtest.py --symbol AAPL --algorithm adaptive --start-date 2023-01-01 --end-date 2024-01-01
```

## 🏗️ Architecture

### State Space

```
State = [balance_ratio, shares_held, poc_distance, vah_distance, val_distance, price_norm]
  - balance_ratio: Current balance / Initial balance
  - shares_held: Normalized number of shares
  - poc_distance: (Current Price - POC) / POC
  - vah_distance: (Current Price - VAH) / VAH
  - val_distance: (Current Price - VAL) / VAL
  - price_norm: Current Price / Max Recent Price
```

### Action Space

```
Actions = [Hold, Buy, Sell]
  - Hold (0): No action
  - Buy (1): Purchase shares (incentivized near VAL)
  - Sell (2): Sell all shares (incentivized near VAH)
```

### Reward Function

```python
Reward = Base Reward
       + Portfolio Growth Reward
       + Buy Signal Bonus (if near VAL)
       + Sell Signal Bonus (if near VAH)
       + Profit Reward
```

### Volume Profile Levels

| Level | Definition | Usage |
|-------|-----------|-------|
| **POC** | Price with highest traded volume | Strongest support/resistance |
| **VAH** | Upper bound of 70% volume area | Resistance zone / sell target |
| **VAL** | Lower bound of 70% volume area | Support zone / buy target |

## 🤖 Algorithm Comparison

| Metric | DQN | PPO | A3C |
|--------|-----|-----|-----|
| **Type** | Value-based | Policy-based | Actor-Critic |
| **Convergence** | Moderate | Fast | Fast |
| **Stability** | Good | Very Good | Good |
| **Sample Efficiency** | Moderate | Good | Very Good |
| **Parallel Training** | No | No | Yes |
| **Memory Usage** | High (replay buffer) | Low | Moderate |
| **Best For** | Discrete actions | Continuous learning | Real-time adaptation |

The adaptive selector uses market-regime features such as volatility, trend strength, momentum, relative volume, RSI, Bollinger width, and volume-profile distances to choose among DQN, PPO, A3C, deterministic volume-profile execution, and a defensive cash policy.

## 📊 Training Results Example

**AAPL, 50 Episodes**

| Algorithm | Initial | Final | Return | Trades | Win Rate |
|-----------|---------|-------|--------|--------|----------|
| DQN | $10,000 | $12,350 | +23.5% | 15 | 73% |
| PPO | $10,000 | $13,100 | +31.0% | 12 | 83% |
| A3C | $10,000 | $12,800 | +28.0% | 18 | 78% |

## 📁 Project Structure

```
trading-agent/
├── src/
│   ├── __init__.py
│   ├── trading_agent.py          # Environment & Volume Profile
│   ├── agents.py                 # DQN, PPO, A3C implementations
│   ├── visualization.py          # Plotting & visualization
│   ├── train.py                  # Training script
│   ├── backtest.py               # Backtesting framework
│   ├── portfolio.py              # Multi-stock portfolio
│   └── utils.py                  # Helper functions
├── notebooks/
│   ├── exploration.ipynb         # Jupyter notebook for exploration
│   └── analysis.ipynb            # Performance analysis
├── data/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── logs/
│   └── .gitkeep
├── README.md
├── requirements.txt
├── setup.sh
├── LICENSE
└── .gitignore
```

## 🔧 Advanced Features

### 1. Backtesting Framework

Test strategies on historical data with transaction costs:

```python
from src.backtest import BacktestEngine

engine = BacktestEngine(stock_symbol='AAPL')
results = engine.run_backtest(
    start_date='2023-01-01',
    end_date='2024-01-01',
    algorithm='dqn',
    initial_capital=10000,
    transaction_cost=0.001
)
```

### 2. Multi-Stock Portfolio

Trade multiple stocks simultaneously:

```python
from src.portfolio import PortfolioAgent

portfolio = PortfolioAgent(
    stocks=['AAPL', 'MSFT', 'TSLA', 'GOOGL'],
    initial_capital=50000,
    max_position_size=0.25
)
portfolio.train(episodes=100, algorithm='ppo')
```

### 3. Risk Management

- Position sizing
- Stop-loss orders
- Take-profit levels
- Portfolio rebalancing

## 💡 How It Works

1. **Data Loading**: Downloads 1 year of historical OHLCV data
2. **Volume Profile**: Computes POC, VAH, VAL for each trading period
3. **Environment**: Tracks portfolio value, positions, and trades
4. **Agent Training**:
   - DQN: Learns Q-values via experience replay
   - PPO: Learns policy with gradient clipping
   - A3C: Parallel actor-critic training
5. **Visualization**: Charts trades, volume profile, and metrics
6. **Evaluation**: Backtests on historical data

## 📈 Performance Optimization

### Hyperparameters

```python
# DQN
epsilon = 1.0           # Exploration rate
epsilon_decay = 0.995   # Decay per episode
gamma = 0.95            # Discount factor
batch_size = 32         # Experience replay batch

# PPO
learning_rate = 0.0003
epochs = 10             # Training epochs per batch
clip_ratio = 0.2        # Clipping parameter

# A3C
learning_rate = 0.0001
entropy_coeff = 0.01    # Entropy regularization
```

### Tips for Better Results

- Increase `episodes` for better convergence (100+)
- Adjust `lookback_days` for volume profile window (20-60)
- Use `--compare` flag to find best algorithm for your stock
- Monitor training with visualization outputs
- Test on multiple time periods

## ⚠️ Disclaimer

**This project is for educational purposes only.** 

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always conduct your own research
- Consult with a financial advisor before trading
- Start with small amounts on paper trading
- Never invest money you cannot afford to lose

## 🔬 Future Enhancements

- [ ] Multi-timeframe analysis (1m, 5m, 15m, 1h, daily)
- [ ] Advanced technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Real-time trading integration (Alpaca, Interactive Brokers)
- [ ] Options trading support
- [ ] Sentiment analysis integration
- [ ] Deep meta-learning for quick adaptation
- [ ] Ensemble methods combining multiple agents
- [ ] Model persistence and checkpointing
- [ ] Hyperparameter optimization (Bayesian, Optuna)
- [ ] GPU acceleration support
- [ ] Web dashboard for monitoring

## 📚 References

- **DQN**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- **PPO**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **A3C**: [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- **Volume Profile**: [Market Profile & Volume Profile Analysis](https://www.investopedia.com/terms/v/volume-profile.asp)

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

**MCCdvd**

- GitHub: [@MCCdvd](https://github.com/MCCdvd)
- Email: contact@example.com

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

Questions? Issues? Ideas?

- Open an issue on GitHub
- Start a discussion
- Check existing issues for solutions

---

**Made with ❤️ for algo traders and RL enthusiasts**

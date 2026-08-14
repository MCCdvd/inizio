import logging
import sys
import os

from trading_agent.src.logging_config import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

# Enable GPU memory growth to avoid OOM errors
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        logger.info("GPU(s) detected: %d", len(gpus))
        logger.info("GPU device: %s", gpus[0].name)
    else:
        logger.info("No GPU detected. Using CPU.")
except Exception as e:
    logger.warning("GPU setup warning: %s", e)

# Add the one directory to Python path so modules can be found
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import yfinance as yf
import yaml
from features.indicators import add_indicators
from env.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent

# Load config
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Fetch data from Yahoo Finance
ticker = "AAPL"  # Change this to any stock ticker
logger.info("Fetching data for %s from Yahoo Finance...", ticker)
try:
    df = yf.download(ticker, start="2020-01-01", end="2023-12-31", progress=False)
except Exception as e:
    logger.error("Failed to download data: %s", e)
    raise

# Reset index to make Date a column
if 'Date' in df.columns:
    df = df.reset_index()

# Flatten multi-level columns if needed and rename properly
if isinstance(df.columns[0], tuple):
    # Extract just the first level of the tuple
    df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower() for col in df.columns]
else:
    df.columns = [str(col).lower() for col in df.columns]

# Clean up column names - remove ticker suffix
try:
    df.columns = [col.replace(f'_{ticker.lower()}', '') for col in df.columns]
except Exception:
    pass

logger.info("Data shape: %s", df.shape)
logger.debug("Data columns: %s", df.columns.tolist())

# Rename 'index' to 'date' if it exists
if 'index' in df.columns:
    df = df.rename(columns={'index': 'date'})

logger.info("Data date range: %s to %s", df['date'].min(), df['date'].max())

# Select only OHLCV columns
df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

# Add technical indicators
df = add_indicators(df.dropna())

# Create trading environment
env = TradingEnv(df)

# Initialize DQN agent
agent = DQNAgent(state_size=env._get_state().shape[0], action_size=3)

# Training loop
episodes = config['episodes']
batch_size = config['batch_size']
rewards_history = []

logger.info("Starting training for %s episodes...", episodes)
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    actions_this_ep = []
    
    while True:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        actions_this_ep.append(action)
        
        if done:
            logger.info("Episode %d/%d, Total Reward: %.2f, Epsilon: %.3f", episode+1, episodes, total_reward, agent.epsilon)
            break
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    agent.save_best(total_reward)
    rewards_history.append(total_reward)

# Optionally plot rewards (if running interactively)
try:
    from utils.plot import plot_rewards
    plot_rewards(rewards_history)
    logger.info("Rewards plot displayed!")
except Exception as e:
    logger.debug("Plotting not available: %s", e)

logger.info("Training completed!")

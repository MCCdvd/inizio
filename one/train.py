import pandas as pd
import yfinance as yf
import yaml
from features.indicators import add_indicators
from env.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent

# Load config
with open('one/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Fetch data from Yahoo Finance
ticker = "AAPL"  # Change this to any stock ticker
print(f"Fetching data for {ticker} from Yahoo Finance...")
df = yf.download(ticker, start="2020-01-01", end="2023-12-31", progress=False)

# Rename columns to match expected format (OHLCV)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.columns = ['open', 'high', 'low', 'close', 'volume']
df = df.reset_index()
df = df.rename(columns={'Date': 'date'})

print(f"Data shape: {df.shape}")
print(f"Data date range: {df['date'].min()} to {df['date'].max()}")

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

print(f"\nStarting training for {episodes} episodes...")
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
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            break
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    agent.save_best(total_reward)
    rewards_history.append(total_reward)

# Optionally plot rewards (if running interactively)
try:
    from utils.plot import plot_rewards
    plot_rewards(rewards_history)
    print("Rewards plot displayed!")
except Exception as e:
    print(f"Plotting not available: {str(e)}")

print("Training completed!")

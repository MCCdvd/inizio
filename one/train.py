import pandas as pd
from features.indicators import add_indicators
from env.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent

# Load your CSV or data
df = pd.read_csv('data/AAPL.csv')  # Place your OHLCV file here
df = add_indicators(df.dropna())
env = TradingEnv(df)

agent = DQNAgent(state_size=env._get_state().shape[0], action_size=3)
episodes = 50
batch_size = 32
rewards_history = []

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
            print(f"Episode {episode+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    agent.save_best(total_reward)
    rewards_history.append(total_reward)

# Optionally plot rewards (if running interactively)
try:
    from utils.plot import plot_rewards
    plot_rewards(rewards_history)
except Exception as e:
    print(f"Plotting not available: {str(e)}")

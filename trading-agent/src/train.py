"""
Training script for trading agent with multiple algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from trading_agent import TradingEnvironmentWithVolumeProfile
from agents import DQNAgent, PPOAgent, A3CAgent
from visualization import VolumeProfileVisualizer
import argparse


def train_dqn_agent(stock_symbol: str = "AAPL", episodes: int = 50):
    """Train DQN agent"""
    print(f"\n{'='*60}")
    print(f"Training DQN Agent on {stock_symbol}")
    print(f"{'='*60}\n")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30)
    agent = DQNAgent(state_size=6, action_size=3)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    env.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    batch_size = 32
    episode_rewards = []
    episode_portfolios = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.train(batch_size)
        agent.decay_epsilon()
        
        portfolio_value = env.balance + (env.shares_held * env.prices[env.current_step - 1])
        final_return = ((portfolio_value - env.initial_balance) / env.initial_balance) * 100
        
        episode_rewards.append(total_reward)
        episode_portfolios.append(portfolio_value)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode+1}/{episodes} | Portfolio: ${portfolio_value:,.2f} | Return: {final_return:+.2f}% | Epsilon: {agent.epsilon:.3f}")
    
    print(f"\nTraining completed!")
    print(f"Final Portfolio: ${portfolio_value:,.2f}")
    print(f"Final Return: {final_return:+.2f}%")
    print(f"Total Trades: {len(env.trades)}")
    
    VolumeProfileVisualizer.plot_volume_profile(env.prices, env.volumes, env.poc, env.vah, env.val, 
                                               trades=env.trades, title=f"DQN Agent - {stock_symbol}")
    VolumeProfileVisualizer.plot_training_results(episode_rewards, episode_portfolios, title=f"DQN Training - {stock_symbol}")
    plt.show()
    
    return {
        'agent': agent,
        'env': env,
        'final_portfolio': portfolio_value,
        'trades': env.trades,
        'return_pct': final_return
    }


def train_ppo_agent(stock_symbol: str = "AAPL", episodes: int = 50):
    """Train PPO agent"""
    print(f"\n{'='*60}")
    print(f"Training PPO Agent on {stock_symbol}")
    print(f"{'='*60}\n")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30)
    agent = PPOAgent(state_size=6, action_size=3)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    env.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    episode_rewards = []
    episode_portfolios = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            if agent.critic:
                value = agent.critic.predict(state.reshape(1, -1), verbose=0)[0][0]
            else:
                value = 0
            
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, value)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.train()
        
        portfolio_value = env.balance + (env.shares_held * env.prices[env.current_step - 1])
        final_return = ((portfolio_value - env.initial_balance) / env.initial_balance) * 100
        
        episode_rewards.append(total_reward)
        episode_portfolios.append(portfolio_value)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode+1}/{episodes} | Portfolio: ${portfolio_value:,.2f} | Return: {final_return:+.2f}%")
    
    print(f"\nTraining completed!")
    print(f"Final Portfolio: ${portfolio_value:,.2f}")
    print(f"Final Return: {final_return:+.2f}%")
    print(f"Total Trades: {len(env.trades)}")
    
    VolumeProfileVisualizer.plot_volume_profile(env.prices, env.volumes, env.poc, env.vah, env.val,
                                               trades=env.trades, title=f"PPO Agent - {stock_symbol}")
    VolumeProfileVisualizer.plot_training_results(episode_rewards, episode_portfolios, title=f"PPO Training - {stock_symbol}")
    plt.show()
    
    return {
        'agent': agent,
        'env': env,
        'final_portfolio': portfolio_value,
        'trades': env.trades,
        'return_pct': final_return
    }


def train_a3c_agent(stock_symbol: str = "AAPL", episodes: int = 50):
    """Train A3C agent"""
    print(f"\n{'='*60}")
    print(f"Training A3C Agent on {stock_symbol}")
    print(f"{'='*60}\n")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30)
    agent = A3CAgent(state_size=6, action_size=3)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    env.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    episode_rewards = []
    episode_portfolios = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.train()
        
        portfolio_value = env.balance + (env.shares_held * env.prices[env.current_step - 1])
        final_return = ((portfolio_value - env.initial_balance) / env.initial_balance) * 100
        
        episode_rewards.append(total_reward)
        episode_portfolios.append(portfolio_value)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode: {episode+1}/{episodes} | Portfolio: ${portfolio_value:,.2f} | Return: {final_return:+.2f}%")
    
    print(f"\nTraining completed!")
    print(f"Final Portfolio: ${portfolio_value:,.2f}")
    print(f"Final Return: {final_return:+.2f}%")
    print(f"Total Trades: {len(env.trades)}")
    
    VolumeProfileVisualizer.plot_volume_profile(env.prices, env.volumes, env.poc, env.vah, env.val,
                                               trades=env.trades, title=f"A3C Agent - {stock_symbol}")
    VolumeProfileVisualizer.plot_training_results(episode_rewards, episode_portfolios, title=f"A3C Training - {stock_symbol}")
    plt.show()
    
    return {
        'agent': agent,
        'env': env,
        'final_portfolio': portfolio_value,
        'trades': env.trades,
        'return_pct': final_return
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trading agent with RL')
    parser.add_argument('--algorithm', choices=['dqn', 'ppo', 'a3c', 'all'], default='dqn', help='RL algorithm to use')
    parser.add_argument('--stock', default='AAPL', help='Stock symbol')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--compare', action='store_true', help='Compare multiple stocks')
    
    args = parser.parse_args()
    
    if args.algorithm == 'dqn':
        train_dqn_agent(stock_symbol=args.stock, episodes=args.episodes)
    elif args.algorithm == 'ppo':
        train_ppo_agent(stock_symbol=args.stock, episodes=args.episodes)
    elif args.algorithm == 'a3c':
        train_a3c_agent(stock_symbol=args.stock, episodes=args.episodes)
    elif args.algorithm == 'all':
        dqn_result = train_dqn_agent(stock_symbol=args.stock, episodes=args.episodes)
        ppo_result = train_ppo_agent(stock_symbol=args.stock, episodes=args.episodes)
        a3c_result = train_a3c_agent(stock_symbol=args.stock, episodes=args.episodes)
        
        results = {
            'DQN': dqn_result,
            'PPO': ppo_result,
            'A3C': a3c_result
        }
        
        VolumeProfileVisualizer.plot_multi_stock_comparison(results)
        plt.show()

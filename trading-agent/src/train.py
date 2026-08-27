"""
Training script for trading agent with multiple algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from trading_agent import TradingEnvironmentWithVolumeProfile
from agents import DQNAgent, PPOAgent, A3CAgent
from strategy_selector import AdaptiveStrategySelector
from visualization import VolumeProfileVisualizer
import argparse
import logging

logger = logging.getLogger(__name__)


def train_dqn_agent(stock_symbol: str = "AAPL", episodes: int = 50, seed: int = None):
    """Train DQN agent"""
    logger.info(f"Training DQN Agent on {stock_symbol}")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30, seed=seed)
    agent = DQNAgent(state_size=6, action_size=3, seed=seed)
    
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
        
        portfolio_value = float(env.balance + (env.shares_held * env.prices[env.current_step - 1]))
        final_return = float(((portfolio_value - env.initial_balance) / env.initial_balance) * 100)
        
        episode_rewards.append(total_reward)
        episode_portfolios.append(portfolio_value)
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode: {episode+1}/{episodes} | Portfolio: ${portfolio_value:,.2f} | Return: {final_return:+.2f}% | Epsilon: {agent.epsilon:.3f}")
    
    logger.info("Training completed!")
    logger.info(f"Final Portfolio: ${portfolio_value:,.2f}")
    logger.info(f"Final Return: {final_return:+.2f}%")
    logger.info(f"Total Trades: {len(env.trades)}")
    
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


def train_ppo_agent(stock_symbol: str = "AAPL", episodes: int = 50, seed: int = None):
    """Train PPO agent"""
    logger.info(f"Training PPO Agent on {stock_symbol}")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30, seed=seed)
    agent = PPOAgent(state_size=6, action_size=3, seed=seed)
    
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
            if hasattr(agent, 'critic') and agent.critic is not None:
                try:
                    value = agent.critic.predict(state.reshape(1, -1), verbose=0)[0][0]
                except Exception:
                    value = 0
            else:
                value = 0
            
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, value)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.train()
        
        portfolio_value = float(env.balance + (env.shares_held * env.prices[env.current_step - 1]))
        final_return = float(((portfolio_value - env.initial_balance) / env.initial_balance) * 100)
        
        episode_rewards.append(total_reward)
        episode_portfolios.append(portfolio_value)
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode: {episode+1}/{episodes} | Portfolio: ${portfolio_value:,.2f} | Return: {final_return:+.2f}%")
    
    logger.info("Training completed!")
    logger.info(f"Final Portfolio: ${portfolio_value:,.2f}")
    logger.info(f"Final Return: {final_return:+.2f}%")
    logger.info(f"Total Trades: {len(env.trades)}")
    
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


def train_a3c_agent(stock_symbol: str = "AAPL", episodes: int = 50, seed: int = None):
    """Train A3C agent"""
    logger.info(f"Training A3C Agent on {stock_symbol}")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30, seed=seed)
    agent = A3CAgent(state_size=6, action_size=3, seed=seed)
    
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
        
        portfolio_value = float(env.balance + (env.shares_held * env.prices[env.current_step - 1]))
        final_return = float(((portfolio_value - env.initial_balance) / env.initial_balance) * 100)
        
        episode_rewards.append(total_reward)
        episode_portfolios.append(portfolio_value)
        
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode: {episode+1}/{episodes} | Portfolio: ${portfolio_value:,.2f} | Return: {final_return:+.2f}%")
    
    logger.info("Training completed!")
    logger.info(f"Final Portfolio: ${portfolio_value:,.2f}")
    logger.info(f"Final Return: {final_return:+.2f}%")
    logger.info(f"Total Trades: {len(env.trades)}")
    
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


def train_adaptive_agent(stock_symbol: str = "AAPL", episodes: int = 20, seed: int = None):
    """Train adaptive strategy selector with runtime strategy switching."""
    logger.info(f"Training Adaptive Strategy Selector on {stock_symbol}")

    env = TradingEnvironmentWithVolumeProfile(
        stock_symbol,
        initial_balance=10000,
        transaction_cost=0.001,
        lookback_days=30,
        seed=seed,
    )
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    env.load_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    selector = AdaptiveStrategySelector(
        initial_capital=env.initial_balance,
        transaction_cost=env.transaction_cost,
        backend='auto',
        seed=seed,
    )
    selector.fit(env.prices, env.volumes)
    runtime_strategies = selector.build_runtime_strategies()

    episode_rewards = []
    episode_portfolios = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action, strategy_name, confidence, _ = selector.act(env, state, runtime_strategies)
            next_state, reward, done = env.step(action)
            strategy = runtime_strategies[strategy_name]

            if isinstance(strategy, DQNAgent):
                strategy.remember(state, action, reward, next_state, done)
                strategy.train(batch_size=min(32, max(1, len(strategy.memory))))
                if done:
                    strategy.decay_epsilon()
            elif isinstance(strategy, PPOAgent):
                value = 0.0
                if getattr(strategy, 'critic', None) is not None:
                    try:
                        import torch as _torch
                        tensor_state = _torch.tensor(state, dtype=_torch.float32, device=strategy.device).unsqueeze(0)
                        with _torch.no_grad():
                            value = float(strategy.critic(tensor_state).item())
                    except Exception:
                        value = 0.0
                strategy.store_transition(state, action, reward, value)
                if done:
                    strategy.train()
            elif isinstance(strategy, A3CAgent):
                strategy.store_transition(state, action, reward)
                if done:
                    strategy.train()

            total_reward += reward
            state = next_state

        portfolio_value = float(env.portfolio_history[-1] if env.portfolio_history else env.initial_balance)
        final_return = float(((portfolio_value - env.initial_balance) / env.initial_balance) * 100)
        episode_rewards.append(total_reward)
        episode_portfolios.append(portfolio_value)

        if (episode + 1) % 5 == 0:
            logger.info(
                "Episode: %d/%d | Portfolio: $%0.2f | Return: %+0.2f%% | Strategy: %s | Confidence: %.3f",
                episode + 1,
                episodes,
                portfolio_value,
                final_return,
                selector.active_strategy,
                confidence,
            )

    logger.info("Adaptive training completed!")
    logger.info(f"Final Portfolio: ${portfolio_value:,.2f}")
    logger.info(f"Final Return: {final_return:+.2f}%")
    logger.info(f"Total Trades: {len(env.trades)}")

    VolumeProfileVisualizer.plot_volume_profile(
        env.prices, env.volumes, env.poc, env.vah, env.val,
        trades=env.trades, title=f"Adaptive Selector - {stock_symbol}"
    )
    VolumeProfileVisualizer.plot_training_results(
        episode_rewards, episode_portfolios, title=f"Adaptive Training - {stock_symbol}"
    )
    plt.show()

    return {
        'selector': selector,
        'strategies': runtime_strategies,
        'env': env,
        'final_portfolio': portfolio_value,
        'trades': env.trades,
        'return_pct': final_return,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trading agent with RL')
    parser.add_argument('--algorithm', choices=['dqn', 'ppo', 'a3c', 'adaptive', 'all'], default='dqn', help='RL algorithm to use')
    parser.add_argument('--stock', default='AAPL', help='Stock symbol')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--compare', action='store_true', help='Compare multiple stocks')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    if args.algorithm == 'dqn':
        train_dqn_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed)
    elif args.algorithm == 'ppo':
        train_ppo_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed)
    elif args.algorithm == 'a3c':
        train_a3c_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed)
    elif args.algorithm == 'adaptive':
        train_adaptive_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed)
    elif args.algorithm == 'all':
        dqn_result = train_dqn_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed)
        ppo_result = train_ppo_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed)
        a3c_result = train_a3c_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed)
        adaptive_result = train_adaptive_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed)
        
        results = {
            'DQN': dqn_result,
            'PPO': ppo_result,
            'A3C': a3c_result,
            'ADAPTIVE': adaptive_result
        }
        
        VolumeProfileVisualizer.plot_multi_stock_comparison(results)
        plt.show()

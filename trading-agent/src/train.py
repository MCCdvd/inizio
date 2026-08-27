"""
Training script for trading agent with multiple algorithms
"""
import csv
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from trading_agent import TradingEnvironmentWithVolumeProfile
from agents import DQNAgent, PPOAgent, A3CAgent
from strategy_selector import AdaptiveStrategySelector
from visualization import VolumeProfileVisualizer
from utils import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_trade_metrics,
    calculate_activity_metrics,
)
import argparse
import logging

logger = logging.getLogger(__name__)


def _build_summary(algorithm: str, stock_symbol: str, env, episode_rewards: list, episode_portfolios: list) -> dict:
    """Compute training summary metrics from a completed training run."""
    final_portfolio = float(episode_portfolios[-1]) if episode_portfolios else env.initial_balance
    final_return = float(((final_portfolio - env.initial_balance) / (env.initial_balance + 1e-8)) * 100)
    sharpe = calculate_sharpe_ratio(env.returns_history)
    sortino = calculate_sortino_ratio(env.returns_history)
    max_dd = calculate_max_drawdown(env.portfolio_history)
    trade_metrics = calculate_trade_metrics(env.trades)
    activity_metrics = calculate_activity_metrics(env.trades, env.prices, env.initial_balance, portfolio_history=env.portfolio_history)
    total_fees = sum(float(t.get('fee', 0.0)) for t in env.trades)
    return {
        'algorithm': algorithm,
        'symbol': stock_symbol,
        'initial_capital': float(env.initial_balance),
        'final_portfolio': final_portfolio,
        'total_return_pct': final_return,
        'sharpe_ratio': float(sharpe) if math.isfinite(sharpe) else 0.0,
        'sortino_ratio': float(sortino) if math.isfinite(sortino) else 0.0,
        'max_drawdown': float(max_dd),
        'total_trades': len(env.trades),
        'total_fees_paid': total_fees,
        'trade_metrics': trade_metrics,
        'activity_metrics': activity_metrics,
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_portfolios': [float(p) for p in episode_portfolios],
        'trades': env.trades,
        'portfolio_history': [float(v) for v in env.portfolio_history],
    }


def _early_stop_check(
    episode_portfolios: list,
    initial_balance: float,
    window: int = 50,
    patience: int = 50,
    min_episodes: int = 200,
) -> bool:
    """Return True when training should stop early.

    Stops when the rolling Sharpe over the last *window* episodes has been
    declining for *patience* consecutive episodes and at least *min_episodes*
    have been completed.  Uses period returns derived from episode portfolio
    values.
    """
    n = len(episode_portfolios)
    if n < min_episodes + window:
        return False

    def _rolling_sharpe(portfolios):
        if len(portfolios) < 2:
            return 0.0
        returns = [
            (portfolios[i] - portfolios[i - 1]) / (portfolios[i - 1] + 1e-8)
            for i in range(1, len(portfolios))
        ]
        mean_r = float(np.mean(returns))
        std_r = float(np.std(returns)) + 1e-8
        return mean_r / std_r * (252 ** 0.5)

    current_sharpe = _rolling_sharpe(episode_portfolios[-window:])
    prev_sharpe = _rolling_sharpe(episode_portfolios[-window - patience: -patience])
    if current_sharpe < prev_sharpe:
        logger.info(
            'Early stop: rolling Sharpe dropped %.3f → %.3f over last %d episodes',
            prev_sharpe,
            current_sharpe,
            patience,
        )
        return True
    return False


def export_results(summary: dict, output_dir: str) -> None:
    """Write per-trade CSV, episode CSV and JSON summary to *output_dir*."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- trades.csv ---
    trades_path = out / 'trades.csv'
    fieldnames = ['step', 'type', 'price', 'shares', 'profit_pct', 'fee']
    with open(trades_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for trade in summary.get('trades', []):
            row = {k: trade.get(k, '') for k in fieldnames}
            writer.writerow(row)
    logger.info('Saved trades CSV to %s', trades_path)

    # --- episode_summary.csv ---
    episodes_path = out / 'episode_summary.csv'
    rewards = summary.get('episode_rewards', [])
    portfolios = summary.get('episode_portfolios', [])
    with open(episodes_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(['episode', 'total_reward', 'portfolio_value', 'return_pct'])
        initial = summary.get('initial_capital', 10000)
        for i, (reward, portfolio) in enumerate(zip(rewards, portfolios), 1):
            ret = ((portfolio - initial) / (initial + 1e-8)) * 100
            writer.writerow([i, round(float(reward), 6), round(float(portfolio), 2), round(float(ret), 4)])
    logger.info('Saved episode summary CSV to %s', episodes_path)

    # --- summary.json ---
    json_path = out / 'summary.json'
    export = {k: v for k, v in summary.items()
              if k not in ('trades', 'portfolio_history', 'episode_rewards', 'episode_portfolios')}
    with open(json_path, 'w', encoding='utf-8') as fh:
        json.dump(export, fh, indent=2, default=float)
    logger.info('Saved summary JSON to %s', json_path)


def train_dqn_agent(stock_symbol: str = "AAPL", episodes: int = 50, seed: int = None, output_dir: str = None, save_model_path: str = None, flat_fee: float = 4.0):
    """Train DQN agent"""
    logger.info(f"Training DQN Agent on {stock_symbol}")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30, seed=seed, flat_fee=flat_fee)
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

        if _early_stop_check(episode_portfolios, env.initial_balance):
            logger.info('Early stopping at episode %d', episode + 1)
            break
    
    logger.info("Training completed!")
    logger.info(f"Final Portfolio: ${portfolio_value:,.2f}")
    logger.info(f"Final Return: {final_return:+.2f}%")
    logger.info(f"Total Trades: {len(env.trades)}")
    
    VolumeProfileVisualizer.plot_volume_profile(env.prices, env.volumes, env.poc, env.vah, env.val, 
                                               trades=env.trades, title=f"DQN Agent - {stock_symbol}")
    VolumeProfileVisualizer.plot_training_results(episode_rewards, episode_portfolios, title=f"DQN Training - {stock_symbol}")
    plt.show()

    summary = _build_summary('dqn', stock_symbol, env, episode_rewards, episode_portfolios)
    if output_dir:
        export_results(summary, output_dir)
    if save_model_path:
        agent.save_model(save_model_path)

    return {
        'agent': agent,
        'env': env,
        'final_portfolio': portfolio_value,
        'trades': env.trades,
        'return_pct': final_return,
        'summary': summary,
    }


def train_ppo_agent(stock_symbol: str = "AAPL", episodes: int = 50, seed: int = None, output_dir: str = None, save_model_path: str = None, flat_fee: float = 4.0):
    """Train PPO agent"""
    logger.info(f"Training PPO Agent on {stock_symbol}")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30, seed=seed, flat_fee=flat_fee)
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

        if _early_stop_check(episode_portfolios, env.initial_balance):
            logger.info('Early stopping at episode %d', episode + 1)
            break
    
    logger.info("Training completed!")
    logger.info(f"Final Portfolio: ${portfolio_value:,.2f}")
    logger.info(f"Final Return: {final_return:+.2f}%")
    logger.info(f"Total Trades: {len(env.trades)}")
    
    VolumeProfileVisualizer.plot_volume_profile(env.prices, env.volumes, env.poc, env.vah, env.val,
                                               trades=env.trades, title=f"PPO Agent - {stock_symbol}")
    VolumeProfileVisualizer.plot_training_results(episode_rewards, episode_portfolios, title=f"PPO Training - {stock_symbol}")
    plt.show()

    summary = _build_summary('ppo', stock_symbol, env, episode_rewards, episode_portfolios)
    if output_dir:
        export_results(summary, output_dir)
    if save_model_path:
        agent.save_model(save_model_path)

    return {
        'agent': agent,
        'env': env,
        'final_portfolio': portfolio_value,
        'trades': env.trades,
        'return_pct': final_return,
        'summary': summary,
    }


def train_a3c_agent(stock_symbol: str = "AAPL", episodes: int = 50, seed: int = None, output_dir: str = None, save_model_path: str = None, flat_fee: float = 4.0):
    """Train A3C agent"""
    logger.info(f"Training A3C Agent on {stock_symbol}")
    
    env = TradingEnvironmentWithVolumeProfile(stock_symbol, initial_balance=10000, lookback_days=30, seed=seed, flat_fee=flat_fee)
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

        if _early_stop_check(episode_portfolios, env.initial_balance):
            logger.info('Early stopping at episode %d', episode + 1)
            break
    
    logger.info("Training completed!")
    logger.info(f"Final Portfolio: ${portfolio_value:,.2f}")
    logger.info(f"Final Return: {final_return:+.2f}%")
    logger.info(f"Total Trades: {len(env.trades)}")
    
    VolumeProfileVisualizer.plot_volume_profile(env.prices, env.volumes, env.poc, env.vah, env.val,
                                               trades=env.trades, title=f"A3C Agent - {stock_symbol}")
    VolumeProfileVisualizer.plot_training_results(episode_rewards, episode_portfolios, title=f"A3C Training - {stock_symbol}")
    plt.show()

    summary = _build_summary('a3c', stock_symbol, env, episode_rewards, episode_portfolios)
    if output_dir:
        export_results(summary, output_dir)
    if save_model_path:
        agent.save_model(save_model_path)

    return {
        'agent': agent,
        'env': env,
        'final_portfolio': portfolio_value,
        'trades': env.trades,
        'return_pct': final_return,
        'summary': summary,
    }


def train_adaptive_agent(stock_symbol: str = "AAPL", episodes: int = 20, seed: int = None, output_dir: str = None, save_model_path: str = None, flat_fee: float = 4.0):
    """Train adaptive strategy selector with runtime strategy switching."""
    if episodes <= 0:
        raise ValueError("episodes must be > 0")

    logger.info(f"Training Adaptive Strategy Selector on {stock_symbol}")

    env = TradingEnvironmentWithVolumeProfile(
        stock_symbol,
        initial_balance=10000,
        transaction_cost=0.001,
        flat_fee=flat_fee,
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

    summary = _build_summary('adaptive', stock_symbol, env, episode_rewards, episode_portfolios)
    if output_dir:
        export_results(summary, output_dir)

    return {
        'selector': selector,
        'strategies': runtime_strategies,
        'env': env,
        'final_portfolio': portfolio_value,
        'trades': env.trades,
        'return_pct': final_return,
        'summary': summary,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trading agent with RL')
    parser.add_argument('--algorithm', choices=['dqn', 'ppo', 'a3c', 'adaptive', 'all'], default='dqn', help='RL algorithm to use')
    parser.add_argument('--stock', default='AAPL', help='Stock symbol')
    parser.add_argument('--episodes', type=int, default=50, help='Number of training episodes')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--compare', action='store_true', help='Compare multiple stocks')
    parser.add_argument('--output-dir', type=str, default=None, help='Directory to save results (trades.csv, episode_summary.csv, summary.json)')
    parser.add_argument('--save-model', type=str, default=None, help='Path to save trained model weights (e.g. output/model.pt)')
    parser.add_argument('--flat-fee', type=float, default=4.0, help='Flat fee in $ per trade (buy and sell). Default 4.0')

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.algorithm == 'dqn':
        train_dqn_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed, output_dir=args.output_dir, save_model_path=args.save_model, flat_fee=args.flat_fee)
    elif args.algorithm == 'ppo':
        train_ppo_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed, output_dir=args.output_dir, save_model_path=args.save_model, flat_fee=args.flat_fee)
    elif args.algorithm == 'a3c':
        train_a3c_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed, output_dir=args.output_dir, save_model_path=args.save_model, flat_fee=args.flat_fee)
    elif args.algorithm == 'adaptive':
        train_adaptive_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed, output_dir=args.output_dir, flat_fee=args.flat_fee)
    elif args.algorithm == 'all':
        dqn_result = train_dqn_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed, flat_fee=args.flat_fee,
                                     output_dir=str(Path(args.output_dir) / 'dqn') if args.output_dir else None)
        ppo_result = train_ppo_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed, flat_fee=args.flat_fee,
                                     output_dir=str(Path(args.output_dir) / 'ppo') if args.output_dir else None)
        a3c_result = train_a3c_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed, flat_fee=args.flat_fee,
                                     output_dir=str(Path(args.output_dir) / 'a3c') if args.output_dir else None)
        adaptive_result = train_adaptive_agent(stock_symbol=args.stock, episodes=args.episodes, seed=args.seed, flat_fee=args.flat_fee,
                                               output_dir=str(Path(args.output_dir) / 'adaptive') if args.output_dir else None)

        results = {
            'DQN': dqn_result,
            'PPO': ppo_result,
            'A3C': a3c_result,
            'ADAPTIVE': adaptive_result
        }
        
        VolumeProfileVisualizer.plot_multi_stock_comparison(results)
        plt.show()

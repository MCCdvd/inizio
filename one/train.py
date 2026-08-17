import argparse
import os
import sys

import pandas as pd
import yaml

# Add one directory to import path for script execution
sys.path.insert(0, os.path.dirname(__file__))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from env.trading_env import TradingEnv
from features.indicators import add_indicators
from utils.logger import setup_logger
from utils.metrics import save_metrics_csv


def _enable_gpu_memory_growth(logger):
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info('GPU detected: %s', len(gpus))
    except Exception as exc:
        logger.warning('GPU setup warning: %s', exc)


def _load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _download_data(ticker, start_date, end_date, logger):
    import yfinance as yf

    logger.info('Fetching %s from Yahoo Finance (%s -> %s)', ticker, start_date, end_date)
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as exc:
        raise RuntimeError(f'Failed to download data for {ticker}: {exc}') from exc

    if df is None or df.empty:
        raise RuntimeError(f'No market data returned for ticker {ticker}')

    if 'Date' in df.columns:
        df = df.reset_index()

    if isinstance(df.columns[0], tuple):
        df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    df.columns = [col.replace(f'_{ticker.lower()}', '') for col in df.columns]
    if 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})

    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f'Downloaded data missing required columns: {missing}')

    df = df[required_cols].dropna()
    return add_indicators(df).dropna()


def _build_agent(agent_name, state_size, action_size, config):
    if agent_name == 'ppo':
        return PPOAgent(state_size=state_size, action_size=action_size, config=config.get('ppo', {}))
    return DQNAgent(state_size=state_size, action_size=action_size, config=config.get('dqn', {}))


def _train_dqn(agent, env, episodes, batch_size, checkpoint_interval, logger):
    rewards_history = []
    metrics_history = []
    action_history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn(batch_size=batch_size)
            state = next_state
            total_reward += reward
            action_history.append(action)

        episode_metrics = info.get('metrics', {})
        episode_metrics.update({'episode': episode, 'reward': float(total_reward), 'epsilon': float(agent.epsilon)})
        metrics_history.append(episode_metrics)

        agent.save_best(total_reward, episode=episode)
        if checkpoint_interval > 0 and episode % checkpoint_interval == 0:
            agent.save_checkpoint(episode)

        rewards_history.append(total_reward)
        logger.info(
            'Episode %d/%d | reward=%.4f | return=%.2f%% | drawdown=%.2f%% | sharpe=%.3f | epsilon=%.4f',
            episode,
            episodes,
            total_reward,
            episode_metrics.get('total_return_pct', 0.0),
            episode_metrics.get('max_drawdown_pct', 0.0),
            episode_metrics.get('sharpe_ratio', 0.0),
            agent.epsilon,
        )

    return rewards_history, metrics_history, action_history


def _train_ppo(agent, env, episodes, logger):
    rewards_history = []
    metrics_history = []
    action_history = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0.0
        done = False
        info = {}

        while not done:
            action, value, log_prob = agent.act(state)
            next_state, reward, done, info = env.step(int(action) if not isinstance(action, (list, tuple, pd.Series)) else action)
            agent.store_transition(state, action, reward, done, value, log_prob)
            state = next_state
            total_reward += reward
            action_history.append(int(action) if not isinstance(action, (list, tuple, pd.Series)) else 0)

        agent.learn()

        episode_metrics = info.get('metrics', {})
        episode_metrics.update({'episode': episode, 'reward': float(total_reward), 'epsilon': 0.0})
        metrics_history.append(episode_metrics)
        rewards_history.append(total_reward)

        logger.info(
            'Episode %d/%d | reward=%.4f | return=%.2f%% | drawdown=%.2f%% | sharpe=%.3f',
            episode,
            episodes,
            total_reward,
            episode_metrics.get('total_return_pct', 0.0),
            episode_metrics.get('max_drawdown_pct', 0.0),
            episode_metrics.get('sharpe_ratio', 0.0),
        )

    if agent.use_tf:
        model_path = os.path.join(agent.model_dir, 'best_model')
        agent.save(model_path)

    return rewards_history, metrics_history, action_history


def run_training(config, ticker='AAPL', agent_name='dqn'):
    logging_cfg = config.get('logging', {})
    logger = setup_logger(
        name=f'one.train.{agent_name}',
        level=logging_cfg.get('level', 'INFO'),
        log_file=logging_cfg.get('file', 'one/logs/train.log'),
    )

    _enable_gpu_memory_growth(logger)

    training_cfg = config.get('training', {})
    episodes = int(training_cfg.get('episodes', config.get('episodes', 50)))
    batch_size = int(training_cfg.get('batch_size', config.get('batch_size', 32)))
    checkpoint_interval = int(training_cfg.get('checkpoint_interval', 10))

    data_cfg = config.get('data', {})
    start_date = data_cfg.get('start', '2020-01-01')
    end_date = data_cfg.get('end', '2023-12-31')

    env_cfg = config.get('env', {})

    df = _download_data(ticker=ticker, start_date=start_date, end_date=end_date, logger=logger)
    logger.info('Data prepared with %d rows and %d columns', len(df), len(df.columns))

    env = TradingEnv(
        df,
        initial_balance=env_cfg.get('initial_balance', config.get('initial_balance', 10000)),
        window_size=env_cfg.get('window_size', 30),
        drawdown_penalty=env_cfg.get('drawdown_penalty', 0.1),
        trade_bonus=env_cfg.get('trade_bonus', 0.05),
        hold_penalty=env_cfg.get('hold_penalty', 0.001),
        risk_reward_weight=env_cfg.get('risk_reward_weight', 0.05),
        max_hold_steps=env_cfg.get('max_hold_steps', 20),
    )

    state_size = env.state_size
    action_size = 3
    logger.info('Environment state_size=%d action_size=%d', state_size, action_size)

    agent = _build_agent(agent_name=agent_name, state_size=state_size, action_size=action_size, config=config)

    if agent_name == 'ppo':
        rewards_history, metrics_history, actions = _train_ppo(agent, env, episodes, logger)
    else:
        rewards_history, metrics_history, actions = _train_dqn(agent, env, episodes, batch_size, checkpoint_interval, logger)

    output_cfg = config.get('output', {})
    metrics_path = output_cfg.get('metrics_csv', f'one/{agent_name}_metrics.csv')
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    save_metrics_csv(metrics_history, metrics_path)
    logger.info('Saved training metrics to %s', metrics_path)

    if output_cfg.get('plot', True):
        try:
            from utils.plot import plot_action_distribution, plot_equity_curve, plot_rewards, plot_training_metrics

            plot_rewards(rewards_history)
            if metrics_history:
                plot_training_metrics(pd.DataFrame(metrics_history).to_dict(orient='list'))
            plot_action_distribution(actions)
            plot_equity_curve(env.portfolio_history, title=f'{agent_name.upper()} Training Equity Curve')
        except Exception as exc:
            logger.warning('Plotting skipped: %s', exc)

    logger.info('Training complete for %s', agent_name.upper())
    return {'agent': agent, 'metrics': metrics_history, 'rewards': rewards_history}


def main():
    parser = argparse.ArgumentParser(description='Train trading agents in one/')
    parser.add_argument('--agent', choices=['dqn', 'ppo'], default='dqn')
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    args = parser.parse_args()

    try:
        config = _load_config(args.config)
        run_training(config=config, ticker=args.ticker, agent_name=args.agent)
    except Exception as exc:
        logger = setup_logger(name='one.train', level='ERROR')
        logger.exception('Training failed: %s', exc)
        raise


if __name__ == '__main__':
    main()

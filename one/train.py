import argparse
import itertools
import json
import os
import random
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from backtest import run_backtest, run_walk_forward_validation
from env.trading_env import TradingEnv
from features.indicators import add_indicators
from utils.action import to_discrete_action
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


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


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
    return df[required_cols].copy()


def _prepare_data(raw_df, data_cfg, feature_cfg):
    df = raw_df.copy()
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True).dt.tz_localize(None)
    df = df.dropna(subset=['date']).sort_values('date').drop_duplicates(subset=['date'], keep='last')

    timeframe = str(data_cfg.get('timeframe', '1D')).upper()
    if timeframe != '1D':
        df = (
            df.set_index('date')
            .resample(timeframe)
            .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            .dropna()
            .reset_index()
        )

    if data_cfg.get('fill_missing', True):
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill().bfill()
        df['volume'] = df['volume'].fillna(0.0)

    z_threshold = float(data_cfg.get('outlier_zscore_threshold', 0.0) or 0.0)
    if z_threshold > 0:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            s = df[col]
            std = float(s.std())
            if std > 0:
                z = (s - s.mean()) / std
                df[col] = s.where(z.abs() <= z_threshold, np.nan)
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].ffill().bfill()
        df['volume'] = df['volume'].fillna(df['volume'].median())

    indicators = feature_cfg.get('indicators', ['all'])
    lookback = int(feature_cfg.get('volume_profile_lookback', 30))
    return add_indicators(df, indicators=indicators, volume_profile_lookback=lookback).dropna().reset_index(drop=True)


def _split_df(df, split_ratio):
    split_idx = min(max(int(len(df) * float(split_ratio)), 1), max(len(df) - 1, 1))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


def _infer_market_regime(df):
    if len(df) < 5:
        return 'unknown'
    start_price = float(df['close'].iloc[0])
    end_price = float(df['close'].iloc[-1])
    change = (end_price / max(start_price, 1e-8)) - 1.0
    if change > 0.1:
        return 'bull'
    if change < -0.1:
        return 'bear'
    return 'sideways'


def _build_agent(agent_name, state_size, action_size, config):
    if agent_name == 'ppo':
        return PPOAgent(state_size=state_size, action_size=action_size, config=config.get('ppo', {}))
    if agent_name == 'a3c':
        raise ValueError('A3C is not implemented in one/; supported agents are dqn and ppo.')
    return DQNAgent(state_size=state_size, action_size=action_size, config=config.get('dqn', {}))


def _objective_score(metrics, objective_cfg):
    metric = str(objective_cfg.get('primary_metric', 'sharpe_ratio'))
    if metric == 'max_drawdown_pct':
        return -float(metrics.get('max_drawdown_pct', 0.0))
    return float(metrics.get(metric, 0.0))


def _run_episode(agent_name, agent, env, batch_size):
    state = env.reset()
    done = False
    total_reward = 0.0
    actions = []
    info = {}

    while not done:
        if agent_name == 'ppo':
            action, value, log_prob = agent.act(state)
            discrete_action = to_discrete_action(action)
            next_state, reward, done, info = env.step(discrete_action)
            agent.store_transition(state, action, reward, done, value, log_prob)
            actions.append(discrete_action)
        else:
            discrete_action = agent.act(state)
            next_state, reward, done, info = env.step(discrete_action)
            agent.remember(state, discrete_action, reward, next_state, done)
            agent.learn(batch_size=batch_size)
            actions.append(discrete_action)
        state = next_state
        total_reward += reward

    if agent_name == 'ppo':
        agent.learn()
    return float(total_reward), info.get('metrics', {}), actions


def _train_agent(agent_name, env, config, logger):
    training_cfg = config.get('training', {})
    objective_cfg = config.get('objective', {})
    episodes = int(training_cfg.get('episodes', 50))
    batch_size = int(training_cfg.get('batch_size', 32))
    checkpoint_interval = int(training_cfg.get('checkpoint_interval', 10))
    patience = int(training_cfg.get('early_stopping_patience', 0))
    min_delta = float(training_cfg.get('early_stopping_min_delta', 0.0))

    agent = _build_agent(agent_name, env.state_size, 3, config)
    rewards_history = []
    metrics_history = []
    action_history = []
    best_score = -float('inf')
    stagnant = 0

    for episode in range(1, episodes + 1):
        total_reward, episode_metrics, actions = _run_episode(agent_name, agent, env, batch_size=batch_size)
        action_history.extend(actions)
        episode_metrics.update(
            {
                'episode': episode,
                'reward': total_reward,
                'epsilon': float(getattr(agent, 'epsilon', 0.0)),
            }
        )
        score = _objective_score(episode_metrics, objective_cfg=objective_cfg)
        episode_metrics['objective_score'] = float(score)
        metrics_history.append(episode_metrics)
        rewards_history.append(total_reward)

        if agent_name == 'dqn':
            agent.save_best(score, episode=episode)
            if checkpoint_interval > 0 and episode % checkpoint_interval == 0:
                agent.save_checkpoint(episode)
        elif agent_name == 'ppo' and score > best_score:
            model_path = os.path.join(agent.model_dir, 'best_model')
            agent.save(model_path)

        if score > best_score + min_delta:
            best_score = score
            stagnant = 0
        else:
            stagnant += 1

        logger.info(
            'Episode %d/%d | reward=%.4f | return=%.2f%% | drawdown=%.2f%% | sharpe=%.3f | score=%.4f',
            episode,
            episodes,
            total_reward,
            episode_metrics.get('total_return_pct', 0.0),
            episode_metrics.get('max_drawdown_pct', 0.0),
            episode_metrics.get('sharpe_ratio', 0.0),
            score,
        )

        if patience > 0 and stagnant >= patience:
            logger.info('Early stopping triggered at episode %d due to stagnant objective score.', episode)
            break

    return {'agent': agent, 'metrics': metrics_history, 'rewards': rewards_history, 'actions': action_history}


def _update_leaderboard(output_cfg, row):
    leaderboard_path = output_cfg.get('leaderboard_csv', 'one/leaderboard.csv')
    leaderboard_dir = os.path.dirname(leaderboard_path)
    if leaderboard_dir:
        os.makedirs(leaderboard_dir, exist_ok=True)
    new_df = pd.DataFrame([row])
    if os.path.exists(leaderboard_path):
        existing = pd.read_csv(leaderboard_path)
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.sort_values(by='objective_score', ascending=False, inplace=True)
    new_df.to_csv(leaderboard_path, index=False)


def _save_retrain_status(output_cfg, score, threshold):
    path = output_cfg.get('retrain_status_json', 'one/retrain_status.json')
    path_dir = os.path.dirname(path)
    if path_dir:
        os.makedirs(path_dir, exist_ok=True)
    status = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'latest_objective_score': float(score),
        'degradation_threshold': float(threshold),
        'retrain_required': False,
    }
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as handle:
            prev = json.load(handle)
        previous_score = float(prev.get('latest_objective_score', score))
        status['previous_objective_score'] = previous_score
        status['score_delta'] = float(score - previous_score)
        status['retrain_required'] = (score - previous_score) < -float(threshold)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(status, handle, indent=2)
    return status


def _iter_search_configs(base_config, agent_name):
    search_cfg = base_config.get('search', {})
    grid = search_cfg.get('grid', {}).get(agent_name, {})
    if not grid:
        yield base_config
        return
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    max_trials = int(search_cfg.get('max_trials', 0))
    for idx, combo in enumerate(itertools.product(*values), start=1):
        if max_trials > 0 and idx > max_trials:
            break
        cfg = json.loads(json.dumps(base_config))
        for key, value in zip(keys, combo):
            cfg[agent_name][key] = value
        yield cfg


def run_training(config, ticker='AAPL', agent_name='dqn'):
    logging_cfg = config.get('logging', {})
    logger = setup_logger(
        name=f'one.train.{agent_name}',
        level=logging_cfg.get('level', 'INFO'),
        log_file=logging_cfg.get('file', 'one/logs/train.log'),
    )
    _enable_gpu_memory_growth(logger)

    training_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    env_cfg = config.get('env', {})
    output_cfg = config.get('output', {})
    objective_cfg = config.get('objective', {})
    feature_cfg = config.get('features', {})
    validation_cfg = config.get('validation', {})
    seed = int(training_cfg.get('seed', 42))

    _set_seed(seed)
    raw_df = _download_data(ticker=ticker, start_date=data_cfg.get('start', '2020-01-01'), end_date=data_cfg.get('end', '2023-12-31'), logger=logger)
    df = _prepare_data(raw_df, data_cfg=data_cfg, feature_cfg=feature_cfg)
    train_df, test_df = _split_df(df, data_cfg.get('split_ratio', 0.8))
    logger.info('Prepared dataset rows=%d train=%d test=%d regime=%s', len(df), len(train_df), len(test_df), _infer_market_regime(test_df))

    env_kwargs = {
        'initial_balance': env_cfg.get('initial_balance', config.get('initial_balance', 10000)),
        'window_size': env_cfg.get('window_size', 30),
        'drawdown_penalty': env_cfg.get('drawdown_penalty', 0.1),
        'trade_bonus': env_cfg.get('trade_bonus', 0.05),
        'hold_penalty': env_cfg.get('hold_penalty', 0.001),
        'risk_reward_weight': env_cfg.get('risk_reward_weight', 0.05),
        'max_hold_steps': env_cfg.get('max_hold_steps', 20),
        'commission_pct': env_cfg.get('commission_pct', 0.001),
        'slippage_pct': env_cfg.get('slippage_pct', 0.0005),
        'overtrade_penalty': env_cfg.get('overtrade_penalty', 0.001),
        'stop_loss_pct': env_cfg.get('stop_loss_pct', 0.05),
        'take_profit_pct': env_cfg.get('take_profit_pct', 0.1),
    }

    best = None
    for trial, trial_cfg in enumerate(_iter_search_configs(config, agent_name), start=1):
        _set_seed(seed)
        env = TradingEnv(train_df, **env_kwargs)
        train_result = _train_agent(agent_name=agent_name, env=env, config=trial_cfg, logger=logger)
        trial_metrics = train_result['metrics'][-1] if train_result['metrics'] else {}
        score = _objective_score(trial_metrics, objective_cfg=objective_cfg)
        if best is None or score > best['score']:
            best = {'trial': trial, 'score': score, 'result': train_result, 'config': trial_cfg}
        logger.info('Trial %d completed with objective score %.5f', trial, score)

    train_result = best['result']
    metrics_history = train_result['metrics']
    rewards_history = train_result['rewards']
    actions = train_result['actions']

    metrics_path = output_cfg.get('metrics_csv', f'one/{agent_name}_metrics.csv')
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    save_metrics_csv(metrics_history, metrics_path)
    logger.info('Saved training metrics to %s', metrics_path)

    model_path = None
    if agent_name == 'dqn':
        model_path = config.get('backtest', {}).get('models', {}).get('dqn', 'one/models/dqn/best_model.keras')
    elif agent_name == 'ppo':
        model_path = config.get('backtest', {}).get('models', {}).get('ppo', 'one/models/ppo/best_model')

    backtest_metrics, _ = run_backtest(
        agent_name=agent_name,
        ticker=ticker,
        config=config,
        model_path=model_path,
        prepared_df=df,
    )
    wf_df = run_walk_forward_validation(agent_name=agent_name, ticker=ticker, config=config, prepared_df=df, model_path=model_path)
    objective_score = _objective_score(backtest_metrics, objective_cfg=objective_cfg)

    latest_train = metrics_history[-1] if metrics_history else {}
    leaderboard_row = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'ticker': ticker,
        'agent': agent_name,
        'seed': seed,
        'primary_metric': objective_cfg.get('primary_metric', 'sharpe_ratio'),
        'objective_score': objective_score,
        'train_sharpe': latest_train.get('sharpe_ratio', 0.0),
        'train_drawdown_pct': latest_train.get('max_drawdown_pct', 0.0),
        'test_sharpe': backtest_metrics.get('sharpe_ratio', 0.0),
        'test_drawdown_pct': backtest_metrics.get('max_drawdown_pct', 0.0),
        'test_return_pct': backtest_metrics.get('total_return_pct', 0.0),
        'walk_forward_avg_sharpe': float(wf_df['sharpe_ratio'].mean()) if not wf_df.empty else 0.0,
    }
    _update_leaderboard(output_cfg, leaderboard_row)

    retrain = _save_retrain_status(
        output_cfg=output_cfg,
        score=objective_score,
        threshold=float(validation_cfg.get('retrain_degradation_threshold', 0.1)),
    )
    logger.info('Retrain trigger status: %s', retrain)

    if output_cfg.get('plot', True):
        try:
            from utils.plot import plot_action_distribution, plot_equity_curve, plot_rewards, plot_training_metrics

            plot_rewards(rewards_history)
            if metrics_history:
                plot_training_metrics(pd.DataFrame(metrics_history).to_dict(orient='list'))
            plot_action_distribution(actions)
            plot_equity_curve([m.get('final_portfolio_value', env_kwargs['initial_balance']) for m in metrics_history], title=f'{agent_name.upper()} Episode Equity')
        except Exception as exc:
            logger.warning('Plotting skipped: %s', exc)

    logger.info('Training complete for %s (best trial=%s score=%.5f)', agent_name.upper(), best['trial'], best['score'])
    return {'agent': train_result['agent'], 'metrics': metrics_history, 'rewards': rewards_history, 'walk_forward': wf_df}


def main():
    parser = argparse.ArgumentParser(description='Train trading agents in one/')
    parser.add_argument('--agent', choices=['dqn', 'ppo', 'a3c'], default='dqn')
    parser.add_argument(
        '--ticker',
        default='all',
        help="Ticker symbol (e.g. AAPL) or 'all' to iterate over config.data.tickers",
    )
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    args = parser.parse_args()

    try:
        config = _load_config(args.config)
        if args.ticker == 'all':
            tickers = config.get('data', {}).get('tickers', [])
            if not tickers:
                raise ValueError(
                    "No tickers found in config.data.tickers. "
                    "Add a 'tickers' list to config.yaml or pass --ticker <SYMBOL>."
                )
        else:
            tickers = [args.ticker]

        for ticker in tickers:
            run_training(config=config, ticker=ticker, agent_name=args.agent)
    except Exception as exc:
        logger = setup_logger(name='one.train', level='ERROR')
        logger.exception('Training failed: %s', exc)
        raise


if __name__ == '__main__':
    main()

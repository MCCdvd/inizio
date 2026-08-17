import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from env.trading_env import TradingEnv
from features.indicators import add_indicators
from utils.logger import setup_logger
from utils.metrics import calculate_performance_metrics


def _load_config(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)


def _download_data(ticker, start, end):
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f'No data found for {ticker}')

    if 'Date' in df.columns:
        df = df.reset_index()

    if isinstance(df.columns[0], tuple):
        df.columns = [col[0].lower() if isinstance(col, tuple) else str(col).lower() for col in df.columns]
    else:
        df.columns = [str(col).lower() for col in df.columns]

    df = df.rename(columns={'index': 'date'})
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in df.columns:
            raise RuntimeError(f'Missing expected OHLCV column: {col}')

    base = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna()
    return add_indicators(base).dropna()


def _init_agent(agent_name, env, config):
    if agent_name == 'ppo':
        return PPOAgent(state_size=env.state_size, action_size=3, config=config.get('ppo', {}))
    return DQNAgent(state_size=env.state_size, action_size=3, config=config.get('dqn', {}))


def run_backtest(agent_name, ticker, config, model_path=None):
    logger = setup_logger(
        name=f'one.backtest.{agent_name}',
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('file', 'one/logs/backtest.log'),
    )

    split = config.get('data', {}).get('split_ratio', 0.8)
    start = config.get('data', {}).get('start', '2020-01-01')
    end = config.get('data', {}).get('end', '2023-12-31')

    df = _download_data(ticker, start, end)
    split_idx = max(int(len(df) * split), 1)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    env = TradingEnv(
        test_df,
        initial_balance=config.get('env', {}).get('initial_balance', 10000),
        window_size=config.get('env', {}).get('window_size', 30),
        drawdown_penalty=config.get('env', {}).get('drawdown_penalty', 0.1),
        trade_bonus=config.get('env', {}).get('trade_bonus', 0.05),
        hold_penalty=config.get('env', {}).get('hold_penalty', 0.001),
        risk_reward_weight=config.get('env', {}).get('risk_reward_weight', 0.05),
        max_hold_steps=config.get('env', {}).get('max_hold_steps', 20),
    )
    agent = _init_agent(agent_name, env, config)

    if model_path:
        try:
            agent.load(model_path)
            logger.info('Loaded model from %s', model_path)
        except Exception as exc:
            logger.warning('Model load failed (%s); running with current policy', exc)

    state = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        if agent_name == 'ppo':
            action, _, _ = agent.act(state)
            action = int(action)
        else:
            action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward

    metrics = calculate_performance_metrics(env.portfolio_history, env.trade_profits)
    metrics.update(
        {
            'agent': agent_name,
            'ticker': ticker,
            'total_reward': float(total_reward),
            'final_portfolio_value': float(env.portfolio_history[-1]),
        }
    )

    logger.info(
        'Backtest %s | return=%.2f%% sharpe=%.3f drawdown=%.2f%% win_rate=%.2f%% trades=%d',
        agent_name.upper(),
        metrics['total_return_pct'],
        metrics['sharpe_ratio'],
        metrics['max_drawdown_pct'],
        metrics['win_rate'] * 100.0,
        metrics['trade_count'],
    )

    return metrics, env


def compare_agents(agent_names, ticker, config):
    results = []
    for name in agent_names:
        model_path = config.get('backtest', {}).get('models', {}).get(name)
        metrics, _ = run_backtest(agent_name=name, ticker=ticker, config=config, model_path=model_path)
        results.append(metrics)

    df = pd.DataFrame(results)
    report_path = config.get('backtest', {}).get('report_csv', 'one/backtest_report.csv')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    df.to_csv(report_path, index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description='Backtest one/ trading agents')
    parser.add_argument('--agent', choices=['dqn', 'ppo', 'compare'], default='dqn')
    parser.add_argument('--ticker', default='AAPL')
    parser.add_argument('--config', default=os.path.join(os.path.dirname(__file__), 'config.yaml'))
    parser.add_argument('--model-path', default=None)
    args = parser.parse_args()

    config = _load_config(args.config)

    if args.agent == 'compare':
        compare = config.get('backtest', {}).get('compare_agents', ['dqn', 'ppo'])
        result_df = compare_agents(compare, args.ticker, config)
        print(result_df.to_string(index=False))
        return

    metrics, _ = run_backtest(args.agent, args.ticker, config, model_path=args.model_path)
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == '__main__':
    main()

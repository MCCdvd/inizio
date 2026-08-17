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
from utils.action import to_discrete_action
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

    df.columns = [col.replace(f'_{ticker.lower()}', '') for col in df.columns]
    df = df.rename(columns={'index': 'date'})
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise RuntimeError(f'Missing expected OHLCV column: {col}')
    return df[required_cols]


def _prepare_data(raw_df, config):
    data_cfg = config.get('data', {})
    feature_cfg = config.get('features', {})
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

    indicators = feature_cfg.get('indicators', ['all'])
    lookback = int(feature_cfg.get('volume_profile_lookback', 30))
    return add_indicators(df, indicators=indicators, volume_profile_lookback=lookback).dropna().reset_index(drop=True)


def _init_agent(agent_name, env, config):
    if agent_name == 'ppo':
        return PPOAgent(state_size=env.state_size, action_size=3, config=config.get('ppo', {}))
    if agent_name == 'a3c':
        raise ValueError('A3C is not implemented in one/; supported agents are dqn and ppo.')
    return DQNAgent(state_size=env.state_size, action_size=3, config=config.get('dqn', {}))


def _build_env(df, config):
    env_cfg = config.get('env', {})
    return TradingEnv(
        df,
        initial_balance=env_cfg.get('initial_balance', 10000),
        window_size=env_cfg.get('window_size', 30),
        drawdown_penalty=env_cfg.get('drawdown_penalty', 0.1),
        trade_bonus=env_cfg.get('trade_bonus', 0.05),
        hold_penalty=env_cfg.get('hold_penalty', 0.001),
        risk_reward_weight=env_cfg.get('risk_reward_weight', 0.05),
        max_hold_steps=env_cfg.get('max_hold_steps', 20),
        commission_pct=env_cfg.get('commission_pct', 0.001),
        slippage_pct=env_cfg.get('slippage_pct', 0.0005),
        overtrade_penalty=env_cfg.get('overtrade_penalty', 0.001),
        stop_loss_pct=env_cfg.get('stop_loss_pct', 0.05),
        take_profit_pct=env_cfg.get('take_profit_pct', 0.1),
    )


def _buy_and_hold_metrics(df, initial_balance):
    if df.empty:
        return {'bh_total_return_pct': 0.0, 'bh_sharpe_ratio': 0.0, 'bh_max_drawdown_pct': 0.0}
    prices = df['close'].astype(float).values
    shares = initial_balance / max(prices[0], 1e-8)
    portfolio = shares * prices
    metrics = calculate_performance_metrics(portfolio.tolist(), trade_profits=[])
    return {
        'bh_total_return_pct': metrics['total_return_pct'],
        'bh_sharpe_ratio': metrics['sharpe_ratio'],
        'bh_max_drawdown_pct': metrics['max_drawdown_pct'],
    }


def run_backtest(agent_name, ticker, config, model_path=None, prepared_df=None):
    logger = setup_logger(
        name=f'one.backtest.{agent_name}',
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=config.get('logging', {}).get('file', 'one/logs/backtest.log'),
    )

    split = float(config.get('data', {}).get('split_ratio', 0.8))
    start = config.get('data', {}).get('start', '2020-01-01')
    end = config.get('data', {}).get('end', '2023-12-31')
    raw_df = _download_data(ticker, start, end) if prepared_df is None else prepared_df
    df = _prepare_data(raw_df, config) if prepared_df is None else prepared_df

    split_idx = max(int(len(df) * split), 1)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    env = _build_env(test_df, config)
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
            action = to_discrete_action(action)
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
            'market_regime': _infer_market_regime(test_df),
        }
    )
    metrics.update(_buy_and_hold_metrics(test_df, initial_balance=config.get('env', {}).get('initial_balance', 10000)))

    logger.info(
        'Backtest %s | return=%.2f%% sharpe=%.3f drawdown=%.2f%% bh_return=%.2f%%',
        agent_name.upper(),
        metrics['total_return_pct'],
        metrics['sharpe_ratio'],
        metrics['max_drawdown_pct'],
        metrics['bh_total_return_pct'],
    )
    return metrics, env


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


def run_walk_forward_validation(agent_name, ticker, config, prepared_df=None, model_path=None):
    validation_cfg = config.get('validation', {})
    folds = max(int(validation_cfg.get('walk_forward_folds', 3)), 1)
    split = float(config.get('data', {}).get('split_ratio', 0.8))

    start = config.get('data', {}).get('start', '2020-01-01')
    end = config.get('data', {}).get('end', '2023-12-31')
    raw_df = _download_data(ticker, start, end) if prepared_df is None else prepared_df
    df = _prepare_data(raw_df, config) if prepared_df is None else prepared_df
    if len(df) < 30:
        return pd.DataFrame()

    base_split = max(int(len(df) * split), 1)
    oos_df = df.iloc[base_split:].reset_index(drop=True)
    fold_size = max(int(len(oos_df) / folds), 1)
    rows = []

    for fold in range(folds):
        start_i = fold * fold_size
        end_i = len(oos_df) if fold == folds - 1 else min((fold + 1) * fold_size, len(oos_df))
        test_slice = oos_df.iloc[start_i:end_i].reset_index(drop=True)
        if len(test_slice) < 5:
            continue
        env = _build_env(test_slice, config)
        agent = _init_agent(agent_name, env, config)
        if model_path:
            try:
                agent.load(model_path)
            except Exception:
                pass

        state = env.reset()
        done = False
        while not done:
            if agent_name == 'ppo':
                action, _, _ = agent.act(state)
                action = to_discrete_action(action)
            else:
                action = agent.act(state)
            state, _, done, _ = env.step(action)

        m = calculate_performance_metrics(env.portfolio_history, env.trade_profits)
        m.update({'fold': fold + 1, 'market_regime': _infer_market_regime(test_slice)})
        m.update(_buy_and_hold_metrics(test_slice, initial_balance=config.get('env', {}).get('initial_balance', 10000)))
        rows.append(m)

    wf_df = pd.DataFrame(rows)
    output_cfg = config.get('output', {})
    path = output_cfg.get('walk_forward_csv', 'one/walk_forward_report.csv')
    path_dir = os.path.dirname(path)
    if path_dir:
        os.makedirs(path_dir, exist_ok=True)
    wf_df.to_csv(path, index=False)
    return wf_df


def compare_agents(agent_names, ticker, config):
    results = []
    for name in agent_names:
        model_path = config.get('backtest', {}).get('models', {}).get(name)
        metrics, _ = run_backtest(agent_name=name, ticker=ticker, config=config, model_path=model_path)
        results.append(metrics)

    df = pd.DataFrame(results)
    report_path = config.get('backtest', {}).get('report_csv', 'one/backtest_report.csv')
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    df.to_csv(report_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(description='Backtest one/ trading agents')
    parser.add_argument('--agent', choices=['dqn', 'ppo', 'a3c', 'compare'], default='dqn')
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
    wf_df = run_walk_forward_validation(args.agent, args.ticker, config, model_path=args.model_path)
    print(pd.DataFrame([metrics]).to_string(index=False))
    if not wf_df.empty:
        print('\nWalk-forward summary:')
        print(wf_df.to_string(index=False))


if __name__ == '__main__':
    main()

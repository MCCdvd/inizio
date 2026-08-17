import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest import compare_agents, run_walk_forward_validation
from features.indicators import add_indicators
from utils.action import to_discrete_action
from utils.metrics import calculate_performance_metrics


def test_calculate_performance_metrics_basic():
    portfolio = [10000, 10200, 10100, 10500]
    trade_profits = [0.01, -0.02, 0.03]

    metrics = calculate_performance_metrics(portfolio, trade_profits)

    assert metrics['total_return_pct'] > 0
    assert metrics['trade_count'] == 3
    assert 0 <= metrics['win_rate'] <= 1
    assert metrics['max_drawdown_pct'] >= 0


def test_compare_agents_writes_report(monkeypatch, tmp_path):
    import backtest

    def fake_run_backtest(agent_name, ticker, config, model_path=None):
        return ({'agent': agent_name, 'ticker': ticker, 'total_return_pct': 1.0, 'trade_count': 2}, None)

    monkeypatch.setattr(backtest, 'run_backtest', fake_run_backtest)

    config = {'backtest': {'report_csv': str(tmp_path / 'report.csv')}}
    result = compare_agents(['dqn', 'ppo'], 'AAPL', config)

    assert isinstance(result, pd.DataFrame)
    assert (tmp_path / 'report.csv').exists()
    assert set(result['agent']) == {'dqn', 'ppo'}


def test_add_indicators_supports_ablation_selection():
    rows = 60
    df = pd.DataFrame(
        {
            'date': pd.date_range('2022-01-01', periods=rows, freq='D'),
            'open': np.linspace(100, 110, rows),
            'high': np.linspace(101, 111, rows),
            'low': np.linspace(99, 109, rows),
            'close': np.linspace(100, 110, rows),
            'volume': np.linspace(1000, 1300, rows),
        }
    )
    out = add_indicators(df, indicators=['volume_profile'], volume_profile_lookback=20)
    assert 'vp_poc' in out.columns
    assert 'dist_val' in out.columns
    assert 'macd' not in out.columns


def test_walk_forward_validation_outputs_report(tmp_path):
    rows = 90
    prepared_df = pd.DataFrame(
        {
            'date': pd.date_range('2022-01-01', periods=rows, freq='D'),
            'open': np.linspace(100, 120, rows),
            'high': np.linspace(101, 121, rows),
            'low': np.linspace(99, 119, rows),
            'close': np.linspace(100, 120, rows),
            'volume': np.linspace(1000, 2000, rows),
            'sma_10': np.linspace(100, 120, rows),
        }
    )
    config = {
        'validation': {'walk_forward_folds': 3},
        'data': {'split_ratio': 0.7},
        'env': {'initial_balance': 10000, 'window_size': 5},
        'output': {'walk_forward_csv': str(tmp_path / 'wf.csv')},
    }
    out = run_walk_forward_validation('dqn', 'AAPL', config, prepared_df=prepared_df, model_path=None)
    assert isinstance(out, pd.DataFrame)
    assert (tmp_path / 'wf.csv').exists()
    if not out.empty:
        assert 'bh_total_return_pct' in out.columns


def test_continuous_action_mapping_matches_env_convention():
    assert to_discrete_action([-0.9]) == 2  # sell
    assert to_discrete_action([0.0]) == 0   # hold
    assert to_discrete_action([0.9]) == 1   # buy

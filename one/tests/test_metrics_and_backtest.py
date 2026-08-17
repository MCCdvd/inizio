import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest import compare_agents
from features.indicators import add_indicators
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

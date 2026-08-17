import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backtest import compare_agents
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

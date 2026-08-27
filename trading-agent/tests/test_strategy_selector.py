import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from strategy_selector import AdaptiveStrategySelector, MarketRegimeFeatureExtractor
from trading_agent import TradingEnvironmentWithVolumeProfile
from backtest import BacktestEngine


def test_market_regime_feature_extractor_returns_finite_features():
    prices = np.linspace(100.0, 110.0, 40) + np.sin(np.arange(40))
    volumes = np.linspace(1000.0, 1500.0, 40)

    features = MarketRegimeFeatureExtractor.extract(prices, volumes, lookback_window=30)
    values = features.as_array()

    assert values.shape == (10,)
    assert np.isfinite(values).all()
    assert 0.0 <= features.rsi <= 100.0


def test_selector_switching_respects_min_holding_period(monkeypatch):
    selector = AdaptiveStrategySelector(decision_interval=1, min_holding_period=5, confidence_threshold=0.0, backend='random')
    env = TradingEnvironmentWithVolumeProfile('TEST', lookback_days=10)
    env.prices = np.linspace(100.0, 120.0, 30)
    env.volumes = np.linspace(1000.0, 1300.0, 30)
    env.reset()

    decisions = iter([
        ('ppo', {'ppo': 0.9, 'dqn': 0.1}),
        ('ppo', {'ppo': 0.9, 'dqn': 0.1}),
    ])
    selector.active_strategy = 'dqn'
    selector._last_switch_step = env.current_step
    monkeypatch.setattr(selector, '_score_with_model', lambda features: next(decisions))

    env.current_step += 1
    strategy_name, _, _ = selector.select_strategy(env)
    assert strategy_name == 'dqn'

    env.current_step += 5
    strategy_name, _, _ = selector.select_strategy(env)
    assert strategy_name == 'ppo'


def test_backtest_engine_supports_adaptive_mode(monkeypatch, tmp_path):
    def fake_load_data(self, start_date, end_date):
        base = np.linspace(100.0, 120.0, 90)
        noise = np.sin(np.arange(90) / 3.0) * 2.0
        self.prices = base + noise
        self.volumes = np.linspace(1000.0, 1800.0, 90)
        return self.prices, self.volumes

    monkeypatch.setattr(TradingEnvironmentWithVolumeProfile, 'load_data', fake_load_data)

    engine = BacktestEngine('TEST', initial_capital=1000.0, transaction_cost=0.002, seed=7)
    result = engine.run_backtest(
        start_date='2024-01-01',
        end_date='2024-04-01',
        algorithm='adaptive',
        episodes=1,
        backend='random',
        output_json=str(tmp_path / 'adaptive.json'),
        no_plot=True,
    )

    assert result['algorithm'] == 'adaptive'
    assert 'selector' in result
    assert result['selector']['training_samples'] >= 1
    assert set(result['selector']['strategy_usage']).issubset({'dqn', 'ppo', 'a3c', 'volume_profile', 'cash'})
    assert result['transaction_cost'] == 0.002

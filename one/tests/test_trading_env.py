import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.trading_env import TradingEnv
from features.indicators import add_indicators


def _sample_df(rows=80):
    dates = pd.date_range('2022-01-01', periods=rows, freq='D')
    close = np.linspace(100, 120, rows)
    data = pd.DataFrame(
        {
            'date': dates,
            'open': close * 0.99,
            'high': close * 1.01,
            'low': close * 0.98,
            'close': close,
            'volume': np.linspace(1000, 2000, rows),
        }
    )
    return add_indicators(data)


def test_state_contains_indicators_and_portfolio_features():
    df = _sample_df()
    env = TradingEnv(df, window_size=20)

    state = env.reset()

    expected_size = env.window_size * len(env.feature_cols) + env.portfolio_feature_count
    assert state.shape[0] == expected_size
    assert np.isfinite(state).all()


def test_reward_shaping_penalizes_bad_holds():
    rows = 90
    dates = pd.date_range('2022-01-01', periods=rows, freq='D')
    close = np.linspace(120, 90, rows)
    df = pd.DataFrame(
        {
            'date': dates,
            'open': close,
            'high': close,
            'low': close,
            'close': close,
            'volume': np.linspace(1000, 2000, rows),
        }
    )
    df = add_indicators(df)

    env = TradingEnv(df, window_size=20, max_hold_steps=1)
    env.reset()

    _, _, _, _ = env.step(1)  # buy
    _, reward, _, _ = env.step(0)  # hold into declining market

    assert reward < 0

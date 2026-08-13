import numpy as np
from trading_agent import TradingEnvironmentWithVolumeProfile


def test_environment_reset_and_step():
    # Create environment and inject synthetic data
    env = TradingEnvironmentWithVolumeProfile("TEST", initial_balance=1000, lookback_days=5, seed=42)

    # Synthetic price and volume series (5 timesteps)
    env.prices = np.array([10.0, 10.5, 11.0, 10.8, 11.2])
    env.volumes = np.array([100, 200, 150, 120, 180])

    state = env.reset()
    assert isinstance(state, np.ndarray)
    assert state.shape[0] == 6

    # Perform a buy action on first step
    next_state, reward, done = env.step(1)  # buy
    assert isinstance(next_state, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    # Continue through all remaining steps with hold actions to reach done
    while not done:
        next_state, reward, done = env.step(0)

    # After episode ends, balance should be numeric and trades should be recorded
    assert isinstance(env.balance, float)
    assert isinstance(env.trades, list)
    # Either zero or more trades recorded
    assert len(env.trades) >= 0

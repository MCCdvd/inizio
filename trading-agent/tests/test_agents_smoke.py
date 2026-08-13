import numpy as np
import pytest
from trading_agent import TradingEnvironmentWithVolumeProfile
from agents import DQNAgent, PPOAgent, A3CAgent

@pytest.mark.parametrize("AgentClass", [DQNAgent, PPOAgent, A3CAgent])
def test_agent_smoke_runs_one_episode(AgentClass):
    # Synthetic environment (short sequence) and deterministic seed
    seed = 123
    env = TradingEnvironmentWithVolumeProfile("TEST", initial_balance=1000, lookback_days=3, seed=seed)
    env.prices = np.array([10.0, 10.2, 10.1, 10.3])
    env.volumes = np.array([100, 120, 110, 130])

    agent_kwargs = {}
    try:
        agent = AgentClass(seed=seed, **agent_kwargs)
    except TypeError:
        # Some agent constructors may not accept seed param
        agent = AgentClass(**agent_kwargs)

    state = env.reset()
    done = False
    step_count = 0

    # Run until episode ends or max steps to avoid infinite loops
    while not done and step_count < 50:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        # call minimal training hooks if available
        if hasattr(agent, "remember"):
            try:
                agent.remember(state, action, reward, next_state, done)
            except Exception:
                pass
        if hasattr(agent, "train"):
            try:
                agent.train(batch_size=8)
            except Exception:
                # smoke tests must not fail due to optional training internals
                pass
        state = next_state
        step_count += 1

    assert isinstance(env.balance, float)
    assert isinstance(env.trades, list)

"""
Advanced RL Agents: DQN, PPO, A3C with TensorFlow fallback and PyTorch integration
"""
import logging
import numpy as np
import random
from collections import deque
from pathlib import Path
from typing import Tuple, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all agents"""

    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001, seed: Optional[int] = None):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.seed = seed
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int):
        import numpy as _np, random as _random
        _np.random.seed(seed)
        _random.seed(seed)
        try:
            import torch as _torch
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        pass

    @abstractmethod
    def train(self):
        pass

    def save_model(self, path: str) -> None:
        """Persist model weights to *path*. Subclasses with trainable weights must override."""
        logger.warning('%s does not implement save_model(); nothing saved.', self.__class__.__name__)

    def load_model(self, path: str) -> None:
        """Load model weights from *path*. Subclasses with trainable weights must override."""
        logger.warning('%s does not implement load_model(); nothing loaded.', self.__class__.__name__)


# Try to detect PyTorch availability
_HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# Try to detect TensorFlow availability (legacy fallback)
_HAS_TF = False
try:
    import tensorflow as tf
    from tensorflow import keras
    _HAS_TF = True
except Exception:
    _HAS_TF = False


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent with PyTorch implementation if available, otherwise TensorFlow fallback or random policy."""

    def __init__(self, state_size: int = 6, action_size: int = 3, learning_rate: float = 0.001, seed: Optional[int] = None):
        super().__init__(state_size, action_size, learning_rate, seed)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.97
        self.gamma = 0.95
        self.memory = deque(maxlen=2000)
        self.batch_size = 32

        if _HAS_TORCH:
            # PyTorch model
            class QNetwork(nn.Module):
                def __init__(self, input_dim, output_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_dim, 128),
                        nn.ReLU(),
                        nn.BatchNorm1d(128),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(p=0.2),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, output_dim)
                    )

                def forward(self, x):
                    return self.net(x)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = QNetwork(self.state_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.loss_fn = nn.MSELoss()

        elif _HAS_TF:
            # TensorFlow model (existing approach)
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
            self.model = model
        else:
            # No deep learning backend installed
            self.model = None

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if _HAS_TORCH and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.model(s).cpu().numpy()[0]
            return int(np.argmax(q))

        if _HAS_TF and self.model is not None:
            act_values = self.model.predict(state.reshape(1, -1), verbose=0)
            return int(np.argmax(act_values[0]))

        return random.randrange(self.action_size)

    def train(self, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size

        if (not self.memory) or (len(self.memory) < batch_size):
            return

        minibatch = random.sample(self.memory, batch_size)

        if _HAS_TORCH and self.model is not None:
            self.model.train()
            states = torch.tensor(np.vstack([m[0] for m in minibatch]), dtype=torch.float32, device=self.device)
            actions = torch.tensor([m[1] for m in minibatch], dtype=torch.int64, device=self.device)
            rewards = torch.tensor([m[2] for m in minibatch], dtype=torch.float32, device=self.device)
            next_states = torch.tensor(np.vstack([m[3] for m in minibatch]), dtype=torch.float32, device=self.device)
            dones = torch.tensor([m[4] for m in minibatch], dtype=torch.float32, device=self.device)

            q_values = self.model(states)
            q_next = self.model(next_states).detach()
            q_target = q_values.clone().detach()

            max_next = torch.max(q_next, dim=1)[0]
            for i in range(batch_size):
                if dones[i]:
                    q_target[i, actions[i]] = rewards[i]
                else:
                    q_target[i, actions[i]] = rewards[i] + self.gamma * max_next[i]

            loss = self.loss_fn(q_values, q_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        elif _HAS_TF and self.model is not None:
            states = np.array([x[0] for x in minibatch])
            actions = np.array([x[1] for x in minibatch])
            rewards = np.array([x[2] for x in minibatch])
            next_states = np.array([x[3] for x in minibatch])
            dones = np.array([x[4] for x in minibatch])

            targets = self.model.predict(states, verbose=0)
            next_q_values = self.model.predict(next_states, verbose=0)

            for i in range(batch_size):
                if dones[i]:
                    targets[i][actions[i]] = rewards[i]
                else:
                    targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

            self.model.fit(states, targets, epochs=1, verbose=0)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_TORCH and hasattr(self, 'q_network'):
            torch.save({'q_network': self.q_network.state_dict(), 'epsilon': self.epsilon}, str(p))
            logger.info('DQNAgent model saved to %s', p)
        elif not _HAS_TORCH and hasattr(self, 'model'):
            self.model.save(str(p))
            logger.info('DQNAgent TF model saved to %s', p)
        else:
            logger.warning('DQNAgent: no trainable model to save.')

    def load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Model file not found: {p}')
        if _HAS_TORCH and hasattr(self, 'q_network'):
            checkpoint = torch.load(str(p), map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            self.epsilon = float(checkpoint.get('epsilon', self.epsilon_min))
            logger.info('DQNAgent model loaded from %s (epsilon=%.4f)', p, self.epsilon)
        elif not _HAS_TORCH and hasattr(self, 'model'):
            import tensorflow as _tf
            self.model = _tf.keras.models.load_model(str(p))
            logger.info('DQNAgent TF model loaded from %s', p)
        else:
            logger.warning('DQNAgent: no trainable model to load into.')


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent with PyTorch if available, otherwise TensorFlow fallback."""

    def __init__(self, state_size: int = 6, action_size: int = 3, learning_rate: float = 0.0003, seed: Optional[int] = None):
        super().__init__(state_size, action_size, learning_rate, seed)
        self.gamma = 0.99
        self.clip_ratio = 0.2
        self.epochs = 10

        self.episode_states: List = []
        self.episode_actions: List = []
        self.episode_rewards: List = []
        self.episode_values: List = []
        self.episode_log_probs: List = []
        self._last_log_prob: float = float(-np.log(max(self.action_size, 1)))

        if _HAS_TORCH:
            class Actor(nn.Module):
                def __init__(self, s_dim, a_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(s_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, a_dim),
                        nn.Softmax(dim=-1)
                    )

                def forward(self, x):
                    return self.net(x)

            class Critic(nn.Module):
                def __init__(self, s_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(s_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1)
                    )

                def forward(self, x):
                    return self.net(x)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.actor = Actor(self.state_size, self.action_size).to(self.device)
            self.critic = Critic(self.state_size).to(self.device)
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.learning_rate)

        elif _HAS_TF:
            self.actor = None
            self.critic = None
        else:
            self.actor = None
            self.critic = None

    def act(self, state: np.ndarray) -> int:
        if _HAS_TORCH and self.actor is not None:
            self.actor.eval()
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                probs_t = self.actor(s)
                dist = torch.distributions.Categorical(probs_t)
                action_t = dist.sample()
                self._last_log_prob = float(dist.log_prob(action_t).item())
                probs = probs_t.cpu().numpy()[0]
            action = int(action_t.item())
            return action

        if _HAS_TF and self.actor is not None:
            policy = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
            action = int(np.random.choice(self.action_size, p=policy))
            self._last_log_prob = float(np.log(max(policy[action], 1e-8)))
            return action

        self._last_log_prob = float(-np.log(max(self.action_size, 1)))
        return random.randrange(self.action_size)

    def store_transition(self, state, action, reward, value=0):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_values.append(value)
        self.episode_log_probs.append(self._last_log_prob)

    def _compute_returns(self, rewards):
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def train(self):
        if not self.episode_states:
            return

        if _HAS_TORCH and self.actor is not None and self.critic is not None:
            states = torch.tensor(np.array(self.episode_states), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array(self.episode_actions), dtype=torch.int64, device=self.device)
            rewards = np.array(self.episode_rewards)
            returns = torch.tensor(self._compute_returns(rewards), dtype=torch.float32, device=self.device)
            old_log_probs = torch.tensor(np.array(self.episode_log_probs), dtype=torch.float32, device=self.device)

            values = self.critic(states).squeeze(-1)
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for _ in range(self.epochs):
                # Compute policy
                probs = self.actor(states)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions)

                # Actor loss (surrogate)
                ratio = torch.exp(log_probs - old_log_probs)
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Critic loss
                critic_loss = F.mse_loss(self.critic(states).squeeze(-1), returns)

                # Update actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()

                # Update critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

        # Clear buffers
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
        self.episode_log_probs.clear()

    def save_model(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_TORCH and hasattr(self, 'actor'):
            torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, str(p))
            logger.info('PPOAgent model saved to %s', p)
        else:
            logger.warning('PPOAgent: no trainable model to save.')

    def load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Model file not found: {p}')
        if _HAS_TORCH and hasattr(self, 'actor'):
            checkpoint = torch.load(str(p), map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            logger.info('PPOAgent model loaded from %s', p)
        else:
            logger.warning('PPOAgent: no trainable model to load into.')


class A3CAgent(BaseAgent):
    """Asynchronous Advantage Actor-Critic Agent (single-threaded simplified)"""

    def __init__(self, state_size: int = 6, action_size: int = 3, learning_rate: float = 0.0001, seed: Optional[int] = None):
        super().__init__(state_size, action_size, learning_rate, seed)
        self.gamma = 0.99
        self.entropy_coeff = 0.01

        self.episode_states: List = []
        self.episode_actions: List = []
        self.episode_rewards: List = []

        if _HAS_TORCH:
            class Actor(nn.Module):
                def __init__(self, s_dim, a_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(s_dim, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, a_dim), nn.Softmax(dim=-1)
                    )
                def forward(self, x):
                    return self.net(x)

            class Critic(nn.Module):
                def __init__(self, s_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(s_dim, 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, 1)
                    )
                def forward(self, x):
                    return self.net(x)

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.actor = Actor(self.state_size, self.action_size).to(self.device)
            self.critic = Critic(self.state_size).to(self.device)
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
            self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        else:
            self.actor = None
            self.critic = None

    def act(self, state: np.ndarray) -> int:
        if _HAS_TORCH and self.actor is not None:
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                probs = self.actor(s).cpu().numpy()[0]
            action = int(np.random.choice(self.action_size, p=probs))
            return action
        return random.randrange(self.action_size)

    def store_transition(self, state, action, reward):
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)

    def _compute_returns(self, rewards):
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

    def train(self):
        if not self.episode_states or not _HAS_TORCH or self.actor is None or self.critic is None:
            # No-op if no backend
            self.episode_states.clear()
            self.episode_actions.clear()
            self.episode_rewards.clear()
            return

        states = torch.tensor(np.array(self.episode_states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.episode_actions), dtype=torch.int64, device=self.device)
        returns = torch.tensor(self._compute_returns(np.array(self.episode_rewards)), dtype=torch.float32, device=self.device)

        # Compute losses
        probs = self.actor(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        values = self.critic(states).squeeze(-1)
        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        critic_loss = F.mse_loss(values, returns)

        total_loss = actor_loss + critic_loss - self.entropy_coeff * entropy

        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        total_loss.backward()
        # Clip gradients for stability
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optim.step()
        self.critic_optim.step()

        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()

    def save_model(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if _HAS_TORCH and hasattr(self, 'actor'):
            torch.save({'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}, str(p))
            logger.info('A3CAgent model saved to %s', p)
        else:
            logger.warning('A3CAgent: no trainable model to save.')

    def load_model(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f'Model file not found: {p}')
        if _HAS_TORCH and hasattr(self, 'actor'):
            checkpoint = torch.load(str(p), map_location=self.device)
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            logger.info('A3CAgent model loaded from %s', p)
        else:
            logger.warning('A3CAgent: no trainable model to load into.')


# Simple random policy agent used as a fallback or for quick experiments
class RandomPolicyAgent(BaseAgent):
    def __init__(self, state_size: int = 6, action_size: int = 3, seed: Optional[int] = None):
        super().__init__(state_size, action_size, seed=seed)

    def act(self, state: np.ndarray) -> int:
        return random.randrange(self.action_size)

    def train(self, *args, **kwargs):
        # No training for random policy
        return


def _check_backend_available(backend: str) -> bool:
    if backend == 'torch':
        try:
            import importlib
            importlib.import_module('torch')
            return True
        except Exception:
            return False
    if backend == 'tf' or backend == 'tensorflow':
        try:
            import importlib
            importlib.import_module('tensorflow')
            return True
        except Exception:
            return False
    if backend == 'random':
        return True
    if backend == 'auto':
        # auto is always acceptable; selection happens in factory
        return True
    return False


def get_agent(algorithm: str, backend: str = 'auto', **kwargs):
    """Factory to construct agents with explicit backend selection.

    algorithm: 'dqn'|'ppo'|'a3c'
    backend: 'auto'|'torch'|'tf'|'random'
    kwargs: passed to agent constructor
    """
    alg = algorithm.lower()
    be = backend.lower()

    # Validate requested backend availability
    if be not in ('auto', 'torch', 'tf', 'random'):
        raise ValueError(f"Unknown backend: {backend}")

    if be != 'auto' and not _check_backend_available(be):
        raise RuntimeError(f"Requested backend '{backend}' is not available in the environment")

    # Decision order: if backend forced to 'random' -> RandomPolicyAgent
    if be == 'random':
        if alg == 'dqn':
            return RandomPolicyAgent(**kwargs)
        elif alg == 'ppo':
            return RandomPolicyAgent(**kwargs)
        elif alg == 'a3c':
            return RandomPolicyAgent(**kwargs)

    # If backend forced to 'torch' or 'tf', attempt to construct the matching implementation
    if be == 'torch':
        if alg == 'dqn':
            if _check_backend_available('torch'):
                return DQNAgent(**kwargs)
            else:
                raise RuntimeError("PyTorch requested but not available")
        if alg == 'ppo':
            if _check_backend_available('torch'):
                return PPOAgent(**kwargs)
            else:
                raise RuntimeError("PyTorch requested but not available")
        if alg == 'a3c':
            if _check_backend_available('torch'):
                return A3CAgent(**kwargs)
            else:
                raise RuntimeError("PyTorch requested but not available")

    if be == 'tf':
        if alg == 'dqn':
            if _check_backend_available('tf'):
                return DQNAgent(**kwargs)
            else:
                raise RuntimeError("TensorFlow requested but not available")
        if alg == 'ppo':
            # PPO TF implementation not provided; fall back to RandomPolicyAgent if TF only
            if _check_backend_available('tf'):
                return PPOAgent(**kwargs)
            else:
                raise RuntimeError("TensorFlow requested but not available")
        if alg == 'a3c':
            if _check_backend_available('tf'):
                return A3CAgent(**kwargs)
            else:
                raise RuntimeError("TensorFlow requested but not available")

    # Auto-selection: prefer PyTorch if available, then TF, otherwise Random
    if be == 'auto':
        if _check_backend_available('torch') and _HAS_TORCH:
            if alg == 'dqn':
                return DQNAgent(**kwargs)
            if alg == 'ppo':
                return PPOAgent(**kwargs)
            if alg == 'a3c':
                return A3CAgent(**kwargs)
        if _check_backend_available('tf') and _HAS_TF:
            if alg == 'dqn':
                return DQNAgent(**kwargs)
            if alg == 'ppo':
                return PPOAgent(**kwargs)
            if alg == 'a3c':
                return A3CAgent(**kwargs)
        # Fallback
        return RandomPolicyAgent(**kwargs)

    # Should not reach here
    raise RuntimeError("Unable to construct agent for given algorithm/backend")

import os
import random
from collections import deque

import numpy as np

from .base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """Double DQN agent with target network and checkpointing."""

    def __init__(self, state_size, action_size, config=None):
        super().__init__(state_size, action_size, config=config)
        self.gamma = float(self.config.get('gamma', 0.99))
        self.learning_rate = float(self.config.get('lr', 0.001))
        self.batch_size = int(self.config.get('batch_size', 32))

        self.epsilon = float(self.config.get('epsilon_start', 1.0))
        self.epsilon_min = float(self.config.get('epsilon_min', 0.01))
        self.epsilon_decay = float(self.config.get('epsilon_decay', 0.995))

        self.target_update_freq = int(self.config.get('target_update_freq', 100))
        self.memory = deque(maxlen=int(self.config.get('memory_size', 20000)))
        self.train_steps = 0

        self.model_dir = self.config.get('model_dir', 'one/models/dqn')
        self.best_reward = -float('inf')

        try:
            import tensorflow as tf

            self.tf = tf
            self.use_tf = True
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.update_target_model(hard=True)
        except Exception:
            self.use_tf = False
            self.tf = None
            self.q_table = {}

    def _build_model(self):
        model = self.tf.keras.Sequential(
            [
                self.tf.keras.layers.Input(shape=(self.state_size,)),
                self.tf.keras.layers.Dense(256, activation='relu'),
                self.tf.keras.layers.Dense(128, activation='relu'),
                self.tf.keras.layers.Dense(64, activation='relu'),
                self.tf.keras.layers.Dense(self.action_size, activation='linear'),
            ]
        )
        model.compile(
            optimizer=self.tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.tf.keras.losses.Huber(),
        )
        return model

    def update_target_model(self, hard=True, tau=1.0):
        if not self.use_tf:
            return
        if hard:
            self.target_model.set_weights(self.model.get_weights())
            return

        online_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        new_weights = [tau * ow + (1.0 - tau) * tw for ow, tw in zip(online_weights, target_weights)]
        self.target_model.set_weights(new_weights)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, int(action), float(reward), next_state, bool(done)))

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        if self.use_tf:
            q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
            return int(np.argmax(q_values))

        state_key = tuple(np.round(state, 4))
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size, dtype=np.float32)
        return int(np.argmax(self.q_table[state_key]))

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(self, batch_size=None):
        batch_size = int(batch_size or self.batch_size)
        if len(self.memory) < batch_size:
            return None

        batch = random.sample(self.memory, batch_size)

        if self.use_tf:
            states = np.array([exp[0] for exp in batch], dtype=np.float32)
            actions = np.array([exp[1] for exp in batch], dtype=np.int64)
            rewards = np.array([exp[2] for exp in batch], dtype=np.float32)
            next_states = np.array([exp[3] for exp in batch], dtype=np.float32)
            dones = np.array([exp[4] for exp in batch], dtype=np.float32)

            targets = self.model.predict(states, verbose=0)
            next_online = self.model.predict(next_states, verbose=0)
            next_target = self.target_model.predict(next_states, verbose=0)

            best_actions = np.argmax(next_online, axis=1)
            next_values = next_target[np.arange(batch_size), best_actions]
            td_targets = rewards + (1.0 - dones) * self.gamma * next_values
            targets[np.arange(batch_size), actions] = td_targets

            history = self.model.fit(states, targets, epochs=1, verbose=0)
            loss = float(history.history['loss'][0]) if history.history.get('loss') else None

            self.train_steps += 1
            if self.train_steps % self.target_update_freq == 0:
                self.update_target_model(hard=False, tau=0.25)

            self._decay_epsilon()
            return loss

        # fallback tabular update
        for state, action, reward, next_state, done in batch:
            sk = tuple(np.round(state, 4))
            nk = tuple(np.round(next_state, 4))
            self.q_table.setdefault(sk, np.zeros(self.action_size, dtype=np.float32))
            self.q_table.setdefault(nk, np.zeros(self.action_size, dtype=np.float32))
            target = reward if done else reward + self.gamma * np.max(self.q_table[nk])
            self.q_table[sk][action] += self.learning_rate * (target - self.q_table[sk][action])

        self._decay_epsilon()
        return None

    def save_best(self, score, episode=None):
        if score <= self.best_reward:
            return
        self.best_reward = score
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, 'best_model.keras')
        self.save(path)
        if episode is not None:
            with open(os.path.join(self.model_dir, 'best_meta.txt'), 'w', encoding='utf-8') as handle:
                handle.write(f'episode={episode}\nscore={score}\n')

    def save_checkpoint(self, episode):
        os.makedirs(self.model_dir, exist_ok=True)
        path = os.path.join(self.model_dir, f'checkpoint_ep_{episode}.keras')
        self.save(path)

    def save(self, path):
        if self.use_tf:
            self.model.save(path)
            return
        np.save(path + '.npy', self.q_table, allow_pickle=True)

    def load(self, path):
        if self.use_tf:
            self.model = self.tf.keras.models.load_model(path)
            self.target_model = self.tf.keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())
            return

        table_path = path if path.endswith('.npy') else path + '.npy'
        if os.path.exists(table_path):
            self.q_table = np.load(table_path, allow_pickle=True).item()

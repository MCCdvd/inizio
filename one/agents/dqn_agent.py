import tensorflow as tf
from collections import deque
import numpy as np
import os

class DQNAgent:
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self._build_model(lr)
        self.best_reward = -np.inf

    def _build_model(self, lr):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss='mse')
        return model

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        minibatch = np.random.choice(len(self.memory), min(batch_size, len(self.memory)), replace=False)
        for idx in minibatch:
            s, a, r, s_, done = self.memory[idx]
            target = r
            if not done:
                target += self.gamma * np.amax(self.model.predict(np.expand_dims(s_, axis=0), verbose=0)[0])
            target_f = self.model.predict(np.expand_dims(s, axis=0), verbose=0)
            target_f[0][a] = target
            self.model.fit(np.expand_dims(s, axis=0), target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_best(self, avg_reward):
        if avg_reward > self.best_reward:
            print(f"New best avg reward: {avg_reward:.2f}. Saving model.")
            self.model.save("models/dqn_best.h5")
            self.best_reward = avg_reward

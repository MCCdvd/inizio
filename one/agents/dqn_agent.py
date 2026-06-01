import numpy as np
from collections import deque
import random

class DQNAgent:
    """Deep Q-Network Agent for trading."""
    
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=2000)
        self.best_reward = -float('inf')
        
        try:
            import tensorflow as tf
            self.model = self._build_model(tf)
            self.use_tf = True
        except ImportError:
            self.use_tf = False
            self.q_table = {}
    
    def _build_model(self, tf):
        """Build neural network model."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_dim=self.state_size),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy."""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if self.use_tf:
            state_tensor = state.reshape(1, -1)
            q_values = self.model.predict(state_tensor, verbose=0)
            return np.argmax(q_values[0])
        else:
            state_key = tuple(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            return np.argmax(self.q_table[state_key])
    
    def replay(self, batch_size):
        """Train on batch of experiences."""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        if self.use_tf:
            states = np.array([exp[0] for exp in batch])
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch])
            next_states = np.array([exp[3] for exp in batch])
            dones = np.array([exp[4] for exp in batch])
            
            target_q_values = self.model.predict(states, verbose=0)
            next_q_values = self.model.predict(next_states, verbose=0)
            
            for i in range(batch_size):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            self.model.fit(states, target_q_values, epochs=1, verbose=0)
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_best(self, reward):
        """Save model if reward is better."""
        if reward > self.best_reward:
            self.best_reward = reward
            if self.use_tf:
                self.model.save('one/best_model.h5')

"""
Advanced RL Agents: DQN, PPO, A3C
"""
import numpy as np
import random
from collections import deque
from typing import Tuple, List
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, state_size: int, action_size: int, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
    
    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        pass
    
    @abstractmethod
    def train(self):
        pass


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size: int = 6, action_size: int = 3, learning_rate: float = 0.001):
        super().__init__(state_size, action_size, learning_rate)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.memory = deque(maxlen=2000)
        
        self.model = self._build_model()
    
    def _build_model(self):
        """Build neural network"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(self.action_size, activation='linear')
            ])
            model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
            return model
        except ImportError:
            print("TensorFlow not installed")
            return None
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        if self.model is None:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(act_values[0])
    
    def train(self, batch_size: int = 32):
        """Train on batch"""
        if self.model is None or len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
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
        """Decay exploration rate"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent"""
    
    def __init__(self, state_size: int = 6, action_size: int = 3, learning_rate: float = 0.0003):
        super().__init__(state_size, action_size, learning_rate)
        
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.epochs = 10
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
    
    def _build_actor(self):
        """Build actor network"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(self.action_size, activation='softmax')
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
            return model
        except ImportError:
            return None
    
    def _build_critic(self):
        """Build critic network"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
            return model
        except ImportError:
            return None
    
    def act(self, state: np.ndarray) -> int:
        """Sample action from policy"""
        if self.actor is None:
            return random.randrange(self.action_size)
        
        policy = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action
    
    def store_transition(self, state, action, reward, value):
        """Store transition for batch training"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_values.append(value)
    
    def train(self):
        """Train PPO on episode batch"""
        if self.actor is None or self.critic is None or len(self.episode_states) == 0:
            return
        
        try:
            import tensorflow as tf
            
            states = np.array(self.episode_states)
            actions = np.array(self.episode_actions)
            rewards = np.array(self.episode_rewards)
            
            returns = self._compute_returns(rewards)
            values = self.critic.predict(states, verbose=0).flatten()
            advantages = returns - values
            
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            for _ in range(self.epochs):
                with tf.GradientTape() as tape:
                    new_policies = self.actor(states, training=True)
                    new_values = self.critic(states, training=True)
                    
                    log_probs = tf.math.log(tf.reduce_sum(new_policies * tf.one_hot(actions, self.action_size), axis=1) + 1e-8)
                    actor_loss = -tf.reduce_mean(log_probs * advantages)
                    
                    # Use tf.reshape instead of .flatten() for TensorFlow tensors
                    critic_loss = tf.reduce_mean((tf.reshape(new_values, [-1]) - returns) ** 2)
                    
                    total_loss = actor_loss + critic_loss
                
                gradients = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
                self.actor.optimizer.apply_gradients(zip(gradients[:len(self.actor.trainable_variables)], self.actor.trainable_variables))
                self.critic.optimizer.apply_gradients(zip(gradients[len(self.actor.trainable_variables):], self.critic.trainable_variables))
        except ImportError:
            pass
        
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
        self.episode_values.clear()
    
    def _compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = np.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns


class A3CAgent(BaseAgent):
    """Asynchronous Advantage Actor-Critic Agent"""
    
    def __init__(self, state_size: int = 6, action_size: int = 3, learning_rate: float = 0.0001):
        super().__init__(state_size, action_size, learning_rate)
        
        self.gamma = 0.99
        self.entropy_coeff = 0.01
        
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def _build_actor(self):
        """Build actor network"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(self.action_size, activation='softmax')
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
            return model
        except ImportError:
            return None
    
    def _build_critic(self):
        """Build critic network"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1)
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
            return model
        except ImportError:
            return None
    
    def act(self, state: np.ndarray) -> int:
        """Sample action from policy"""
        if self.actor is None:
            return random.randrange(self.action_size)
        
        policy = self.actor.predict(state.reshape(1, -1), verbose=0)[0]
        action = np.random.choice(self.action_size, p=policy)
        return action
    
    def store_transition(self, state, action, reward):
        """Store transition"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def train(self):
        """Train A3C on episode batch"""
        if self.actor is None or self.critic is None or len(self.episode_states) == 0:
            return
        
        try:
            import tensorflow as tf
            
            states = np.array(self.episode_states)
            actions = np.array(self.episode_actions)
            
            returns = self._compute_returns(np.array(self.episode_rewards))
            
            with tf.GradientTape() as tape:
                policies = self.actor(states, training=True)
                values = tf.reshape(self.critic(states, training=True), [-1])
                
                advantages = returns - values
                
                log_probs = tf.math.log(tf.reduce_sum(policies * tf.one_hot(actions, self.action_size), axis=1) + 1e-8)
                actor_loss = -tf.reduce_mean(log_probs * advantages)
                
                entropy = -tf.reduce_mean(tf.reduce_sum(policies * tf.math.log(policies + 1e-8), axis=1))
                
                critic_loss = tf.reduce_mean(advantages ** 2)
                
                total_loss = actor_loss + critic_loss - self.entropy_coeff * entropy
            
            variables = self.actor.trainable_variables + self.critic.trainable_variables
            gradients = tape.gradient(total_loss, variables)
            self.actor.optimizer.apply_gradients(zip(gradients[:len(self.actor.trainable_variables)], self.actor.trainable_variables))
            self.critic.optimizer.apply_gradients(zip(gradients[len(self.actor.trainable_variables):], self.critic.trainable_variables))
        except ImportError:
            pass
        
        self.episode_states.clear()
        self.episode_actions.clear()
        self.episode_rewards.clear()
    
    def _compute_returns(self, rewards):
        """Compute discounted returns"""
        returns = np.zeros_like(rewards, dtype=np.float32)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns

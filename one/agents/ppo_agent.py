import os

import numpy as np

from .base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """TensorFlow/Keras PPO with clipped objective and GAE."""

    def __init__(self, state_size, action_size, config=None):
        super().__init__(state_size, action_size, config=config)
        self.gamma = float(self.config.get('gamma', 0.99))
        self.gae_lambda = float(self.config.get('gae_lambda', 0.95))
        self.clip_ratio = float(self.config.get('clip_ratio', 0.2))
        self.entropy_coef = float(self.config.get('entropy_coef', 0.01))
        self.value_coef = float(self.config.get('value_coef', 0.5))
        self.actor_lr = float(self.config.get('actor_lr', 0.0003))
        self.critic_lr = float(self.config.get('critic_lr', 0.001))
        self.epochs = int(self.config.get('epochs', 10))
        self.minibatch_size = int(self.config.get('minibatch_size', 64))
        self.continuous_actions = bool(self.config.get('continuous_actions', False))

        self.model_dir = self.config.get('model_dir', 'one/models/ppo')

        try:
            import tensorflow as tf

            self.tf = tf
            self.use_tf = True
            self.actor = self._build_actor()
            self.critic = self._build_critic()
            self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
            self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        except Exception:
            self.tf = None
            self.use_tf = False
            self.actor = None
            self.critic = None

        self.reset_buffer()

    def _build_actor(self):
        inputs = self.tf.keras.layers.Input(shape=(self.state_size,))
        x = self.tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = self.tf.keras.layers.Dense(128, activation='relu')(x)
        if self.continuous_actions:
            mean_out = self.tf.keras.layers.Dense(self.action_size, activation='tanh', name='mean')(x)
            log_std = self.tf.Variable(initial_value=-0.5 * np.ones(self.action_size, dtype=np.float32), trainable=True)
            model = self.tf.keras.Model(inputs=inputs, outputs=mean_out)
            model.log_std = log_std
            return model

        probs = self.tf.keras.layers.Dense(self.action_size, activation='softmax')(x)
        return self.tf.keras.Model(inputs=inputs, outputs=probs)

    def _build_critic(self):
        return self.tf.keras.Sequential(
            [
                self.tf.keras.layers.Input(shape=(self.state_size,)),
                self.tf.keras.layers.Dense(256, activation='relu'),
                self.tf.keras.layers.Dense(128, activation='relu'),
                self.tf.keras.layers.Dense(1, activation='linear'),
            ]
        )

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.log_probs = []

    def act(self, state):
        if not self.use_tf:
            return int(np.random.randint(self.action_size)) if not self.continuous_actions else np.random.uniform(-1.0, 1.0, size=self.action_size)

        state_t = state.reshape(1, -1).astype(np.float32)
        value = float(self.critic(state_t, training=False).numpy()[0, 0])

        if self.continuous_actions:
            mean = self.actor(state_t, training=False).numpy()[0]
            std = np.exp(self.actor.log_std.numpy())
            action = np.random.normal(mean, std)
            action = np.clip(action, -1.0, 1.0)
            log_prob = -0.5 * np.sum(((action - mean) / (std + 1e-8)) ** 2 + 2 * np.log(std + 1e-8) + np.log(2 * np.pi))
            return action.astype(np.float32), value, float(log_prob)

        probs = self.actor(state_t, training=False).numpy()[0]
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        action = int(np.random.choice(self.action_size, p=probs))
        log_prob = float(np.log(probs[action]))
        return action, value, log_prob

    def store_transition(self, state, action, reward, done, value, log_prob):
        self.states.append(state.astype(np.float32))
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))

    def _compute_gae(self, next_value=0.0):
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [next_value], dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        return advantages, returns

    def learn(self, *args, **kwargs):
        if not self.states:
            return None

        if not self.use_tf:
            self.reset_buffer()
            return None

        states = np.array(self.states, dtype=np.float32)
        actions = np.array(self.actions)
        old_log_probs = np.array(self.log_probs, dtype=np.float32)

        next_value = 0.0
        if not self.dones[-1]:
            next_value = float(self.critic(states[-1:].astype(np.float32), training=False).numpy()[0, 0])

        advantages, returns = self._compute_gae(next_value=next_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        data_size = len(states)
        batch = self.minibatch_size if self.minibatch_size < data_size else data_size

        for _ in range(self.epochs):
            indices = np.random.permutation(data_size)
            for start in range(0, data_size, batch):
                idx = indices[start:start + batch]
                s = states[idx]
                a = actions[idx]
                adv = advantages[idx]
                ret = returns[idx]
                old_lp = old_log_probs[idx]

                self._train_step(s, a, adv, ret, old_lp)

        self.reset_buffer()
        return True

    def _train_step(self, states, actions, advantages, returns, old_log_probs):
        tf = self.tf
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            if self.continuous_actions:
                mean = self.actor(states, training=True)
                std = tf.exp(self.actor.log_std)
                a = tf.cast(actions, tf.float32)
                log_probs = -0.5 * tf.reduce_sum(
                    tf.square((a - mean) / (std + 1e-8)) + 2.0 * tf.math.log(std + 1e-8) + tf.math.log(2.0 * np.pi),
                    axis=1,
                )
                entropy = tf.reduce_mean(0.5 * tf.math.log(2.0 * np.pi * np.e * tf.square(std)))
            else:
                probs = self.actor(states, training=True)
                probs = tf.clip_by_value(probs, 1e-8, 1.0)
                action_mask = tf.one_hot(tf.cast(actions, tf.int32), depth=self.action_size)
                selected = tf.reduce_sum(probs * action_mask, axis=1)
                log_probs = tf.math.log(selected)
                entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs), axis=1))

            ratios = tf.exp(log_probs - old_log_probs)
            clipped = tf.clip_by_value(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
            surrogate = tf.minimum(ratios * advantages, clipped * advantages)
            actor_loss = -tf.reduce_mean(surrogate) - self.entropy_coef * entropy

            values = tf.squeeze(self.critic(states, training=True), axis=1)
            critic_loss = self.value_coef * tf.reduce_mean(tf.square(returns - values))

        actor_vars = self.actor.trainable_variables
        if self.continuous_actions:
            actor_vars = actor_vars + [self.actor.log_std]

        actor_grads = actor_tape.gradient(actor_loss, actor_vars)
        critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

    def save(self, path):
        if not self.use_tf:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.actor.save(path + '_actor.keras')
        self.critic.save(path + '_critic.keras')
        if self.continuous_actions:
            np.save(path + '_log_std.npy', self.actor.log_std.numpy())

    def load(self, path):
        if not self.use_tf:
            return
        self.actor = self.tf.keras.models.load_model(path + '_actor.keras')
        self.critic = self.tf.keras.models.load_model(path + '_critic.keras')
        if self.continuous_actions:
            log_std = np.load(path + '_log_std.npy')
            self.actor.log_std = self.tf.Variable(initial_value=log_std, trainable=True)

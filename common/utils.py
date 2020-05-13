import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from collections import deque


class ReplayBuffer(object):
    def __init__(self, buffer_size: int) -> None:
        self.buffer_size: int = buffer_size
        self.num_experiences: int = 0
        self.buffer: deque = deque()

    def get_batch(self, batch_size: int) -> dict:
        # Randomly sample batch_size examples
        experiences = random.sample(self.buffer, batch_size)
        return {
            "states0": np.asarray([exp[0] for exp in experiences], np.float32),
            "actions": np.asarray([exp[1] for exp in experiences], np.float32),
            "rewards": np.asarray([exp[2] for exp in experiences], np.float32),
            "states1": np.asarray([exp[3] for exp in experiences], np.float32),
            "terminals1": np.asarray([exp[4] for exp in experiences], np.float32)
        }

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, new_state: np.ndarray, done: bool):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    @property
    def size(self) -> int:
        return self.buffer_size

    @property
    def n_entries(self) -> int:
        # If buffer is full, return buffer size
        # Otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

class ActorNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units, n_actions, logprob_epsilon):
        super(ActorNetwork, self).__init__()
        self.logprob_epsilon = logprob_epsilon
        w_bound = 3e-3
        self.hidden = Sequential()
        for _ in range(n_hidden_layers):
            self.hidden.add(Dense(n_hidden_units, activation="relu"))

        self.mean = Dense(n_actions,
                          kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                          bias_initializer=tf.random_uniform_initializer(-w_bound, w_bound))
        self.log_std = Dense(n_actions,
                             kernel_initializer=tf.random_uniform_initializer(-w_bound, w_bound),
                             bias_initializer=tf.random_uniform_initializer(-w_bound, w_bound))
    @tf.function
    def call(self, inp):
        x = self.hidden(inp)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std_clipped = tf.clip_by_value(log_std, -20, 2)
        normal_dist = tfp.distributions.Normal(mean, tf.exp(log_std_clipped))
        action = tf.stop_gradient(normal_dist.sample())
        squashed_actions = tf.tanh(action)
        logprob = normal_dist.log_prob(action) - tf.math.log(1.0 - tf.pow(squashed_actions, 2) + self.logprob_epsilon)
        logprob = tf.reduce_sum(logprob, axis=-1, keepdims=True)
        return squashed_actions, logprob

    def _get_params(self):
        ''
        with self.graph.as_default():
            params = tf.trainable_variables()
        names = [p.name for p in params]
        values = self.sess.run(params)
        params = {k: v for k, v in zip(names, values)}
        return params

    def __getstate__(self):
        params = self._get_params()
        state = self.args_copy, params
        return state

    def __setstate__(self, state):
        args, params = state
        self.__init__(**args)
        self.restore_params(params)

class SoftQNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units):
        super(SoftQNetwork, self).__init__()
        self.softq = Sequential()
        for _ in range(n_hidden_layers):
            self.softq.add(Dense(n_hidden_units, activation="relu"))
        self.softq.add(Dense(1,
                             kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                             bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)))
    @tf.function
    def call(self, states, actions):
        x = tf.concat([states, actions], 1)
        return self.softq(x)

class ValueNetwork(Model):
    def __init__(self, n_hidden_layers, n_hidden_units):
        super(ValueNetwork, self).__init__()
        self.value = Sequential()
        for _ in range(n_hidden_layers):
            self.value.add(Dense(n_hidden_units, activation="relu"))

        self.value.add(Dense(1,
                             kernel_initializer=tf.random_uniform_initializer(-3e-3, 3e-3),
                             bias_initializer=tf.random_uniform_initializer(-3e-3, 3e-3)))

    def call(self, inp):
        return self.value(inp)

def plot_episode_stats(episode_lengths, episode_rewards, smoothing_window=5):
    # Plot the episode length over time
    plt.figure(figsize=(10,5))
    plt.plot(episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.show()

    # Plot the episode reward over time
    plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.show()

    # Plot time steps and episode number
    plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(episode_lengths), np.arange(len(episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.show()

def plot_reward(episode_rewards):
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward over Time")
    plt.show()

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=-0.2, decay_period=100):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.low = -1.0
        self.high = 1.0
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action*0.2 + ou_state, self.low, self.high)

import tensorflow_addons as tfa
from typing import Sequence
from common.utils import *



def soft_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable], tau: float) -> None:
    """Move each source variable by a factor of tau towards the corresponding target variable.
    Arguments:
        source_vars {Sequence[tf.Variable]} -- Source variables to copy from
        target_vars {Sequence[tf.Variable]} -- Variables to copy data to
        tau {float} -- How much to change to source var, between 0 and 1.
    """
    if len(source_vars) != len(target_vars):
        raise ValueError("source_vars and target_vars must have the same length.")
    for source, target in zip(source_vars, target_vars):
        target.assign((1.0 - tau) * target + tau * source)


def hard_update(source_vars: Sequence[tf.Variable], target_vars: Sequence[tf.Variable]) -> None:
    """Copy source variables to target variables.
    Arguments:
        source_vars {Sequence[tf.Variable]} -- Source variables to copy from
        target_vars {Sequence[tf.Variable]} -- Variables to copy data to
    """
    # Tau of 1, so get everything from source and keep nothing from target
    soft_update(source_vars, target_vars, 1.0)

class SAC:
    def __init__(self, obs_dim, n_actions, act_lim, seed, discount, temperature, polyak_coef, lr,
                 hidden_layers, n_hidden_units, save_dir, env):

        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.act_lim = act_lim
        self.seed = seed
        self.discount = discount
        self.temperature = temperature
        self.polyak_coef = polyak_coef
        self.lr = lr
        self.save_dir = save_dir
        self.env = env
        self.gamma=discount

        ### Creating networks and optimizers ###
        # Policy network
        # action_output are the squashed actions and action_original those straight from the normal distribution
        logprob_epsilon = 1e-6  # For numerical stability when computing tf.log
        self.actor_network = ActorNetwork(hidden_layers, n_hidden_units, n_actions, logprob_epsilon)

        # 2 Soft q-functions networks + targets
        self.softq_network = SoftQNetwork(hidden_layers, n_hidden_units)
        self.softq_target_network = SoftQNetwork(hidden_layers, n_hidden_units)

        self.softq_network2 = SoftQNetwork(hidden_layers, n_hidden_units)
        self.softq_target_network2 = SoftQNetwork(hidden_layers, n_hidden_units)

        # Building up 2 soft q-function with their relative targets
        input1 = tf.keras.Input(shape=(obs_dim), dtype=tf.float32)
        input2 = tf.keras.Input(shape=(n_actions), dtype=tf.float32)

        self.softq_network(input1, input2)
        self.softq_target_network(input1, input2)
        hard_update(self.softq_network.variables, self.softq_target_network.variables)

        self.softq_network2(input1, input2)
        self.softq_target_network2(input1, input2)
        hard_update(self.softq_network2.variables, self.softq_target_network2.variables)


        # Optimizers for the networks
        self.softq_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        self.softq_optimizer2 = tfa.optimizers.RectifiedAdam(learning_rate=lr)
        self.actor_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr)


    def softq_value(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network(states, actions)

    def softq_value2(self, states: np.ndarray, actions: np.ndarray):
        return self.softq_network2(states, actions)

    def actions(self, states: np.ndarray) -> np.ndarray:
        """Get the actions for a batch of states."""
        return self.actor_network(states)[0]

    def action(self, state: np.ndarray) -> np.ndarray:
        """Get the action for a single state."""
        return self.actor_network(state[None, :])[0][0]

    def step(self, obs):
        return self.actor_network(obs)[0]

    @tf.function
    def train(self, sample, action_batch, batch_size):
        state0_batch = sample["states0"]
        reward_batch = sample["rewards"]
        state1_batch = sample["states1"]
        terminal1_batch = sample["terminals1"]

        # Computing action and a_tilde
        action, action_logprob2 = self.actor_network(state1_batch)

        value_target1 = self.softq_target_network(state1_batch, action)
        value_target2 = self.softq_target_network2(state1_batch, action)

        # Taking the minimum of the q-functions values
        next_value_batch = tf.math.minimum(value_target1, value_target2) - self.temperature * action_logprob2

        # Computing target for q-functions
        softq_targets = reward_batch + self.gamma * (1 - terminal1_batch) * tf.reshape(next_value_batch, [-1])
        softq_targets = tf.reshape(softq_targets, [batch_size, 1])

        # Gradient descent for the first q-function
        with tf.GradientTape() as softq_tape:
            softq = self.softq_network(state0_batch, action_batch)
            softq_loss = tf.reduce_mean(tf.square(softq - softq_targets))

        # Gradient descent for the second q-function
        with tf.GradientTape() as softq_tape2:
            softq2 = self.softq_network2(state0_batch, action_batch)
            softq_loss2 = tf.reduce_mean(tf.square(softq2 - softq_targets))

        # Gradient ascent for the policy (actor)
        with tf.GradientTape() as actor_tape:
            actions, action_logprob = self.actor_network(state0_batch)
            new_softq = tf.math.minimum(self.softq_network(state0_batch, actions), self.softq_network2(state0_batch, actions))

            # Loss implementation from the pseudocode -> works worse
            #actor_loss = tf.reduce_mean(action_logprob - new_softq)

            # New actor_loss -> works better
            advantage = tf.stop_gradient(action_logprob - new_softq)
            actor_loss = tf.reduce_mean(action_logprob * advantage)


        # Computing the gradients with the tapes and applying them
        actor_gradients = actor_tape.gradient(actor_loss, self.actor_network.trainable_weights)
        softq_gradients = softq_tape.gradient(softq_loss, self.softq_network.trainable_weights)
        softq_gradients2 = softq_tape2.gradient(softq_loss2, self.softq_network2.trainable_weights)

        # Minimize gradients wrt weights
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_weights))
        self.softq_optimizer.apply_gradients(zip(softq_gradients, self.softq_network.trainable_weights))
        self.softq_optimizer2.apply_gradients(zip(softq_gradients2, self.softq_network2.trainable_weights))


        # Update the weights of the soft q-function target networks
        soft_update(self.softq_network.variables, self.softq_target_network.variables, self.polyak_coef)
        soft_update(self.softq_network2.variables, self.softq_target_network2.variables, self.polyak_coef)

        # Computing mean and variance of soft-q function
        softq_mean, softq_variance = tf.nn.moments(softq, axes=[0])

        return softq_mean[0], tf.sqrt(softq_variance[0]), softq_loss, actor_loss, tf.reduce_mean(action_logprob)


    def save(self):
        self.actor_network.save_weights(self.save_dir+"/actor.ckpt")
        print("Model saved!")

    def load(self, filepath):
        self.actor_network.load_weights(filepath+"/actor.ckpt")
        print("Model loaded!")




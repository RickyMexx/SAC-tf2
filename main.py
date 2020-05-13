######### Configuration files #########
from conf import *
from SAC.SAC_conf import *
#######################################

import os
from tensorflow.python.util import deprecation
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow as tf
import gym

from common.agent import Agent
from common.utils import ReplayBuffer
from SAC.SAC_rla import SAC


if __name__ == "__main__":
    print('TensorFlow version: %s' % tf.__version__)
    print('Keras version: %s' % tf.keras.__version__)

    train_env = gym.make(env_id)
    test_env = gym.make(env_id)
    train_env.seed(seed)
    test_env.seed(seed)

    # Creating a ReplayBuffer for the training process
    replay_buffer = ReplayBuffer(buffer_size)

    obs_dim = train_env.observation_space.shape[0]
    n_actions = train_env.action_space.shape[0]
    act_lim = train_env.action_space.high
    model_dir = os.path.join(save_dir, exp_name)

    # We first choose a model
    model = SAC(obs_dim=obs_dim, n_actions=n_actions, act_lim=act_lim, save_dir=model_dir,
                discount=gamma, lr=lr, seed=seed, polyak_coef=polyak_coef, temperature=temperature,
                hidden_layers=hidden_layers, n_hidden_units=n_hidden_units, env=train_env)

    # Now we are going to create an Agent to train / test the model
    agent = Agent(model=model, replay_buffer=replay_buffer, train_env=train_env, test_env=test_env,
                  replay_start_size=replay_start_size, n_episodes=n_episodes, batch_size=batch_size, n_actions=n_actions)

    if train:
        # Perform a training using an agent with a certain model
        agent.train()
    else:
        # We are going to test an existing model
        agent.test(model_path)
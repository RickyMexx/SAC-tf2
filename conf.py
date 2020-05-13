# TEST / TRAIN PARAMETERS #
replay_start_size = 10000               # Number of steps in order to fill the ReplayBuffer with random samples
n_episodes = 100                        # Number of episodes
batch_size = 128                        # Get batch_size sample from the ReplayBuffer

buffer_size = int(1e6)                  # ReplayBuffer size
train_n_steps = 1e4                     # Number of training steps
seed = 0                                # Initial seed for the environment

# ENVIRONMENT / MODE PARAMETERS #
save_dir = 'models'                     # Directory for the saved models
exp_name = 'test'                       # Directory name for the current experiment
env_id='MountainCarContinuous-v0'       # Environment to train/test a model on
#env_id='Walker2d-v2'                   # Usually trained with 1000 episodes

train = True                            # If true the agent will perform a training. Otherwise a test will be performed using model_path
model_path = save_dir + '/' + exp_name

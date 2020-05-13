from common.utils import *

class Agent:
    def __init__(self, model, replay_buffer, train_env, test_env, replay_start_size,
                 n_episodes, batch_size, n_actions):
        self.model = model
        self.replay_buffer = replay_buffer
        self.train_env = train_env
        self.test_env = test_env
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.n_timesteps = train_env.spec.tags.get("wrapper_config.TimeLimit.max_episode_steps")
        self.total_steps = 0
        self.total_episodes = 0

    def train(self):
        check = 1
        episode_lengths = [None] * self.n_episodes
        episode_rewards = [None] * self.n_episodes

        # Parameters for the consecutive actions technique
        cons_acts = 4
        prob_act = 0.5

        # Noise + epsilon parameters
        noise = OUNoise(self.n_actions)
        epsilon = 1
        epsilon_min = 0.1
        epsilon_dk = 0.999

        for e in range(self.n_episodes):
            state = self.train_env.reset().astype(np.float32)
            episode_reward = 0
            episode_length = 0

            for k in range(self.n_timesteps):
                action = self.model.action(state)

                #### Techniques to force exploration, useful in sparse rewards environments ####

                # Using the consecutive steps technique
                if check == 1 and np.random.uniform() < prob_act:
                    # print(self.replay_buffer.n_entries)
                    for i in range(cons_acts):
                        self.train_env.step(action) 

                '''
                # Using OUNoise technique + epsilon-greedy
                if np.random.uniform() < epsilon:
                    action = noise.get_action(action, k)
                if check==0 and epsilon > epsilon_min:
                    epsilon = epsilon * epsilon_dk
                '''
                ################################################################################

                new_state, reward, done, _ = self.train_env.step(action)
                new_state = new_state.astype(np.float32)
                episode_length += 1
                self.total_steps += 1
                episode_reward += reward
                self.replay_buffer.add(state, action, reward, new_state, done)
                if self.replay_buffer.n_entries > self.replay_start_size:
                    if check == 1:
                        print("The buffer is ready, training is starting!")
                        check = 0


                    sample = self.replay_buffer.get_batch(self.batch_size)
                    softq_mean, softq_std, softq_loss, actor_loss, action_logprob_mean = self.model.train(sample,
                                                                                                          np.resize(sample["actions"], [self.batch_size, self.n_actions]),
                                                                                                          self.batch_size)

                    # print("Actor loss is", np.array(actor_loss))
                    # print("Q loss is", np.array(softq_loss))

                state = new_state

                if done:
                    episode_lengths[e] = k
                    episode_rewards[e] = episode_reward
                    self.total_episodes += 1
                    print("Episode n.", self.total_episodes, "is end! The reward is:", episode_reward,
                          ", number of steps:", k)
                    self.model.save()
                    break

        plot_episode_stats(episode_lengths, episode_rewards)
        plot_reward(episode_rewards)

    def test(self, model_path):
        self.model.load(model_path)
        while True:
            obs, done = self.test_env.reset(), False
            while not done:
                action = self.model.action(obs.astype(np.float32))
                obs, reward, done, info = self.test_env.step(action)
                self.test_env.render()
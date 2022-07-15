import replay
import utils
import numpy as np
import copy
import time
import hashlib, os, csv
import ddpg
import networks
import torch
from Env.pybullet_adapted.gym_locomotion_envs import AntEnv as Env

class Trainer(object):
    def __init__(self):

        # Create the environment and render
        self._env = Env(True)
        # Extract the dimesions of states and actions
        observation_dim = self._env.observation_space.low.size
        action_dim = self._env.action_space.low.size

        self._device = 'cpu'
        # Uncomment if you do trianing on the GPU
        # self._device = 'cuda:0'

        hidden_sizes = [256] * 2
        self._q_net = networks.QvalueNetwork(hidden_sizes=hidden_sizes, input_size = observation_dim + action_dim).to(device=self._device)
        self._target_q_net = networks.QvalueNetwork(hidden_sizes=hidden_sizes, input_size = observation_dim + action_dim).to(device=self._device)
        self._policy_net = networks.PolicyNetwork(hidden_sizes=hidden_sizes, input_size = observation_dim, output_size=action_dim).to(device=self._device)
        self._target_policy_net = networks.PolicyNetwork(hidden_sizes=hidden_sizes, input_size = observation_dim, output_size=action_dim).to(device=self._device)
        # Target update rate
        tau = 0.001

        # Set to true if you want to slow down the simulator
        self._slow_simulation = False

        # Create the ddpg agent
        self.agent = ddpg.DDPG(q_net=self._q_net, target_q_net=self._target_q_net, policy_net=self._policy_net, target_policy_net=self._target_policy_net, tau=tau, device=self._device)


        # Create a replay buffer - we use here one from a popular framework
        self._replay = replay.SimpleReplayBuffer(
            max_replay_buffer_size=1000000,
            observation_dim=observation_dim,
            action_dim=action_dim,
            env_info_sizes={},)

        # Stores the cummulative rewards
        self._rewards = []
        self._rewards_test = []

        # The following logging works only on linux at the moment - might cause issues if you use windows

        folder = 'experiment_data_test_runs'
        #generate random hash string - unique identifier if we start
        # multiple experiments at the same time
        rand_id = hashlib.md5(os.urandom(128)).hexdigest()[:8]
        self._file_path = './' + folder + '/' + time.ctime().replace(' ', '_') + '__' + rand_id

        # Create experiment folder
        if not os.path.exists(self._file_path):
          os.makedirs(self._file_path)

    def collect_training_data(self, noise=False, std=0.2):
        # Number of steps per episode - 300 is okay, but you might want to increase it
        nmbr_steps = 300

        current_state = self._env.reset()

        cum_reward = 0


        for _ in range(nmbr_steps):
            current_state_torch = torch.from_numpy(current_state).to(device=self._device)
            action = self._policy_net(current_state_torch)
            action = action.detach().cpu().numpy()
            if noise:
                # We add just some random noise to the action
                action = action + np.random.normal(scale=std, size=action.shape)
                # we have to make sure the action is still in the range [-1,1]
                action = np.clip(action, -1.0, 1.0)

            # Make a step in the environment with the action and receive the next state, a reward and terminal
            state, reward, terminal, info = self._env.step(action)

            # If we want to slow down the simulator
            if self._slow_simulation:
                time.sleep(0.1)

            # Just for logging
            cum_reward += reward

            # We add the transition to the replay buffer
            self._replay.add_sample(
                observation=current_state,
                action=action,
                reward=reward,
                terminal=terminal,
               next_observation=state,
               env_info = {})

            # The next current state
            current_state = state

        # Did we collect training data with noise (training) us using the policy only (test)
        if noise:
            self._rewards.append(float(cum_reward))
        else:
            self._rewards_test.append(float(cum_reward))

        # Just print the rewards for debugging
        # You could instead use visualizatin boards etc.
        print(self._rewards)


    def single_train_step(self):
        """
        This function collects first training data, then performs several
        training iterations and finally evaluates the current policy.
        """
        training_iters = 1000
        # Collect training data with noise
        self.collect_training_data(noise=True)
        for _ in range(training_iters):
            self.agent.train(self._replay)
        # Collect data without noise
        self.collect_training_data(noise=False)
        # Save the cum. rewards achieved into a csv file
        self.save_logged_data(rewards_training=self._rewards, rewards_test=self._rewards_test)

    def save_logged_data(self, rewards_training, rewards_test):
        """ Saves logged rewards to a csv file.
        """
        with open(
            os.path.join(self._file_path,
                'rewards.csv'), 'w') as fd:
                cwriter = csv.writer(fd)
                cwriter.writerow(rewards_training)
                cwriter.writerow(rewards_test)

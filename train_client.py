import replay
import utils
import numpy as np
import copy
import time
import hashlib, os, csv
import ddpg
import networks
import torch
import zmq
import tty
import sys
import termios
#from Env.pybullet_adapted.gym_locomotion_envs import AntEnv as Env

from Env.myEnv import AntEnv as Env

ACT_SIZE = 8
OBS_SIZE = 18
ctx1 = zmq.Context()
socket = ctx1.socket(zmq.REP)
socket.bind('tcp://*:5555')


class Trainer(object):
    def __init__(self):
        
        # Create the environment and render
        self._env = Env()
        # Extract the dimesions of states and actions
        observation_dim = OBS_SIZE
        action_dim = ACT_SIZE

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

        self.terminate_step=False
        self.terminated=False

    def collect_training_data(self, noise=False, std=0.2):
        #setting up keyboard interrupt for reset function
        orig_settings = termios.tcgetattr(sys.stdin)

        tty.setcbreak(sys.stdin)
        # Number of steps per episode - 300 is okay, but you might want to increase it
        nmbr_steps = 93

        current_state = self._env.reset()

        cum_reward = 0
        
        x=sys.stdin.read(1)[0]
        
        if x=="r":
            self.terminate_step=True           

            for _ in range(nmbr_steps):                
                current_state_torch = torch.from_numpy(current_state).to(device=self._device)
                current_state_torch=current_state_torch.float()
                action_d = self._policy_net(current_state_torch)
                
                action_d = action_d.detach().cpu().numpy()
                action=action_d.squeeze()
                #y=sys.stdin.read(1)[0]
                #print("action: ",action)
                if noise:
                    # We add just some random noise to the action
                    action = action + np.random.normal(scale=std, size=action.shape)
                    # we have to make sure the action is still in the range [-1,1]
                    action = np.clip(action, -1.0, 1.0)

                # Make a step in the environment with the action and receive the next state, a reward and terminal
                #state, reward, terminal, info = self._env.step(action)
                state, reward, terminal, info = self._env.step(action)
                if info[2]==True:
                    self.terminated=True
                    break

                # If we want to slow down the simulator
                if self._slow_simulation:
                    time.sleep(0.1)
                cum_reward += reward

                #enable user to ignore unrealistic rewards which occur at the limits of camera foeld of view
                if -30<reward<30:
                    # Just for logging
                    cum_reward += reward
                else:
                    
                    print("WARNING: reward is unrealistic - Press T to terminate step, or ANY KEY to not give any reward")
                    
                    z=sys.stdin.read(1)[0]
                    if z=="t":
                        self.terminated=True                      
                        state = self._env.reset()
                        print("EPISODE TERMINATED--")
                        break
                    else:
                        self.terminate_step=False
                        self.terminated=False
                        cum_reward+=0
                        pass
                
                #enable user to terminate the episode for training if any practical mistakes occurs
                limit_rewards=-20
                if cum_reward<-1 and self.terminate_step==True:
                     print("Cum_rewards less than ",limit_rewards," To Terminate Step: Press T, To continue: press ANY KEY")
                     y=sys.stdin.read(1)[0]
                     
                     if y=="t":
                        self.terminated=True                      
                        state = self._env.reset()
                        print("EPISODE TERMINATED--")
                        break
                    
                     else:
                        self.terminate_step=False
                        self.terminated=False
                        pass

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
                if self.terminated==True:
                    pass
                else:
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
        #training_iters = 1000
        training_iters =1000
        # Collect training data with noise
        self.collect_training_data(noise=True)
        orig_settings = termios.tcgetattr(sys.stdin)

        tty.setcbreak(sys.stdin)
        print("Want to train? Press y or n")
        x=sys.stdin.read(1)[0]
        if x=='y':
            for _ in range(training_iters):
                self.agent.train(self._replay)
                print("------Training -----")
            # Collect data without noise
            print("Policy Testing")
            self.collect_training_data(noise=False)
        else:
            return 0
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

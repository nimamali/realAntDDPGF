import torch
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

from Env.myEnv import AntEnv as Env

from student_ddpg.networks import PolicyNetwork

FRAME_STACKING = 4
ACT_SIZE = 8
#OBS_SIZE = 18*FRAME_STACKING
OBS_SIZE = 18
ctx1 = zmq.Context()
socket = ctx1.socket(zmq.REP)
socket.bind('tcp://*:5555')

class Trained(object):
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

        self.policy_n=PolicyNetwork()
        self.policy_n.load_state_dict(torch.load("/home/oli/realAntDDPG/SavedModels/policy_n/policy.pt"))
        self.policy_n.eval()

        


        



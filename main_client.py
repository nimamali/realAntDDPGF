
import train_client as T
import os
from datetime import datetime
import glob
import argparse
import pickle
import torch
import zmq
from Env.myEnv import AntEnv as Env

import tty
import sys
import termios

# connect to robot
ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')


def main():
    trainer = T.Trainer()
    env_1=Env()
    # for _ in range(1):
    #     print("Data Collection ON")
    #     trainer.collect_training_data(True, std = 0.5)
    # for _ in range(1000):
    #     print("Training--------step", 1_)
    #     trainer.single_train_step()
    for _ in range(5):
        print("Data Collection ON")
        trainer.collect_training_data(True, std = 0.5)
    for _ in range(50):
        print("Training--------step", _)
        trainer.single_train_step()
        steps=_
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin)

    for _ in range(10):
        x=sys.stdin.read(1)[0]
        print('Press s to train another step OR press t to terminate training--------')
        if x=='s':
            trainer.single_train_step()
            steps+=1
            print("Training--------step", steps)
        if x=='t':
            break
       
    print("DONE######")
    env_1.reset()


if __name__ == '__main__':
    main()

import train_client as T
import os
from datetime import datetime
import glob
import argparse
import pickle
import torch
import zmq

# connect to robot
ctx = zmq.Context()
socket = ctx.socket(zmq.REQ)
socket.connect('tcp://localhost:5555')

def main():
    trainer = T.Trainer()
    # for _ in range(1):
    #     print("Data Collection ON")
    #     trainer.collect_training_data(True, std = 0.5)
    # for _ in range(1000):
    #     print("Training--------step", _)
    #     trainer.single_train_step()
    for _ in range(5):
        print("Data Collection ON")
        trainer.collect_training_data(True, std = 0.5)
    for _ in range(20):
        print("Training--------step", _)
        trainer.single_train_step()
    print("DONE######")

if __name__ == '__main__':
    main()
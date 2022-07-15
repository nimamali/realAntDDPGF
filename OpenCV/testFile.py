import time
import torch
import zmq
import json
import threading
import numpy as np
import datetime
from collections import deque


import multiprocessing
import serial
FRAME_STACKING = 4
ACT_SIZE = 8
OBS_SIZE = 29 * FRAME_STACKING 

ctx = zmq.Context()

act_pub = ctx.socket(zmq.PUB)
act_pub.connect('tcp://localhost:3002')



last_ant_meas = None
last_camera_meas = None

last_frame_ant_meas = None
last_frame_camera_meas = None
last_frame_jpos = None

ctx1 = zmq.Context()

obs_sub = ctx1.socket(zmq.SUB)
obs_sub.connect('tcp://localhost:3001')
obs_sub.setsockopt(zmq.SUBSCRIBE, b'')

print("observations collection started")
last_ant_time = 0
last_ant_meas = None
last_camera_meas = None
if __name__ == "__main__":

    while True:
        print("in=====")
        d = obs_sub.recv_multipart()
        print(d)
        if d[0][0] == ord(b"{"):
            j = json.loads(d[0])
            if j["id"] == "serial":
                last_ant_meas = j
                #print(last_ant_meas["s1_angle"])
                #print("typeT",type(last_ant_meas))
                #print("last_ant_meas: ", j)
                last_ant_time = j["ant_time"]
            if j["id"] == "external_tag_tracking_camera":
                last_camera_meas = j
                print("Camera: ",last_camera_meas)
        
   

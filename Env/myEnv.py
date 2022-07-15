import time
from turtle import distance
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
#OBS_SIZE = 18 * FRAME_STACKING 
OBS_SIZE = 18
ctx = zmq.Context()

act_pub = ctx.socket(zmq.PUB)
act_pub.connect('tcp://localhost:3002')


last_ant_meas = None
last_camera_meas = None

last_frame_ant_meas = None
last_frame_camera_meas = None
last_frame_jpos = None

past_obses = deque([np.zeros(OBS_SIZE//FRAME_STACKING)]*FRAME_STACKING, maxlen=FRAME_STACKING)

def collect_and_distribute_measurements(child_conn):
    ctx1 = zmq.Context()

    obs_sub = ctx1.socket(zmq.SUB)
    obs_sub.connect('tcp://localhost:3001')
    obs_sub.setsockopt(zmq.SUBSCRIBE, b'')

    print("observations collection started")
    last_ant_time = 0
    last_ant_meas = None
    last_camera_meas = None

    while True:
        d = obs_sub.recv_multipart()
        if d[0][0] == ord(b"{"):
            j = json.loads(d[0])
            if j["id"] == "serial":
                last_ant_meas = j
                #print(last_ant_meas["s1_angle"])
                #print("typeT",type(last_ant_meas))
                #print("last_ant_meas: ", last_ant_meas)
                last_ant_time = j["ant_time"]
            if j["id"] == "external_tag_tracking_camera":
                last_camera_meas = j
                #print("Camera: ",last_camera_meas)
         
        if child_conn.poll():
            #print("poll==")
            child_conn.recv()
            child_conn.send([last_ant_meas, last_camera_meas])
            #child_conn.send(last_ant_meas)

class EnvironmentHandler():
    def __init__(self):
        self.running = True
        self.parent_conn, self.child_conn = multiprocessing.Pipe()
        self.p = multiprocessing.Process(target=collect_and_distribute_measurements, args=(self.child_conn,))
        self.p.start()
        self.zero_j_cnt = 0
        self.zero_c_cnt = 0

    def step(self):
        global last_ant_meas, last_camera_meas, last_frame_ant_meas, last_frame_camera_meas, last_frame_jpos, past_obses
        self.parent_conn.send([])
        last_ant_meas, last_camera_meas = self.parent_conn.recv()

        while last_ant_meas == None and last_camera_meas == None:
            time.sleep(0.01)

        default_dt = 0.05 # 
        camera_dt = (last_camera_meas['server_epoch_ms'] - last_frame_camera_meas['server_epoch_ms']) / 1000 if last_frame_camera_meas != None else None
        #print("last_camera based dt", camera_dt)
        if last_frame_ant_meas != None:
            print("ant time vs last ant", last_ant_meas['ant_time'], last_frame_ant_meas['ant_time'])
        joint_dt = (float(last_ant_meas['ant_time']) - float(last_frame_ant_meas['ant_time'])) / 1000 if last_frame_ant_meas != None else None
        #print("last_joint based dt", joint_dt)
        #print("Camera: ",last_camera_meas)
        # sanity checks
        if camera_dt == 0:
            self.zero_c_cnt += 1
            if self.zero_c_cnt > 3:
                print("observations stuck, quitting (camera)")
                quit()
            camera_dt = default_dt 
        else:
            self.zero_c_cnt = 0

        if joint_dt == 0:
            self.zero_j_cnt += 1
            if self.zero_j_cnt > 3:
                print("observations stuck, quitting (serial)")
                quit()
            joint_dt = default_dt 
        else:
            self.zero_j_cnt = 0

        # Ote Robotics RealAnt action spaceroll
        # 0 - hip right front  
        # 1 - ankle right front
        # 2 - hip right back
        # 3 - ankle right back
        # 4 - hip left back
        # 5 - ankle left back
        # 6 - hip left front
        # 7 - ankle left front

        # servo angles to joint positions
        angles = ["s%d_angle" %d for d in range(1,9)]
       
        angles = np.array([float(last_ant_meas[a]) for a in angles])


        # re-order and position accordingly
        jpos = np.zeros(8)
        servo_middle = 512  # ax12a value for servo middle position
        servo_half_range = 512-224  # ax12a range from middle to zero degrees
        jpos[0] = -np.clip(-(angles[6] - servo_middle) / servo_half_range, -1, 1)
        jpos[1] = (np.clip((angles[7] - servo_middle) / servo_half_range, -1, 0) * 2 + 1)
        jpos[2] = -np.clip(-(angles[4] - servo_middle) / servo_half_range, -1, 1)
        jpos[3] = -(np.clip((angles[5] - servo_middle) / servo_half_range, -1, 0) * 2 + 1)
        jpos[4] = -np.clip(-(angles[2] - servo_middle) / servo_half_range, -1, 1)
        jpos[5] = -(np.clip((angles[3] - servo_middle) / servo_half_range, -1, 0) * 2 + 1)
        jpos[6] = -np.clip(-(angles[0] - servo_middle) / servo_half_range, -1, 1)
        jpos[7] = (np.clip((angles[1] - servo_middle) / servo_half_range, -1, 0) * 2 + 1)

        jpos_vel = (last_frame_jpos - jpos)/joint_dt if last_frame_jpos is not None else np.zeros((8,))
        torso_pos=np.array([last_camera_meas["x"],last_camera_meas["y"]])
        #print("torso pos: ", torso_pos)

        state = np.concatenate([torso_pos,jpos, jpos_vel])

        # past_obses.append(state)
        # state = np.concatenate(past_obses)

        last_frame_ant_meas = last_ant_meas
        last_frame_camera_meas = last_camera_meas
        last_frame_jpos = jpos

        info = np.array([last_camera_meas["x"], last_camera_meas["y"]])
        #return obs, info

        self.rewards=state[0]
        #self.reward +=sum(self.rewards)
        #print ("reward: ", self.rewards)

        return state,self.rewards, True, info

    def apply_controls(self, a):
        a = np.array(a)
        #print("set_points: ",a)

        a = (np.clip(np.array(a),-1,1) + 1) / 2.0  # scale to 0...1

        hip_range = 256 
        hip_offset = 368 # this limits hip from middle to +-45deg
        ankle_range = 224
        ankle_offset = 288

        # adjust ordering, range and offsets for the physical ant
        b = np.zeros(8)
        b[0] = a[6] * hip_range + hip_offset      # right front
        b[1] = a[7] * ankle_range + ankle_offset
        b[2] = a[4] * hip_range + hip_offset      # right back
        b[3] = a[5] * ankle_range + ankle_offset
        b[4] = a[2] * hip_range + hip_offset     # left back
        b[5] = a[3] * ankle_range + ankle_offset 
        b[6] = a[0] * hip_range + hip_offset     # left front
        b[7] = a[1] * ankle_range + ankle_offset
        a = b
        #print("set_points: ",[b"cmd", b"s1 %d s2 %d s3 %d s4 %d s5 %d s6 %d s7 %d s8 %d\n" % (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])])
        act_pub.send_multipart([b"cmd", b"s1 %d s2 %d s3 %d s4 %d s5 %d s6 %d s7 %d s8 %d\n" % (a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7])])
        #act_pub.send_multipart([b"cmd", b"s1 224 s2 512 s3 224 s4 512 s5 224 s6 512 s7 224 s8 512\n"])
    
    def reset_tracking(self):
        """ reset tracking state and tracking camera pose """
        global last_frame_ant_meas, last_frame_camera_meas, last_frame_jpos, past_obses

        last_frame_ant_meas = None
        last_frame_camera_meas = None
        last_frame_jpos = None

        past_obses = deque([np.zeros(OBS_SIZE//FRAME_STACKING)]*FRAME_STACKING, maxlen=FRAME_STACKING)

        act_pub.send_multipart([b"tracking_cmd", b"reset_tracking"])

    def reset_servos(self):
        """ reset orientation and servos to initial state """
        act_pub.send_multipart([b"cmd", b"reset\n"])

    def reset_stand(self):
        """standing stance to start walkign motion"""
        #act_pub.send_multipart([b"cmd", b"s2 300 s4 300 s6 300 s8 300\n"])
        act_pub.send_multipart([b"cmd", b"s2 210 s4 210 s6 210 s8 210\n"])

    def detach_servos(self):
        """ cut torque to servos to save power """
        act_pub.send_multipart([b"cmd", b"detach_servos\n"])
    
    def attach_servos(self):
        """ enable torque to servos to start actuation """
        act_pub.send_multipart([b"cmd", b"attach_servos\n"])

    def command(self):
        # act_pub.send_multipart([b"cmd", b"s1 512 s2 512 s3 512 s4 512 s5 512 s6 512 s7 512 s8 512\n"])
        # time.sleep(1)
        act_pub.send_multipart([b"cmd", b"s1 224 s2 512 s3 224 s4 512 s5 224 s6 512 s7 224 s8 512\n"])
        time.sleep(1)

class AntEnv():
    def __init__(self):
        
        self.env=EnvironmentHandler()
        time.sleep(0.5)
        self.env.reset_servos()
        time.sleep(0.5)
        self.env.detach_servos()
        #env.command()
        time.sleep(1)

        print("Running")

    def reset(self):
        """ reset robot joints and everything before rollout """
        self.env.reset_tracking()
        self.env.reset_servos()
        time.sleep(0.05)
        self.env.reset_stand()
        time.sleep(0.2)
        r=np.zeros((1,18))
        #r[2]=r[4]=r[6]=r[8]=300
        state, reward, done, info = self.env.step()
        self._old_x = info[0]
        return state

    def detach_servos(self):
        """ cut torque to servos to save power """
        self.env.detach_servos()

    def attach_servos(self):
        """ enable torque to servos to start actuation """
        self.env.attach_servos()

    # def get_state(self):
    #     """ get current state of joints and camera x,y data """
    #     return self.env.get_obs()

    def apply_controls(self,pid_setpoints):
        """apply controls to the robot"""
        self.attach_servos()
        time.sleep(0.2)
        self.reset()
        time.sleep(0.2)
        self.env.apply_controls(pid_setpoints)

    def step(self,a):      
        action_repeat = 1
        self.apply_controls(a)
        last_time = datetime.datetime.utcnow()
        time.sleep(0.1) # SOmething to tune
        state, reward, done, info = self.env.step()
        x1 = info[0]
        reward = x1 -self._old_x 
        print("reward: ",reward)
        self._old_x = x1
        # # x1=last_camera_meas["x"]
        # for _ in range(action_repeat):
        #     state, reward, done, info = self.env.step()
        #     time.sleep(0.05)
        #     if _==0:
        #         x1=info[0]
        #         print("x1: ",x1)
        #     if _==2:
        #         x2=info[0]
        #         print("x2: ",x2)
        # # x2=last_camera_meas["x"]
        # now = datetime.datetime.utcnow()
        # interval = (now - last_time).total_seconds()
        # print("interval: ",interval)
        # last_time = now
        # distance=x2-x1
        # speed=distance/interval
        #speed = self.robot.robot_body.speed()
        #vx = speed[0]
        #reward = vx 
        return state, reward, done, info
    def command(self):
        self.attach_servos()
        time.sleep(0.2)
        self.reset()
        time.sleep(0.2)
        self.env.command()

# if __name__ == '__main__':
#     ant=AntEnv()
#     ant.reset()
#     action = np.random.uniform(-1, 1, ACT_SIZE)
#     print("action: ",action)
#     ant.step(action) 
        









    



        
            

        
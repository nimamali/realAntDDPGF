# import the necessary packages
from collections import deque
from sqlite3 import Timestamp
from typing import OrderedDict
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
from collections import OrderedDict, deque
import zmq
import datetime
import json

class Marker():
    def __init__(self,):
        self.greenLower = (33, 38, 55)
        self.greenUpper = (100, 255, 255)
        self.redLower=(0,161,0)
        self.redUpper=(94,194,255)

        self.pts = deque(maxlen=args["buffer"])
        self.startpoints=deque(maxlen=4)
        self.endpoints=deque(maxlen=4)
        self.current_pt=deque(maxlen=1)

        self.x_ps=deque(maxlen=4)
        self.y_ps=deque(maxlen=4)

        self.start=1
        self.end=0

    def vidStream(self):
        # if a video path was not supplied, grab the reference
        # to the webcam
        # if not args.get("video", False):
        #     self.vs = VideoStream(src=0).start()
        # # otherwise, grab a reference to the video file
        # else:
        #     self.vs = cv2.VideoCapture(args["video"])
        self.vs = cv2.VideoCapture(args["video"])
        # allow the camera or video file to warm up
        time.sleep(2.0)

    def getCenter(self):
        int_x0=0
        int_y0=0
        
        #keep looping
        # grab the current frame
        frame = self.vs.read()
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        # if frame is None:
        #     break
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.greenLower, self.greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None

    # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cord="x: " + str(center[0]) + ", y: " + str(center[1])
                int_x0=int(center[0])
                int_y0=int(center[1])
                cv2.putText(frame, cord, (int_x0 - 10, int_y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #update the points queue
        self.pts.appendleft(center)
        

        # loop over the set of tracked points
        for i in np.arange(1, len(self.pts)):
            # if either of the tracked points are None, ignore
            # them
            if self.pts[i - 1] is None or self.pts[i] is None:
                continue
            
        # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)


        # show the frame to our screen and increment the frame counter
        cv2.imshow("Frame", frame)
        return int_x0,int_y0

    def resetStart(self):
        self.startpoints.clear()

    def resetEnd(self):
        self.endpoints.clear()

    def reset(self):
        self.startpoints.clear()
        self.endpoints.clear()

    def printPoints(self):
        print("Start point: ",self.startpoints," End points: ",self.endpoints)

def main():
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.connect("tcp://127.0.0.1:3002")

    v1=Marker()
    v1.vidStream()
    i=0
    while True:

        x0,y0=v1.getCenter()
        utcnowms = int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000) 
        d=OrderedDict()
        d["id"]="external_tag_tracking_camera"
        d["server_epoch_ms"]=utcnowms
        #d["server_epoch_ms"]=Timestamp
        #d["dist"]=float(np.linalg.norm(cord))
        d["x"]=x0
        d["y"]=y0
        d["sent"]=False
        #print(d)
        if d is not None:
            d["sent"] = True
            sock.send_json(d)
            #print(d)           
         
    
        # if the 'q' key is pressed, stop the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",type=int,default=4,
        help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=32,
        help="max buffer size")
    args = vars(ap.parse_args())

    main()

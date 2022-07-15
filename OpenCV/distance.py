# import the necessary packages
from collections import deque

from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import threading
import asyncio

class CenterDistance() :
    def __init__( self,d=0,Upper=0,Lower=0,start_pt=0):
        self.d=d
        self.start_pt=start_pt
        
        # define the lower and upper boundaries of the "green"
        # ball in the HSV color space, then initialize the
        # list of tracked points
        greenLower = (33, 38, 55)
        greenUpper = (72, 255, 255)
        redLower=(0,161,0)
        redUpper=(94,194,255)

        self.Lower=greenLower
        self.Upper=greenUpper
        self.pts = deque(maxlen=args["buffer"])
        self.strt_pt=deque(maxlen=10)
        self.end_pt=deque(maxlen=2)

        # t=threading.Thread(target=self.startpt,args=(self,)).start()
        
        # t1=threading.Thread(target=self.distance,args=(self,)).start()

    def videoStream(self):
        # if a video path was not supplied, grab the reference
        # to the webcam
        if not args.get("video", False):
            self.vs = VideoStream(src=0).start()
        # otherwise, grab a reference to the video file
        else:
            self.vs = cv2.VideoCapture(args["video"])
        # allow the camera or video file to warm up
        time.sleep(2.0)
        return self.vs

    def getFrame(self):
    # grab the current frame
        self.frame = self.vs.read()
        # handle the frame from VideoCapture or VideoStream
        self.frame = self.frame[1] if args.get("video", False) else self.frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if self.frame is None:
            return 0

    def getCenter(self):        
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = self.vs.read()
        # handle the frame from VideoCapture or VideoStream
        frame = frame[1] if args.get("video", False) else frame
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        if frame is None:
            return 0
       
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, self.Lower, self.Upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball def drawPath():

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
        self.strt_pt.appendleft(center)
        self.end_pt.appendleft(center)         
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
        # if cv2.waitKey(1) & 0xFF == ord('s'):
        #     strp=self.startpt()
        #     print('start_point:',strp)

        # elif cv2.waitKey(1) & 0xFF == ord('d'):
        #     distance=self.distance()
        #     print('distance: ', distance )
        
    def startpt(self):
        self.x_ps=deque(maxlen=10)
        self.y_ps=deque(maxlen=10)
        strtpt=self.strt_pt
        for i in np.arange(1,len(self.strt_pt)):
            if self.strt_pt[i - 1] is None or self.strt_pt[i] is None:
                    continue
            self.x_ps.appendleft(self.strt_pt[i][0])
            self.y_ps.appendleft(self.strt_pt[i][1])
            

        self.str_ptx=np.average(self.x_ps) 
        self.str_pty=np.average(self.y_ps) 
        self.start_points=np.array((self.str_ptx,self.str_pty))
        print('start pt:',self.start_points)
        return self.start_points  
        
    def distance(self):
        self.x_pe=deque(maxlen=10)
        self.y_pe=deque(maxlen=10)
        for i in np.arange(1,len(self.end_pt)):
            if self.end_pt[i - 1] is None or self.end_pt[i] is None:
                    continue
            self.x_pe.appendleft(self.end_pt[i][0])
            self.y_pe.appendleft(self.end_pt[i][1])
            

        self.end_ptx=np.average(self.x_pe) 
        self.end_pty=np.average(self.y_pe) 
        self.end_points=np.array((self.end_ptx,self.end_pty))

        #dist=np.linalg.norm(self.start_points-self.end_points)
        disty=np.subtract(self.end_pty,self.str_pty)
        print('distance: ',disty)

        return disty

    def reset(self):
        self.strt_pt.clear()
        self.end_pt.clear()
        return 0

    def stepDistance(self):
        self.startpt()
        time.sleep(0.5)
        self.distance()
        time.sleep(0.5)        


""" def set(self):
    if cv2.waitKey(1) & 0xFF == ord('s'):
        self.start_pt=np.average(self.pts)
    print(self.start_pt)
    pass """


if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video",type=int,default=4,
        help="path to the (optional) video file")
    ap.add_argument("-b", "--buffer", type=int, default=32,
        help="max buffer size")
    args = vars(ap.parse_args())
    v1=CenterDistance()
    v1.videoStream()
    while True:
        if v1.getFrame()==0:
            break
        else:          
            
            for i in range(50):
                
                v1.getCenter()
                v1.stepDistance()
                time.sleep(0.5)
                i+=1
                if i==30:
                    v1.reset()
                    print('reset============================================')
                                 # if the 'q' key is pressed, stop the loop
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break   
    
    

    
    


    

    
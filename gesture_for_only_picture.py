# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:58:00 2015

@author: yegor
"""

import cv2
import numpy as np


if __name__ == "__main__":    
    
    device = 0 
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        capture.open(device)
    
    # Установить разрешение
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
    
    ret = None
    frame = None
    while True:
        ret, frame = capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        num_fingers = 0
        
        h,w = frame.shape[:2]
        cv2.circle(frame, (w/2, h/2), 2, [0,102,255], 2)
        cv2.rectangle(frame, (w/3, h/3), (w*2/3, h*2/3), [0,102,255], 2)
        cv2.rectangle(frame, (w/2-10, h/2 + 10), (w/2+10, h/2-10),  [0,255,255], 1)
        cv2.putText(frame, "Num of fingers: " + str(num_fingers), (w - 300,h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('example1.jpg', frame)                
            break
    
    capture.release()
    cv2.destroyAllWindows()

    
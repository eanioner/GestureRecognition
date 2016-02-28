# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 19:41:57 2015

@author: yegor

"""

from abc import ABCMeta
import numpy as np
import wx
import cv2

class BaseLayout(wx.Frame):
    __metaclass__ = ABCMeta

    def __init__(self, ancestor, _id, window_name, capture, frame_per_second = 10):
        self.capture = capture
        self.fps = frame_per_second
        
        res, frame = self.get_frame()
        self.m_frame = frame.copy()
        if not res:
            print "Не могу показать кадр из камеры"
            raise SystemExit
        
        # Определение высоты и ширины изображения
        self.imgH, self.imgW = frame.shape[:2]
        
        # Получить BitMap
        self.bmp = wx.BitmapFromBuffer(self.imgW, self.imgH, frame)
        
        wx.Frame.__init__(self, ancestor, _id, window_name, size = (self.imgW, self.imgH))
        
        self._init_base_layout()
        self._create_base_layout()
        self.cnt = 0
        
    def _init_base_layout(self):
        self.timer = wx.Timer(self)
        self.timer.Start(1000./self.fps)
        self.Bind(wx.EVT_TIMER, self.get_next_frame)
        self.Bind(wx.EVT_PAINT, self.paint_layout)
        self._init_custom_layout()
    
    def _create_base_layout(self):
        self.pnl = wx.Panel(self, -1, size=(self.imgW, self.imgH))
        self.pnl.SetBackgroundColour(wx.BLUE)
        self.panels_vertical = wx.BoxSizer(wx.VERTICAL)
        self.panels_vertical.Add(self.pnl, 1, flag=wx.EXPAND | wx.TOP, border=1)
        self.SetMinSize((self.imgW, self.imgH))
        self.SetSizer(self.panels_vertical)
        self.Centre()
        
    def get_next_frame(self, event):
        res, frame = self.get_frame()
        self.m_frame = frame.copy()
        if res:
            try:
                self._process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
#                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                h,w = frame.shape[:2]
#                cv2.circle(frame, (w/2, h/2), 2, [0,102,255], 2)
#                cv2.rectangle(frame, (w/3, h/3), (w*2/3, h*2/3), [0,102,255], 2)
#                cv2.rectangle(frame, (w/2-10, h/2 + 10), (w/2+10, h/2-10),  [0,255,255], 1)
#                cv2.putText(frame, "Num of fingers: " + str(0), (w - 300,h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
                self.bmp.CopyFromBuffer(self.m_frame)       
            except:
                print "Не могу обработать кадр"
                raise SystemExit
           
            self.Refresh(eraseBackground=False)
            
            
    def paint_layout(self, event):
        deviceContext = wx.BufferedPaintDC(self.pnl)
        deviceContext.DrawBitmap(self.bmp, 0, 0)
    
    def get_frame(self):
        return self.capture.read()
    
    
    def _init_custom_layout(self):
        self.hand_gestures = RecognitionOfFingers()
        
    def _process_frame(self, frame):
         
        
#        np.clip(frame, 0, 2**10 - 1, frame)
#        frame >>= 2
        frame = frame.astype(np.uint8)

#        num_fingers = 0
#        img_draw = frame        
        
        num_fingers, img_draw = self.hand_gestures.recognize(frame, self.m_frame)
        
        h,w = frame.shape[:2]
        cv2.circle(self.m_frame, (w/2, h/2), 2, [0,102,255], 2)
        cv2.rectangle(self.m_frame, (w/3, h/3), (w*2/3, h*2/3), [0,102,255], 2)
        cv2.rectangle(self.m_frame, (w/2-10, h/2 + 10), (w/2+10, h/2-10),  [0,255,255], 1)
        cv2.putText(self.m_frame, "Num of fingers: " + str(num_fingers), (w - 300,h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        
        print str(num_fingers)
        
        return img_draw

def angle_rad(v1, v2):        
    return np.arctan2(np.linalg.norm(np.cross(v1,v2)), np.dot(v1,v2))
    
def deg2rad(angle_deg):
    return angle_deg/180.0 * np.pi
    
class RecognitionOfFingers:
    def __init__(self):
        self.abs_depth_dev = 20
        self.thresh_deg = 85.0
        
    def recognize(self, img_gray, m_frame):
        
        self.h, self.w = img_gray.shape[:2]
        
        segment = self._segment_arm(img_gray)
#        img_gray = self._segment_arm(img_gray)
#        img_gray = segment.copy()
        (contour, defects) = self._find_hull_defects(segment)
        
        img_draw = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
              
        num_fingers  = 0
        (num_fingers, img_draw) = self._detect_num_fingers(contour, defects, m_frame)
        return (min(num_fingers, 5), img_draw)

    def _segment_arm(self, frame):
               
        
        center_half = 10
        center = frame[self.h / 2 -  center_half:self.h/2 + center_half, self.w/2-center_half:self.w/2+center_half]
        med_val = np.median(center)
        print "median in blue box: " + str(med_val)        
        
        frame = np.where(abs(frame-med_val) <= self.abs_depth_dev, 128, 0).astype(np.uint8)
        
        
        
        kernel = np.ones((3,3), np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)

             
        
        small_kernel = 3
        frame[self.h/2-small_kernel:self.h/2+small_kernel, self.w/2 - small_kernel: self.w/2+small_kernel] = 128
        
        mask = np.zeros((self.h+2, self.w+2), np.uint8)
        flood = frame.copy()
        cv2.floodFill(flood, mask, (self.w/2, self.h/2), 255, flags = 4 | (255 << 8))
        
        
        ret, flooded = cv2.threshold(flood, 129, 255, 0)
       
        return flooded
        
    def _find_hull_defects(self, segment):
        
        
        try:
            contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            print "Ошибка при поиске контуров"       
        
        
        max_contour = max(contours, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(max_contour, True)
        max_contour = cv2.approxPolyDP(max_contour, eps, True)
        
        
        
        # поиск выпуклой оболочки и дефектов
        
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)
        
        return (max_contour, defects)
    
    def _detect_num_fingers(self, contour, defects, img_draw):
        if defects is None:
            return [0, img_draw]
        
        if len(defects) <= 2:
            return [0, img_draw]
            
        num_fingers = 1
        
        for i in range(defects.shape[0]):
            
            start_idx, end_idx, farthest_idx, _ = defects[i,0]
            start = tuple(contour[start_idx][0])
            end = tuple(contour[end_idx][0])
            far = tuple(contour[farthest_idx][0])
            
            cv2.line(img_draw, start, end, [0, 255, 0], 2)
            
            if angle_rad(np.subtract(start, far), np.subtract(end,far)) < deg2rad(self.thresh_deg):
                num_fingers += 1
                cv2.circle(img_draw, far, 5, [0,255,0], -1)
            else:
                cv2.circle(img_draw, far, 5, [255,0,0], -1)
        
        return ((num_fingers), img_draw)
        
        
    
   
        
def main():
   
   # Выббать устройство, установить захват видео с него.    
    device = 0 
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        capture.open(device)
    
    # Установить разрешение
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)
        
    # Создать приложение
    app = wx.App()
    # Задать внешний вид
    layout = BaseLayout(None, -1, 'Video', capture)
    # Показать приложение
    layout.Show(True)
    # Запустить приложение в основный цикл
    app.MainLoop()


    # Завершить работу приложения
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
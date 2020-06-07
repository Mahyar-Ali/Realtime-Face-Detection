#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 08:59:17 2020

@author: mahyar
"""

#Importing the required Libraries
import cv2

#Creating the cascade Objects
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#Function to detect face and eyes
def detect(gray,original):
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x, y, w, h) in faces:
        cv2.rectangle(original,(x,y),(x+w,y+h),(255,0,0),2)
        e_gray = gray[y:y+h,x:x+h]
        e_original = original[y:y+h,x:x+h]
        eyes = eyes_cascade.detectMultiScale(e_gray,1.1,3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(e_original,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return original
    
#Dealing with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _,frame = video_capture.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow("Video",canvas)
    if cv2.waitKey(1) and 0xFF==ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import tensorflow


detector=HandDetector(maxHands=1)
cap=cv2.VideoCapture(0)
classifier=Classifier("Models/keras_model.h5","Models/labels.txt")
folder="Data\B"
counter=0

#detecting hand
offset=20
imgSize=300

labels=['A','B','C']

while True:
    success,img=cap.read()
    imgOutput=img.copy()
    hands,img=detector.findHands(img)
    
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        imgWhite=np.ones((imgSize,imgSize,3),np.uint8)*255 #multiply it  with 255 so it becomes white
        
        
        
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgCropShape=imgCrop.shape
        
        aspectRatio=h/w
        try:
         if aspectRatio >1:
            k=imgSize/h
            wCal=math.ceil(k*w)
            imgResize=cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape=imgResize.shape
            wGap=math.ceil((imgSize-wCal)/2)
            imgWhite[:,wGap:wCal+wGap] = imgResize
            prediction,index=classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index)      
        
         else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            hGap=math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap,:] = imgResize
            prediction,index=classifier.getPrediction(imgWhite,draw=False)
            print(prediction,index) 
            
         cv2.rectangle(imgOutput,(x-offset,y-offset-50),(x-offset+90,y-offset-50+50),(255,0,255),cv2.FILLED)        
         cv2.putText(imgOutput,labels[index],(x,y-25),cv2.FONT_HERSHEY_COMPLEX,1.7,(255,255,255),2)      
         cv2.rectangle(imgOutput,(x-offset,y-offset),(x+w+offset,y+h+offset),(255,0,255),2)        
        
         if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0: # Add this condition to check if the image size is valid
          #imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop  
          cv2.imshow("imageCrop", imgCrop)
          cv2.imshow("imageWhite", imgWhite)
          
          #print(imgCrop.shape)
         else:
           print("Invalid image size:", imgCrop.shape)  # Add this line to check the image size
        except:
          print("don't zoom too much")
        
        
    cv2.imshow("image",imgOutput)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
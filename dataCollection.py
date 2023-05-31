import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

detector=HandDetector(maxHands=1)
cap=cv2.VideoCapture(0)
folder="Data\B"
counter=0

#detecting hand
offset=20
imgSize=300
while True:
    success,img=cap.read()
    hands,img=detector.findHands(img)
    cv2.imshow("image",img)
    #croping img part
    if hands:
        hand=hands[0]
        x,y,w,h=hand['bbox']
        #print("Bounding Box:", x, y, w, h)  # Add this line to check the bounding box coordinates
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
            # Resize imgResize to match imgWhite
            #imgResize = cv2.resize(imgResize, (imgSize, imgSize))
            #to center the image
            wGap=math.ceil((imgSize-wCal)/2)
            #to rezise and print
            #imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]] = imgResize
            imgWhite[:,wGap:wCal+wGap] = imgResize
        
         else:
            k=imgSize/w
            hCal=math.ceil(k*h)
            imgResize=cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape=imgResize.shape
            # Resize imgResize to match imgWhite
            #imgResize = cv2.resize(imgResize, (imgSize, imgSize))
            #to center the image
            hGap=math.ceil((imgSize-hCal)/2)
            #to rezise and print
            #imgWhite[0:imgResizeShape[0],0:imgResizeShape[1]] = imgResize
            imgWhite[hGap:hCal+hGap,:] = imgResize
                
        
         if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0: # Add this condition to check if the image size is valid
          #imgWhite[0:imgCropShape[0],0:imgCropShape[1]]=imgCrop  
          cv2.imshow("imageCrop", imgCrop)
          cv2.imshow("imageWhite", imgWhite)
          
          #print(imgCrop.shape)
         else:
           print("Invalid image size:", imgCrop.shape)  # Add this line to check the image size
        except:
          print("don't zoom too much")
        
    key=cv2.waitKey(1)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if key==ord('s'):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)
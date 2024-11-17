import cv2
import cv2 as cv
from PyQt5.QtCore import center
import os
import urllib
import urllib.request

recoger = cv.face.LBPHFaceRecognizer_create()

recoger.read("trainer/trainer.yml")
face_detect = cv.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

names = ['zxx']
warningtime = 0

def warning():
    pass

def facedetect(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # face_detect = cv.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    face = face_detect.detectMultiScale(gray,1.1,5,cv.CASCADE_SCALE_IMAGE,minSize=(50,50))


    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x + w,y + h),color=(0,0,255),thickness=2)
        cv.circle(img,center = (x + w // 2,y + h // 2),radius=w // 2,color=(0,255,0),thickness=1)

        ids,confidence = recoger.predict(gray[y:y + h,x:x + w])

        if confidence > 80:
            global warningtime
            warningtime += 1
            if warningtime >100:
                warning()
                warningtime = 0

            cv.putText(img,"unknow",(x + 10,y - 10),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),1)
        else:
            cv2.putText(img,str(names[ids]),(x + 10,y - 10),cv.FONT_HERSHEY_SIMPLEX,0.75,(0,255,0),1)

    cv.imshow("result",img)


cap = cv.VideoCapture(0)
while True:
    flag,flame = cap.read()
    if not flag:
        break
    facedetect(flame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
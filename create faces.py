import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('/home/revanth/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/revanth/Downloads/opencv-master/data/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
i=0
while(1):
    _, img =cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        gray_img = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img (press s to save ur screeshot or Esc to exit)',img)

    filename="/home/revanth/Desktop/face recognition/faces/rev/file_%i.jpg"%i
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k== ord('s'):
        cv2.imwrite(filename, gray_img)
        i+=1
cv2.destroyAllWindows()

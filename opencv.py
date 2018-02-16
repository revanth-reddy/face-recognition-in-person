import numpy as np
import cv2
import os
rootdir = '/home/revanth/Desktop/face recognition/faces'

for subdir, dirs, files in os.walk(rootdir):
    for file in files:



def mse(imageA, imageB):
    err = np.sum((imageA.astype("float")-imageB.astype("float"))**2)
    err /= float(imageA.shape[0]*imageA.shape[1])
    return err



face_cascade = cv2.CascadeClassifier('/home/revanth/Downloads/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/home/revanth/Downloads/opencv-master/data/haarcascades/haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
i=0
while(1):
    _, img =cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        rr = 10.0/roi_gray.shape[1]
        dimr = (10,int(roi_gray.shape[0]*rr))
        resizedimg = cv2.resize(roi_gray, dimr, interpolation = cv2.INTER_AREA)
        src=cv2.imread('/home/revanth/Desktop/face recognition/faces/rev.jpg',0)
        r = 10.0/src.shape[1]
        dim = (10,int(src.shape[0]*r))
        resizedsrc = cv2.resize(src, dim, interpolation = cv2.INTER_AREA)
        error=float(mse(resizedimg,resizedsrc))
        
        if error<95:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Rev',(x,y), font, 1,(255,255,255),2)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Unknown',(x,y), font, 1,(255,255,255),2)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img (press s to save ur screeshot or Esc to exit)',img)
    filename="/Desktop/face recognition/faces/file_%i.jpg"%i
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    if k== ord('s'):
        cv2.imwrite(filename, img)
        i+=1
cv2.destroyAllWindows()


# we need to keep in mind aspect ratio so the image does
# not look skewed or distorted -- therefore, we calculate
# the ratio of the new image to the old image
# r = 100.0 / image.shape[1]
# dim = (100, int(image.shape[0] * r))

# # perform the actual resizing of the image and show it
# resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("resized", resized)
cv2.waitKey(0)
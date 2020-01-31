import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml') 
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read ("trainner.yml")

labels = {"person_name":1}
with open("label.pickle", "rb") as f:
    og_labels = pickle.load(f) 
    labels = {v:k for k,v in og_labels.items()}
 
cap = cv2.VideoCapture(0)

def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

make_480p()



while (True):
    #Capture frame by frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5) 
    for (x,y,w,h) in faces:
        print(x,y,w,h)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 85:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            
        img_item = "my-image.png"
        cv2.imwrite(img_item,roi_color )

        #To draw rectangle
        color = (0, 191, 255) #BGR (0 - 255)
        stroke = 2
        width = x+w
        height = y+h
        cv2.rectangle(frame, (x , y), (width , height), color, stroke)
        #Detect Eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        #Detect Smile
        smile = smile_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in smile :
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)    

    #Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()    
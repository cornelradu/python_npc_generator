import cv2
import sys
import numpy as np
import math

repeat_blur = 5

imagePath = sys.argv[1]
face_cascade = "face.xml"
eye_cascade = "eye.xml"
nose_cascade = "nose.xml"
mouth_cascade = "mouth.xml"

cascades = {
	"face": cv2.CascadeClassifier(face_cascade),
	"eye": cv2.CascadeClassifier(eye_cascade),
        "nose": cv2.CascadeClassifier(nose_cascade),
        "mouth": cv2.CascadeClassifier(mouth_cascade)
}

objs_config = {
   "face": [1.1, 5, (30, 30)],
   "eye": [1.1, 3, (5, 5)],
   "nose": [1.1, 1, (2, 2)],
   "mouth": [1.1, 1, (2, 2)]
}

configs = ["scaleFactor", "minNeighbors", "minSize"]

def detect_objs(gray_img, obj_type = "face"):
    return cascades[obj_type].detectMultiScale(gray_img, **dict(zip(configs, objs_config[obj_type])))

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detect_objs(gray, "face")

for (x, y, w, h) in faces:
    
    face_copy = image[y:y+h, x:x+w]
    eyes = detect_objs(face_copy, obj_type="eye")[0:2]
  
    for (ex, ey , ew, eh) in eyes:
        if ey < np.size(face_copy, 1) / 2:
           cv2.circle(face_copy, (ex + ew / 2, ey + eh / 2), 17, (0, 0, 0), -1)     
    
    nose = detect_objs(face_copy, obj_type="nose")[0:1]
    for (ex, ey , ew, eh) in nose:
        cv2.line(face_copy, (ex, ey + eh * 3 / 4), (ex + ew/2, ey), (0,0,0), 10)
        cv2.line(face_copy, (ex, ey + eh * 3 / 4), (ex + ew, ey + eh * 6 / 10), (0,0,0), 10)
    
    mouth = detect_objs(face_copy, obj_type="mouth")[0:1]
    for (ex, ey , ew, eh) in mouth:
        cv2.line(face_copy, (ex, ey + eh / 2), (ex + ew, ey + eh / 2), (0,0,0), 18)

    to_be_blurred = image[y:y+h, x:x+w]
    
    while repeat_blur:
        to_be_blurred = cv2.bilateralFilter(to_be_blurred,30,75,75)
        repeat_blur -= 1
    
    circle_center = (y + h/2, x + w/2)
    radius = (w + h) / 4
    for y1 in range(y, y+h):
       for x1 in range(x, x+w):
          if math.sqrt( (x1 - circle_center[1]) ** 2 + (y1 - circle_center[0]) ** 2 ) < radius:
             image[y1][x1] = to_be_blurred[y1-y, x1-x]
     

cv2.imshow("Face", image)
cv2.waitKey(0)

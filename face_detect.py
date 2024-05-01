import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from rescale_utils import rescale_frames


img_fullsize = cv.imread('Photos/faces (3).jpg')
img = rescale_frames(img_fullsize)
cv.imshow('Cat', img)

# based on edges the face is detected.
# first turn to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# classifier reads XML code and stores it in variable haar_cascade
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# returns coordinates of the face as array
# minNeighbors= is the sensitivity for how noise-tolerant the detection is.
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

# we can loop through values:
for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow('Detected faces', img)

cv.waitKey(0)
cv.destroyAllWindows()

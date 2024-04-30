import cv2 as cv
from rescale_utils import rescale_frames
import matplotlib.pyplot as plt

img_full = cv.imread('Photos/Cats/Cat (16).jpg')
img = rescale_frames(img_full)
cv.imshow('Cat', img)

# convert from BGR to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# BGR to HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('HSV', hsv)

# BGR to L*a*b
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('LAB', lab)

# BGR to RGB
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imshow('RGB', rgb)


# HSV to BGR
hsv_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
cv.imshow('HSV --> BGR', hsv_bgr)

lab_gbr = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
cv.imshow('LAB --> BGR', hsv_bgr)


plt.imshow(rgb)
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()


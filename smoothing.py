import cv2 as cv
import numpy as np
from rescale_utils import rescale_frames

img_fullsize = cv.imread('Photos/Cats/Cat (16).jpg')
img = rescale_frames(img_fullsize)
cv.imshow('Cat', img)

# Averaging
average = cv.blur(img, (3, 3))
cv.imshow('Average Blur', average)

# Gaussian Blur - more natural than averaging
gauss = cv.GaussianBlur(img, (3, 3), 0)
cv.imshow('Gaussian Blur', gauss)

# Median Blur - almost same as average, but instead finds median
# Mostly used in CV projects to reduce noise
median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# Bilateral Blur - most effective in advanced CV - applies blur, but not on edges
bilateral = cv.bilateralFilter(img, 20, 20, 20)
cv.imshow('Bilateral', bilateral)

cv.waitKey()
cv.destroyAllWindows()
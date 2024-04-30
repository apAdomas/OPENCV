import cv2 as cv
import numpy as np

from rescale_utils import rescale_frames

img_fullsize = cv.imread('Photos/Cats/Cat (16).jpg')
img = rescale_frames(img_fullsize)
cv.imshow('Cat', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacian - computes gradients by computing absolute values of pixels and convert
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('Laplacian, lap', lap)

# Sobel - computes gradients in x and y directions
sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
# combining x and y dirs
combined_sobel = cv.bitwise_or(sobelx, sobely)

cv.imshow('Sobel X', sobelx)
cv.imshow('Sobel Y', sobely)
cv.imshow('Sobel X + Sobel Y', combined_sobel)

# combine with canny edge detection
canny = cv.Canny(gray, 150, 175)
cv.imshow('Canny', canny)

cv.waitKey(0)
cv.destroyAllWindows()

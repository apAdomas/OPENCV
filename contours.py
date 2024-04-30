import cv2 as cv
from rescale_utils import rescale_frames
import numpy as np

img = cv.imread('Photos/Cats/Cat (16).jpg')
rescaled = rescale_frames(img)
cv.imshow('Cats', rescaled)

# shape[:2] retrieves height and image width ignoring number of color channels.
blank = np.zeros(rescaled.shape, dtype='uint8')
cv.imshow('Blank', blank)

# 1: grayscale
gray = cv.cvtColor(rescaled, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# # 2: blur to reduce edges and get a more general edge structure and reduce the contours:
# blur = cv.GaussianBlur(gray, (5,5), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

# # 2: Cannny edge detector:
# canny = cv.Canny(rescaled, 125, 175)
# cv.imshow('Canny Edges', canny)

# binarizing image: if it is above threshold, value is set to max (black), if lower - to white
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)

# 3: find contours by using default method, returning contours and hierarchies
contours, hierarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
# Print a list of contours -> blurring reduces contours number
print(f'{len(contours)} contour(s) found')

# Draw contours on blank image:
cv.drawContours(blank, contours, -1, (0, 0, 255), 1)
cv.imshow('Contours Drawn', blank)

cv.waitKey(0)
cv.destroyAllWindows()

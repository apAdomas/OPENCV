import cv2 as cv
import numpy as np
from rescale_utils import rescale_frames

img_fullsize = cv.imread('Photos/Cats/Cat (16).jpg')
img = rescale_frames(img_fullsize)
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

b, g, r = cv.split(img)

blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])

cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

print(img.shape)
print(r.shape)
print(g.shape)
print(b.shape)

merged = cv.merge([b, g, r])
cv.imshow('Merged', merged)

cv.waitKey(0)
cv.destroyAllWindows()

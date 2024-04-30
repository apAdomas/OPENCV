import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from rescale_utils import rescale_frames

img_fullsize = cv.imread('Photos/Cats/Cat (16).jpg')
img = rescale_frames(img_fullsize)
cv.imshow('Cat', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

mask = cv.circle(blank, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
masked = cv.bitwise_and(img, img, mask=mask)
cv.imshow('Mask', mask)

# Histograms to determine pixel distribution density in the image
# Compute grayscale histogram for the image
# gray_histogram = cv.calcHist([gray], [0], mask, [256], [0, 256])

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# plt.plot(gray_histogram)
# plt.xlim([0, 256])
# plt.show()

# Colour Histogram

plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([img], [i], mask, [256], [0,256])
    plt.plot(hist, color=col)
    plt.xlim([0,256])

plt.show()


cv.waitKey(0)
cv.destroyAllWindows()

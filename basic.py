import cv2 as cv

img = cv.imread('Photos/Cats/Cat (7).jpg')
# cv.imshow('Cat', img)

# Convert to greyscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

# Blur
blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# cv.imshow('BLUR', blur)

# Edge cascade (edges present in the img)
# Passing the blur rather than image produces much less edges.
canny = cv.Canny(blur, 125, 175)
# cv.imshow('Canny Edges', canny)

# Dilating image
dilated = cv.dilate(canny, (7, 7), iterations=5)
# cv.imshow('Dilated', dilated)

# Eroding (getting back the edge cascade)
eroded = cv.erode(dilated, (3, 3), iterations=5)
# cv.imshow('Eroded', eroded)

# Resize (To scale use interpolation=cv.CUBIC, slower but high qual)
resize = cv.resize(img, (500, 500), interpolation=cv.INTER_AREA)
# cv.imshow('Resized', resize)

# Crop image
cropped = img[50:200, 200:400]
cv.imshow('Cropped', cropped)

cv.waitKey(0)

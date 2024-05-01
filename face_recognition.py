import numpy as np
import cv2 as cv
import os

# Load the face recognition model
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Try to load the trained model
model_path = 'face_trained.yml'
if os.path.exists(model_path):
    face_recognizer.read(model_path)
    print("Model loaded successfully")
else:
    print("Failed to load model")
    exit(1)  # Exit if the model is not loaded to avoid further errors

# Proceed only if the model is loaded successfully
haar_cascade = cv.CascadeClassifier('haar_face.xml')
people = ['Jay-Z', 'Jim Simons', 'Keanu Reeves', 'Quentin Tarantino', 'Warren Buffet']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

# Load an image to perform face recognition
img_path = r'C:\Users\pusly\PycharmProjects\OPENCV_\Photos\Face_Recognition\Jim Simons\simons (2).jpg'
img = cv.imread(img_path)
if img is None:
    print("Failed to load the image.")
    exit(1)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person', gray)

# Detect face in the image
faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h, x:x+w]
    label, confidence = face_recognizer.predict(faces_roi)
    print(f'label = {people[label]} with confidence of {confidence}')

    # Label the image with the name and confidence
    cv.putText(img, f'{people[label]} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow('Detected Face', img)
cv.waitKey(0)
cv.destroyAllWindows()

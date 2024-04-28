import cv2 as cv
import sys
from rescale_utils import rescale_frames, change_res


def process_image():
    # Load image from file
    img = cv.imread('Photos/Cats/Cat (2).jpg')

    if img is None:
        print("Error: Image not found")
    else:
        # call resize func and display
        resized_img = rescale_frames(img)
        cv.imshow('Resized Cat', resized_img)

        # wait for key to be pressed before closing
        cv.waitKey(0)
        cv.destroyAllWindows()


def process_video():
    # pass 0 as param to use camera, or file path
    capture = cv.VideoCapture('Videos/Dog.mp4')

    # Process
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break # break if no frame read or error

        # call resize func to resize frame
        resized_frame = rescale_frames(frame)

        # display frame
        cv.imshow('Video', resized_frame)

        # stop video
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()


def process_live_video():
    capture = cv.VideoCapture(0)

    width = 640
    height = 480

    change_res(capture, width, height)

    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break

        resized_frame = rescale_frames(frame)
        cv.imshow('Live Video', resized_frame)

        # stop when d is pressed
        if cv.waitKey(20) & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == 'image':
            process_image()
        elif sys.argv[1] == 'video':
            process_video()
        elif sys.argv[1] == 'live':
            process_live_video()
        else:
            print("Invalid argument. Use 'image', 'video', or 'live'.")
    else:
        print("No valid argument provided. Use 'image', 'video', or 'live'.")
import cv2 as cv
import sys


def process_image():
    # Load image from file
    img = cv.imread('Photos/Cats/Cat (2).jpg')

    if img is None:
        print("Error: Image not found")
    else:
        # get curr dim of the img
        height, width = img.shape[:2]

        # determine scaling factor
        max_dim = 700
        scale = min(max_dim/height, max_dim/width)

        # compute dim and resize img
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv.resize(img, (new_width, new_height))

        # display image after read
        cv.imshow('Resized Cat', resized_img)

        # wait for key to be pressed before closing
        cv.waitKey(0)

        cv.destroyAllWindows()

def process_video():
    # pass 0 as param to use camera, or file path
    capture = cv.VideoCapture('Videos/Dog.mp4')

    isTrue, frame = capture.read()
    if not isTrue:
        print("Error: Cannot read video")
        capture.release()
        return

    # Get dimensions for first frame
    height, width = frame.shape[:2]
    max_dim = 700
    scale = min(max_dim/height, max_dim/width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Process
    while True:
        isTrue, frame = capture.read()
        if not isTrue:
            break # break if no frame read or error

        resized_frame = cv.resize(frame, (new_width, new_height))
        # display frame
        cv.imshow('Video', resized_frame)

        # stop video
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
        else:
            print("Invalid argument. Use 'image' or 'video'.")
    else:
        print("No valid argument provided. Use 'image' or 'video'.")
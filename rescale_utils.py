import cv2 as cv


def rescale_frames(frame, max_dim=720):
    # Images, Videos, live video
    # get curr dim of the img
    height, width = frame.shape[:2]

    # determine scaling factor
    scale = min(max_dim / height, max_dim / width)

    # compute dim and resize img
    new_width = int(width * scale)
    new_height = int(height * scale)
    dimensions = (new_width, new_height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def change_res(capture, width, height):
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)


cv.waitKey(0)


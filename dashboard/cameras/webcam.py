import cv2
import os
camera = None


def start():
    global camera

    if camera is None or not camera.isOpened():
        camera_index = int(os.getenv("CAMERA_INDEX", "0"))

        camera = cv2.VideoCapture(camera_index)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return camera


def stop():
    global camera

    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()
        camera = None


def get_camera():
    return camera
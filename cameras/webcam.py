import cv2
import os

camera = None


def start():
    global camera

    if camera is not None and camera.isOpened():
        return camera

    camera_index = int(os.getenv("CAMERA_INDEX", "1"))

    # Use the default OpenCV backend
    camera = cv2.VideoCapture(camera_index)

    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not camera.isOpened():
        print(f"ERROR: Unable to open webcam at index {camera_index}")
        camera.release()
        camera = None
        return None

    print(f"Webcam successfully opened at index {camera_index}!")

    return camera


def stop():
    global camera

    if camera is not None:
        camera.release()
        camera = None
        print("Webcam released successfully")


def get_camera():
    return camera
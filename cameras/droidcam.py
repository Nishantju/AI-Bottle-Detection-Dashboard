import cv2
droidcam_url = None
camera = None
def set_droidcam_url(url):

    global droidcam_url

    droidcam_url = url


def start_camera(url=None):

    global camera


    if droidcam_url is None:
        raise Exception("DroidCam URL not provided")


    camera = cv2.VideoCapture(
        droidcam_url,
        cv2.CAP_FFMPEG
    )
    print("DroidCam URL:", droidcam_url)


    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )


    if not camera.isOpened():

        camera = None

        raise Exception(
            "Unable to connect to DroidCam"
        )


    return camera


def stop_camera():

    global camera

    if camera is not None:

        camera.release()

        camera = None



def read_frame():

    global camera

    if camera is None:
        return False, None

    return camera.read()
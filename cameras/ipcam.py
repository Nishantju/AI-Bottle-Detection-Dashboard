import cv2


camera = None
ipcam_url = None


def set_ipcam_url(url):

    global ipcam_url

    ipcam_url = url



def start_camera():

    global camera


    if ipcam_url is None:

        raise Exception(
            "IP Camera URL not provided"
        )


    camera = cv2.VideoCapture(
        ipcam_url
    )


    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )


    if not camera.isOpened():

        camera = None

        raise Exception(
            "Unable to connect to IP Camera"
        )


    return camera



def stop_camera():

    global camera


    if camera is not None:

        camera.release()

        camera = None
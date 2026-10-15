from .webcam import start as webcam_start
from .webcam import stop as webcam_stop
from .droidcam import start_camera as droid_start
from .droidcam import stop_camera as droid_stop
from .droidcam import set_droidcam_url
from .ipcam import start_camera as ip_start
from .ipcam import stop_camera as ip_stop
from .ipcam import set_ipcam_url
CURRENT_CAMERA = "webcam"


def set_camera_source(source):
    global CURRENT_CAMERA

    valid_sources = (
        "webcam",
        "mobile",
        "droidcam",
        "ipcamera"
    )

    if source not in valid_sources:
        raise ValueError(f"Unsupported camera source: {source}")

    CURRENT_CAMERA = source


def get_camera_source():
    return CURRENT_CAMERA


def start_camera():

    if CURRENT_CAMERA == "webcam":
        return webcam_start()

    elif CURRENT_CAMERA == "droidcam":
        return droid_start()

    elif CURRENT_CAMERA == "mobile":
        raise NotImplementedError("Mobile camera not implemented yet")

    elif CURRENT_CAMERA == "ipcamera":
        return ip_start()

    raise ValueError("Unknown camera source")


def stop_camera():

    if CURRENT_CAMERA == "webcam":
        webcam_stop()
        return


    elif CURRENT_CAMERA == "droidcam":
        droid_stop()
        return

    elif CURRENT_CAMERA == "ipcamera":
        ip_stop()
        return
        
from .webcam import start as webcam_start
from .webcam import stop as webcam_stop

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
        raise NotImplementedError("DroidCam not implemented yet")

    elif CURRENT_CAMERA == "mobile":
        raise NotImplementedError("Mobile camera not implemented yet")

    elif CURRENT_CAMERA == "ipcamera":
        raise NotImplementedError("IP Camera not implemented yet")

    raise ValueError("Unknown camera source")


def stop_camera():

    if CURRENT_CAMERA == "webcam":
        webcam_stop()
        return
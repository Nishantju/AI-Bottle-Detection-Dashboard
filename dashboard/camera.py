from pathlib import Path
import sys
import time
import base64
from cameras.manager import start_camera as manager_start
from cameras.manager import stop_camera as manager_stop

import threading


# Allow importing from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from dashboard.database.database import save_detection

import os
import cv2
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
frame_counter = 0
last_predictions = []

mobile_predictions = []

latest_status = {
    "status": "No Bottle",
    "confidence": 0,
    "count": 0
}

mobile_status = {
    "status": "No Bottle",
    "confidence": 0,
    "count": 0
}

latest_frame = None
frame_lock = threading.Lock()
prediction_lock = threading.Lock()
worker_running = False
last_saved_status = None

camera = None



def start_camera():
    global worker_running
    global camera

    camera = manager_start()

    worker_running = True

    threading.Thread(
        target=inference_worker,
        daemon=True
    ).start()

def stop_camera():
    global worker_running
    global camera

    worker_running = False

    manager_stop()

    camera = None


client = InferenceHTTPClient(
    api_url=os.getenv("ROBOFLOW_API_URL"),
    api_key=os.getenv("ROBOFLOW_API_KEY")
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)


def inference_worker():
    global latest_frame
    global last_predictions
    global latest_status
    global last_saved_status
    global worker_running
    global frame_counter

    while worker_running:

        with frame_lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue


        frame_counter += 1
        if frame_counter % 5 != 0:
            continue


        small_frame = cv2.resize(
            frame,
            (640,480)
        )

        cv2.imwrite(
            "current_frame.jpg",
            small_frame
        )

        start = time.time()

        result = client.run_workflow(
            workspace_name="nishant-kumar-sdkzt",
            workflow_id=(
                "ai-bottle-filling-detection-vai-bottle-filling-detection-3-"
                "rfdetr-nas-t1-14fdc8-logic"
            ),
            images={
                "image": "current_frame.jpg"
            },
            use_cache=True
        )

        print(f"workflow took {time.time()-start:.2f} seconds")

        if result and isinstance(result, list):

            output = result[0]

            prediction_data = output.get("predictions", {})

            predictions = prediction_data.get("predictions", [])

            with prediction_lock:
                last_predictions = [
                    pred
                    for pred in predictions
                    if pred.get("confidence", 0) > 0.70
                    and pred.get("class", "").lower() == "bottles"
                ]
            print("filtered box data")
            print(last_predictions)

        else:
            last_predictions = []

        latest_status["count"] = len(last_predictions)

        if last_predictions:

            latest_status["status"] = "Bottle Detected"

            latest_status["confidence"] = round(
                last_predictions[0]["confidence"] * 100,
                2
            )

        else:

            latest_status["status"] = "No Bottle"
            latest_status["confidence"] = 0

        if latest_status["status"] != last_saved_status:

            save_detection(
                latest_status["status"],
                latest_status["confidence"],
                latest_status["count"]
            )

            last_saved_status = latest_status["status"]
        time.sleep(0.1)



def generate_frames():
    global camera, frame_counter, last_predictions
    global last_saved_status, latest_frame
    while True:

        if camera is None:
            break

        success, frame = camera.read()

        if not success:
            break

        with frame_lock:
            latest_frame = frame.copy()

        frame_counter += 1


        with prediction_lock:
            predictions = last_predictions.copy()
        latest_status["count"] = len(predictions)

        if len(predictions) > 0:
            latest_status["status"] = "Bottle Detected"
            latest_status["confidence"] = round(
                predictions[0]["confidence"] * 100, 2
            )
        else:
            latest_status["status"] = "No Bottle"
            latest_status["confidence"] = 0
        print("Predictions:", len(predictions))
        print(latest_status)

        if latest_status["status"] != last_saved_status:

            save_detection(
                latest_status["status"],
                latest_status["confidence"],
                latest_status["count"]
            )

            last_saved_status = latest_status["status"]


        # DRAW BOX EVERY FRAME
        for pred in predictions:

            x = int(pred.get("x", 0))
            y = int(pred.get("y", 0))

            w = int(pred.get("width", 0))
            h = int(pred.get("height", 0))

            confidence = pred.get("confidence", 0)

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)

            x2 = int(x + w / 2)
            y2 = int(y + h / 2)


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0,255,0),
                3
            )

            cv2.putText(
                frame,
                f"Bottle {confidence:.2f}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame +
            b"\r\n"
        )

def process_mobile_image(image_path):

    global mobile_predictions
    global mobile_status

    frame = cv2.imread(image_path)


    frame = cv2.resize(
        frame,
        (640,480)
    )


    cv2.imwrite(
        image_path,
        frame
    )


    confidence = 0


    result = client.run_workflow(
        workspace_name="nishant-kumar-sdkzt",
        workflow_id=(
            "ai-bottle-filling-detection-vai-bottle-filling-detection-3-"
            "rfdetr-nas-t1-14fdc8-logic"
        ),
        images={
            "image": image_path
        },
        use_cache=False
    )


    predictions = []


    if result and isinstance(result, list):

        output = result[0]

        prediction_data = output.get("predictions", {})

        predictions = prediction_data.get("predictions", [])


    # update mobile predictions ONLY
    mobile_predictions = [
        p for p in predictions
        if p.get("confidence", 0) > 0.70
        and p.get("class", "").lower() == "bottles"
    ]


    # draw boxes
    for pred in mobile_predictions:

        x = int(pred.get("x",0))
        y = int(pred.get("y",0))

        w = int(pred.get("width", pred.get("w",0)))
        h = int(pred.get("height", pred.get("h",0)))

        confidence = pred.get("confidence",0)


        x1 = int(x - w/2)
        y1 = int(y - h/2)

        x2 = int(x + w/2)
        y2 = int(y + h/2)


        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            3
        )


        cv2.putText(
            frame,
            f"Bottle {confidence:.2f}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


    # update MOBILE status only

    mobile_status["count"] = len(mobile_predictions)


    if mobile_predictions:

        mobile_status["status"] = "Bottle Detected"

        mobile_status["confidence"] = round(
            mobile_predictions[0]["confidence"] * 100,
            2
        )

    else:

        mobile_status["status"] = "No Bottle"

        mobile_status["confidence"] = 0



    _, buffer = cv2.imencode(".jpg", frame)

    frame_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")



    return {

        "image": frame_base64,

        "predictions": mobile_predictions,

        "status": mobile_status["status"],

        "confidence": mobile_status["confidence"],

        "count": mobile_status["count"]

    }
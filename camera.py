from pathlib import Path
import sys
import time
import base64
import threading
import os

import cv2

from cameras.manager import start_camera as manager_start
from cameras.manager import stop_camera as manager_stop

# Allow importing from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from database.services import save_detection
from roboflow_inference import predict_image


# --------------------------------------------------
# GLOBAL VARIABLES
# --------------------------------------------------

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

last_processed_frame = 0


# --------------------------------------------------
# HELPER: EXTRACT ROBOFLOW PREDICTIONS
# --------------------------------------------------

def extract_predictions(result):

    if not result or not isinstance(result, list):
        return []

    try:
        predictions = result[0]["predictions"]["predictions"]
        return predictions

    except (KeyError, IndexError, TypeError):
        print("Unexpected Roboflow response:", result)
        return []


# --------------------------------------------------
# HELPER: FILTER BOTTLE PREDICTIONS
# --------------------------------------------------

def filter_bottles(predictions, confidence_threshold):

    return [
        p for p in predictions
        if p.get("confidence", 0) > confidence_threshold
        and p.get("class", "").lower() == "bottles"
    ]


# --------------------------------------------------
# HELPER: RESIZE FRAME WHILE PRESERVING ASPECT RATIO
# --------------------------------------------------

def resize_frame(frame, max_size=640):

    height, width = frame.shape[:2]

    scale = min(1.0, max_size / max(height, width))

    if scale < 1.0:

        new_width = int(width * scale)
        new_height = int(height * scale)

        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    else:

        resized = frame.copy()

    return resized, scale


# --------------------------------------------------
# START CAMERA
# --------------------------------------------------

def start_camera():

    global worker_running
    global camera
    global last_processed_frame

    if worker_running:
        return camera

    camera = manager_start()

    if camera is None:
        print("Unable to start camera")
        return None

    worker_running = True
    last_processed_frame = frame_counter

    threading.Thread(
        target=inference_worker,
        daemon=True
    ).start()

    print("Roboflow camera inference worker started")

    return camera


# --------------------------------------------------
# STOP CAMERA
# --------------------------------------------------

def stop_camera():

    global worker_running
    global camera

    worker_running = False

    manager_stop()

    camera = None

    print("Camera stopped")


# --------------------------------------------------
# ROBOFLOW LIVE CAMERA INFERENCE WORKER
# --------------------------------------------------

def inference_worker():

    global latest_frame
    global last_predictions
    global latest_status
    global last_saved_status
    global worker_running
    global last_processed_frame

    while worker_running:

        with frame_lock:

            if latest_frame is None:
                frame = None
                current_frame_counter = frame_counter

            else:
                frame = latest_frame.copy()
                current_frame_counter = frame_counter

        if frame is None:
            time.sleep(0.05)
            continue

        # Process only after 10 new camera frames
        if current_frame_counter - last_processed_frame < 10:
            time.sleep(0.02)
            continue

        last_processed_frame = current_frame_counter

        try:

            workflow_start = time.time()

            original_height, original_width = frame.shape[:2]

            # Resize frame for cloud inference
            small_frame, scale = resize_frame(
                frame,
                max_size=640
            )

            temp_image_path = "current_frame.jpg"

            cv2.imwrite(
                temp_image_path,
                small_frame
            )

            # Roboflow inference
            result = predict_image(temp_image_path)

            inference_time = time.time() - workflow_start

            print(
                f"Roboflow inference time: {inference_time:.3f} seconds"
            )

            predictions = extract_predictions(result)

            filtered_predictions = filter_bottles(
                predictions,
                confidence_threshold=0.40
            )

            # Convert prediction coordinates back to original frame size
            processed_predictions = []

            for pred in filtered_predictions:

                x = float(pred["x"])
                y = float(pred["y"])

                width = float(pred["width"])
                height = float(pred["height"])

                if scale > 0:

                    x /= scale
                    y /= scale

                    width /= scale
                    height /= scale

                processed_predictions.append({

                    "x": int(x),
                    "y": int(y),

                    "width": int(width),
                    "height": int(height),

                    "confidence": float(pred["confidence"]),

                    "class": "bottles"

                })

                print(
                    "Bottle detected:",
                    round(float(pred["confidence"]) * 100, 2),
                    "%"
                )

            # Update predictions safely
            with prediction_lock:
                last_predictions = processed_predictions.copy()

            # Update live detection status
            latest_status["count"] = len(processed_predictions)

            if processed_predictions:

                latest_status["status"] = "Bottle Detected"

                latest_status["confidence"] = round(
                    processed_predictions[0]["confidence"] * 100,
                    2
                )

            else:

                latest_status["status"] = "No Bottle"
                latest_status["confidence"] = 0

            # Save detection history when status changes
            if latest_status["status"] != last_saved_status:

                save_detection(
                    latest_status["status"],
                    latest_status["confidence"],
                    latest_status["count"]
                )

                last_saved_status = latest_status["status"]

        except Exception as e:

            print("Roboflow camera inference error:", str(e))

        time.sleep(0.02)


# --------------------------------------------------
# GENERATE LIVE CAMERA FRAMES
# --------------------------------------------------

def generate_frames():

    global camera
    global frame_counter
    global latest_frame

    while True:

        if camera is None:
            break

        success, frame = camera.read()

        if not success:
            time.sleep(0.02)
            continue

        with frame_lock:

            latest_frame = frame.copy()
            frame_counter += 1

        # Read latest predictions
        with prediction_lock:

            predictions = last_predictions.copy()

        # Draw bounding boxes
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
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Bottle {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        # Encode frame for Flask MJPEG streaming
        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# --------------------------------------------------
# MOBILE IMAGE DETECTION
# --------------------------------------------------

def process_mobile_image(image_path):

    global mobile_predictions
    global mobile_status

    frame = cv2.imread(image_path)

    if frame is None:
        raise ValueError("Unable to read mobile image")

    # Preserve aspect ratio and resize
    frame, scale = resize_frame(
        frame,
        max_size=640
    )

    # Save resized image for Roboflow
    cv2.imwrite(
        image_path,
        frame
    )

    try:

        workflow_start = time.time()

        # Roboflow inference
        result = predict_image(image_path)

        print(
            f"Mobile Roboflow inference time: "
            f"{time.time() - workflow_start:.3f} seconds"
        )

        predictions = extract_predictions(result)

        # Filter mobile predictions
        filtered = filter_bottles(
            predictions,
            confidence_threshold=0.70
        )

        mobile_predictions = filtered

    except Exception as e:

        print("Mobile Roboflow inference error:", str(e))

        mobile_predictions = []

    # Draw bounding boxes
    for pred in mobile_predictions:

        x = int(pred.get("x", 0))
        y = int(pred.get("y", 0))

        w = int(pred.get("width", pred.get("w", 0)))
        h = int(pred.get("height", pred.get("h", 0)))

        confidence = pred.get("confidence", 0)

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)

        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            f"Bottle {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # Update mobile status only
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

    # Convert processed image to Base64
    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        raise ValueError("Unable to encode mobile detection image")

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
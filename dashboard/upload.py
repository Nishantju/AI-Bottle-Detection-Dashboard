import os
import time
import cv2

from ultralytics import YOLO

from camera import latest_status


model = YOLO("models/best.pt")

def process_image(image_path):
    start_total = time.time()
    workflow_start = time.time()



    image = cv2.imread(image_path)

    height, width = image.shape[:2]

    max_size = 640

    if max(height, width) > max_size:

        scale = max_size / max(height, width)

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

        cv2.imwrite(image_path, image)







    result = []

    print(f"Inference: {time.time() - workflow_start:.2f} sec")

    image = cv2.imread(image_path)
    print(f"Read image: {time.time() - start_total:.2f} sec")

    predictions = []

    results = model(image)

    for r in results:
        for box in r.boxes:

            confidence = float(box.conf[0])

            if confidence > 0.25:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                predictions.append({
                    "x": (x1 + x2) // 2,
                    "y": (y1 + y2) // 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "confidence": confidence
                })
    if len(predictions) > 0:
    
        latest_status["status"] = "Bottle Detected"
        latest_status["confidence"] = int(
            predictions[0]["confidence"] * 100
        )
        latest_status["count"] = len(predictions)

    else:

        latest_status["status"] = "No Bottle"
        latest_status["confidence"] = 0
        latest_status["count"] = 0

    for pred in predictions:

        x = int(pred["x"])
        y = int(pred["y"])

        w = int(pred["width"])
        h = int(pred["height"])

        confidence = pred["confidence"]

        cv2.rectangle(
            image,
            (x - w//2, y - h//2),
            (x + w//2, y + h//2),
            (0,255,0),
            3
        )

        cv2.putText(
            image,
            f"{confidence:.2f}",
            (x-w//2,y-h//2-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    os.makedirs(
        "static/results",
        exist_ok=True
    )

    output_path = os.path.join(
        "static",
        "results",
        "result.jpg"
    )
    

    cv2.imwrite(
        output_path,
        image
    )

    return output_path
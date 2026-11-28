import os
import time
import cv2

from roboflow_inference import predict_image
from camera import latest_status


def process_image(image_path):

    start_time = time.time()

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Unable to read uploaded image")

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

    # Roboflow inference
    result = predict_image(image_path)

    predictions = result[0]["predictions"]["predictions"]

    filtered_predictions = [
        p for p in predictions
        if p.get("confidence", 0) > 0.25
        and p.get("class", "").lower() == "bottles"
    ]

    # Update detection status
    latest_status["count"] = len(filtered_predictions)

    if filtered_predictions:

        latest_status["status"] = "Bottle Detected"

        latest_status["confidence"] = round(
            filtered_predictions[0]["confidence"] * 100,
            2
        )

    else:

        latest_status["status"] = "No Bottle"
        latest_status["confidence"] = 0

    # Draw bounding boxes
    for pred in filtered_predictions:

        x = int(pred["x"])
        y = int(pred["y"])

        w = int(pred["width"])
        h = int(pred["height"])

        confidence = pred["confidence"]

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)

        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        cv2.putText(
            image,
            f"Bottle {confidence:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    os.makedirs("static/results", exist_ok=True)

    output_path = os.path.join(
        "static",
        "results",
        "result.jpg"
    )

    cv2.imwrite(output_path, image)

    print(
        f"Roboflow upload inference completed in "
        f"{time.time() - start_time:.2f} seconds"
    )

    return output_path
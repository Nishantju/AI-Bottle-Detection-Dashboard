import os
import time
import cv2
from camera import latest_status
from inference_sdk import (
    InferenceHTTPClient,
    InferenceConfiguration
)

client = InferenceHTTPClient(
    api_url=os.getenv("ROBOFLOW_API_URL"),
    api_key=os.getenv("ROBOFLOW_API_KEY")
).configure(
    InferenceConfiguration(
        api_key_transport="header"
    )
)

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








    result = client.run_workflow(
        workspace_name="nishant-kumar-sdkzt",
        workflow_id=(
            "ai-bottle-filling-detection-vai-bottle-filling-detection-3-"
            "rfdetr-nas-t1-14fdc8-logic"
        ),
        images={
            "image": image_path
        },
        use_cache=True
    )
    print(f"Roboflow workflow: {time.time() - workflow_start:.2f} sec")

    image = cv2.imread(image_path)
    print(f"Read image: {time.time() - start_total:.2f} sec")

    predictions = []

    if result and isinstance(result, list):

        output = result[0]

        prediction_data = output.get("predictions", {})

        predictions = prediction_data.get(
            "predictions",
            []
        )
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
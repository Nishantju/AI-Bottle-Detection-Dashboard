
import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()


from flask import (
    Flask,
    render_template,
    Response,
    stream_with_context,
    jsonify,
    request,
    url_for
)

import base64

import os

from camera import (
    generate_frames,
    latest_status,
    start_camera,
    stop_camera,
    process_mobile_image
)

from camera import mobile_status

from cameras.manager import (
    set_camera_source,
    get_camera_source
)


from upload import process_image

from dashboard.database.database import (
    get_detections,
    get_statistics,
    get_chart_data
)



app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static/results", exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():

    def stream():
        try:
            yield from generate_frames()
        finally:
            stop_camera()
            print("Camera stopped")

    return Response(
        stream_with_context(stream()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )




@app.route("/status")
def status():

    if get_camera_source() == "mobile":
        return jsonify(mobile_status)

    return jsonify(latest_status)


@app.route("/camera/source")
def camera_source():

    return {
        "camera": get_camera_source()
    }


@app.route("/camera/source/<source>")
def change_camera(source):

    try:

        set_camera_source(source)

        return {
            "success": True,
            "camera": get_camera_source()
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }, 400






@app.route("/history")
def history():
    detections = get_detections()

    return render_template(
        "history.html",
        detections=detections
    )

@app.route("/analytics")
def analytics():

    stats = get_statistics()
    chart_data = get_chart_data()

    return render_template(
        "analytics.html",
        stats=stats,
        chart_data=chart_data
    )

@app.route("/test")
def test():
    return "TEST OK"

@app.route("/start_camera")
def start_camera_route():
    start_camera()
    return "Camera Started"


@app.route("/stop_camera")
def stop_camera_route():
    stop_camera()
    return "Camera Stopped"

@app.route("/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return {"success": False, "message": "No file selected"}

    file = request.files["file"]

    if file.filename == "":
        return {"success": False, "message": "No file selected"}

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    filename = file.filename.lower()

    image_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    )


    if filename.endswith(image_extensions):

        result_path = process_image(filepath)


    else:

        return {
            "success": False,
            "message": "Unsupported file type."
        }

    return {
        "success": True,
        "file": "/" + result_path.replace("\\", "/"),
    }

@app.route("/mobile_frame", methods=["POST"])
def mobile_frame():

    data = request.get_json()

    if not data:
        return {
            "success": False,
            "message": "No JSON received"
        }, 400

    image = data.get("image")

    if not image:
        return {
            "success": False,
            "message": "No image received"
        }, 400


    image = image.split(",")[1]

    image_bytes = base64.b64decode(image)

    image_path = "mobile_frame.jpg"

    with open(image_path, "wb") as f:
        f.write(image_bytes)


    result = process_mobile_image(image_path)



    return {
        "success": True,
        "image": result["image"],
        "predictions": result["predictions"],
        "status": result["status"],
        "confidence": result["confidence"],
        "count": result["count"]
    }


if __name__ == "__main__":
    app.run(
    host="0.0.0.0",
    port=5000,
    debug=True,
    use_reloader=False
)
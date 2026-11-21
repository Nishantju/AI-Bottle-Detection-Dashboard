from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Train the model
    model.train(
        data="dataset.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,           # Use RTX 3050
        workers=2,          # Good starting value for Windows
        project="runs",
        name="bottle_detector",
        exist_ok=True
    )

if __name__ == "__main__":
    main()
from ultralytics import YOLO

# Load your trained model
model = YOLO("runs/detect/runs/bottle_detector/weights/best.pt")

# Run prediction on an image
results = model.predict(
    source="limca.jpeg",   # Replace with your test image
    conf=0.25,
    save=True
)

print("Prediction completed!")




import cv2
import os

# -----------------------------
# Change this folder as needed
# -----------------------------
SAVE_FOLDER = "dataset_capture"

os.makedirs(SAVE_FOLDER, exist_ok=True)

camera = cv2.VideoCapture(0)

image_count = 1

print("=" * 50)
print("Dataset Capture Tool")
print("=" * 50)
print("Press S -> Save Image")
print("Press Q -> Quit")
print("=" * 50)

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read camera.")
        break

    cv2.imshow("Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save Image
    if key == ord("s"):

        filename = os.path.join(
            SAVE_FOLDER,
            f"image_{image_count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print(f"Saved: {filename}")

        image_count += 1

    # Quit
    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()

print("Capture Finished.")
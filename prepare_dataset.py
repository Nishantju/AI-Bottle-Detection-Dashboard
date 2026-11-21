import random
import shutil
from pathlib import Path


# Source folder containing images and labels
SOURCE_DIR = Path("dataset/bottle_datasets")

# Output folders
TRAIN_IMAGES = Path("dataset/images/train")
VAL_IMAGES = Path("dataset/images/val")

TRAIN_LABELS = Path("dataset/labels/train")
VAL_LABELS = Path("dataset/labels/val")

# Get all unique images
images = []

for ext in ("*.png", "*.jpeg", "*.jpg"):
    images.extend(SOURCE_DIR.glob(ext))

# Remove duplicates
images = list(set(images))

print(f"Total images found: {len(images)}")

random.shuffle(images)

# Calculate 80% training images
split_index = int(len(images) * 0.8)

# Split the list
train_images = images[:split_index]
val_images = images[split_index:]


# Create output folders if they don't exist
TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
VAL_IMAGES.mkdir(parents=True, exist_ok=True)
TRAIN_LABELS.mkdir(parents=True, exist_ok=True)
VAL_LABELS.mkdir(parents=True, exist_ok=True)


# Copy training images and labels
for image_path in train_images:

    label_path = image_path.with_suffix(".txt")

    shutil.copy(image_path, TRAIN_IMAGES / image_path.name)

    if label_path.exists():
        shutil.copy(label_path, TRAIN_LABELS / label_path.name)


# Copy validation images and labels
for image_path in val_images:

    label_path = image_path.with_suffix(".txt")

    shutil.copy(image_path, VAL_IMAGES / image_path.name)

    if label_path.exists():
        shutil.copy(label_path, VAL_LABELS / label_path.name)


print("Dataset prepared successfully!")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")

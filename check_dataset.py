from pathlib import Path

for split in ["train", "val"]:
    image_dir = Path(f"dataset/images/{split}")
    label_dir = Path(f"dataset/labels/{split}")

    print(f"\nChecking {split}...")

    missing = 0

    for img in image_dir.iterdir():
        if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        label = label_dir / (img.stem + ".txt")

        if not label.exists():
            print("Missing label:", img.name)
            missing += 1

    print(f"Missing labels: {missing}")
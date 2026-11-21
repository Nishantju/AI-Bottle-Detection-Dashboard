import xml.etree.ElementTree as ET
from pathlib import Path

# Folder containing images and XML annotations
SOURCE_DIR = Path("dataset/bottle_datasets")

# Class names (must match your LabelImg labels)
CLASSES = ["bottle"]

for xml_file in SOURCE_DIR.glob("*.xml"):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    yolo_lines = []

    for obj in root.findall("object"):

        class_name = obj.find("name").text

        if class_name not in CLASSES:
            continue

        class_id = CLASSES.index(class_name)

        box = obj.find("bndbox")

        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    txt_file = xml_file.with_suffix(".txt")

    with open(txt_file, "w") as f:
        f.write("\n".join(yolo_lines))



print("All XML files converted to YOLO TXT successfully!")
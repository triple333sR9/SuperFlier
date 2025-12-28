"""Rename images to sequential IDs and create CSV with labels."""

import os
import csv
import config


def get_jpg_files(folder):
    return sorted([f for f in folder.iterdir() if f.suffix.lower() == '.jpg'])


def rename_images_and_create_csv():
    if not config.FLY_FOLDER.exists() or not config.NOFLY_FOLDER.exists():
        raise FileNotFoundError("Fly or NoFly folder not found")

    fly_images = get_jpg_files(config.FLY_FOLDER)
    nofly_images = get_jpg_files(config.NOFLY_FOLDER)
    records = []
    current_id = 1

    for img_path in fly_images:
        new_name = f"{current_id:05d}.jpg"
        os.rename(img_path, img_path.parent / new_name)
        records.append({"id": current_id, "filename": new_name, "folder": "fly", "label": 1})
        current_id += 1

    for img_path in nofly_images:
        new_name = f"{current_id:05d}.jpg"
        os.rename(img_path, img_path.parent / new_name)
        records.append({"id": current_id, "filename": new_name, "folder": "nofly", "label": 0})
        current_id += 1

    with open(config.CSV_FILENAME, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "filename", "folder", "label"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Renamed {len(records)} images (Fly: {len(fly_images)}, NoFly: {len(nofly_images)})")
    print(f"CSV saved to: {config.CSV_FILENAME}")


if __name__ == "__main__":
    rename_images_and_create_csv()
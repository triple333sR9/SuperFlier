"""Prepare sample images for visual evaluation by moving them to test folder."""

import shutil
import random
from pathlib import Path
import config

def get_jpg_files(folder):
    return sorted([f for f in folder.iterdir() if f.suffix.lower() == '.jpg'])


def prepare_samples():
    config.TEST_FOLDER.mkdir(parents=True, exist_ok=True)

    fly_images = get_jpg_files(config.FLY_FOLDER)
    nofly_images = get_jpg_files(config.NOFLY_FOLDER)

    if len(fly_images) < 5 or len(nofly_images) < 5:
        raise ValueError(
            f"Need at least 5 images in each folder. Found: {len(fly_images)} fly, {len(nofly_images)} nofly")

    random.seed(42)
    selected_fly = random.sample(fly_images, 5)
    selected_nofly = random.sample(nofly_images, 5)

    fly_ids, nofly_ids = [], []

    for img_path in selected_fly:
        img_id = int(img_path.stem)
        fly_ids.append(img_id)
        shutil.move(str(img_path), str(config.TEST_FOLDER / img_path.name))

    for img_path in selected_nofly:
        img_id = int(img_path.stem)
        nofly_ids.append(img_id)
        shutil.move(str(img_path), str(config.TEST_FOLDER / img_path.name))

    print(f"Moved 10 sample images to {config.TEST_FOLDER}")
    print(f"\nUpdate config.py with these IDs:")
    print(f"SAMPLE_FLY_IDS = {sorted(fly_ids)}")
    print(f"SAMPLE_NOFLY_IDS = {sorted(nofly_ids)}")


if __name__ == "__main__":
    prepare_samples()
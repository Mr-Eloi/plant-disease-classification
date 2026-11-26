
from pathlib import Path
import shutil
import random
from collections import Counter

from sklearn.model_selection import train_test_split

# Project directories
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

EXTRACTED_DIR = DATA_DIR / "tfds_cache" / "downloads" / "extracted"

RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

# Reproducible dataset split
RANDOM_STATE = 42

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def find_dataset():
    """Locate the extracted PlantVillage dataset."""

    matches = list(
        EXTRACTED_DIR.rglob(
            "Plant_leave_diseases_dataset_without_augmentation"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Could not locate the extracted PlantVillage dataset."
        )

    return matches[0]


def prepare_raw_dataset(source_dir):
    """Copy tomato images into data/raw, organized by class."""

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = [
        folder for folder in source_dir.iterdir()
        if folder.is_dir() and folder.name.startswith("Tomato___")
    ]

    if not class_dirs:
        raise ValueError("No tomato disease categories were found.")

    total_images = 0

    for class_dir in class_dirs:
        destination = RAW_DIR / class_dir.name
        destination.mkdir(parents=True, exist_ok=True)

        images = [
            file for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image in images:
            target = destination / image.name

            if not target.exists():
                shutil.copy2(image, target)

        count = len(images)
        total_images += count

        print(f"{class_dir.name}: {count} images")

    print(f"\nTotal tomato images: {total_images}")


def split_dataset():
    """Create stratified 70/15/15 train-validation-test splits."""

    # Avoid accidentally duplicating an existing split
    if TRAIN_DIR.exists() or VAL_DIR.exists() or TEST_DIR.exists():
        raise FileExistsError(
            "A dataset split already exists. "
            "Move or remove data/train, data/validation and data/test "
            "before generating a new split."
        )

    records = []

    for class_dir in RAW_DIR.iterdir():
        if not class_dir.is_dir():
            continue

        for image in class_dir.iterdir():
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                records.append((image, class_dir.name))

    if not records:
        raise ValueError("No images were found in data/raw.")

    paths = [str(record[0]) for record in records]
    labels = [record[1] for record in records]

    # First split: 70% training, 30% remaining
    train_paths, remaining_paths, train_labels, remaining_labels = (
        train_test_split(
            paths,
            labels,
            test_size=0.30,
            random_state=RANDOM_STATE,
            stratify=labels
        )
    )

    # Second split: divide remaining 30% equally
    val_paths, test_paths, val_labels, test_labels = (
        train_test_split(
            remaining_paths,
            remaining_labels,
            test_size=0.50,
            random_state=RANDOM_STATE,
            stratify=remaining_labels
        )
    )

    splits = {
        TRAIN_DIR: train_paths,
        VAL_DIR: val_paths,
        TEST_DIR: test_paths
    }

    for split_dir, image_paths in splits.items():
        for image_path in image_paths:
            source = Path(image_path)
            destination = split_dir / source.parent.name
            destination.mkdir(parents=True, exist_ok=True)

            shutil.copy2(source, destination / source.name)

    print("\nDataset split completed.")

    for split_name, split_labels in [
        ("Training", train_labels),
        ("Validation", val_labels),
        ("Testing", test_labels)
    ]:
        print(f"\n{split_name}: {len(split_labels)} images")
        for class_name, count in sorted(Counter(split_labels).items()):
            print(f"  {class_name}: {count}")


def acquire_dataset():
    print("Locating extracted PlantVillage dataset...")

    source_dir = find_dataset()

    print(f"Dataset found: {source_dir}")

    print("\nPreparing tomato disease images...")
    prepare_raw_dataset(source_dir)

    print("\nCreating training, validation and test splits...")
    split_dataset()

    print("\nPhase 1 dataset preparation completed successfully.")


if __name__ == "__main__":
    acquire_dataset()

import json
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

INCOMING_DIR = DATA_DIR / "retraining" / "incoming"
PROCESSED_DIR = DATA_DIR / "retraining" / "processed"
EXPERIMENTS_DIR = DATA_DIR / "retraining" / "experiments"

MANIFEST_PATH = INCOMING_DIR / "manifest.json"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

PRODUCTION_MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
MAX_ORIGINAL_IMAGES_PER_CLASS = 300
RETRAINING_EPOCHS = 3
RETRAINING_LEARNING_RATE = 0.00001

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------
# Image upload and validation
# --------------------------------------------------

def get_class_names():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_image(image_file):
    extension = Path(image_file.filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Only JPG, JPEG and PNG images are supported.")

    try:
        image_file.stream.seek(0)

        with Image.open(image_file.stream) as image:
            image = image.convert("RGB")
            image.load()

        image_file.stream.seek(0)
        return image

    except (UnidentifiedImageError, OSError):
        raise ValueError("The uploaded file is not a valid image.")


def save_labeled_image(image_file, disease_label):
    class_names = get_class_names()

    if disease_label not in class_names:
        raise ValueError("Invalid disease label.")

    image = validate_image(image_file)

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = INCOMING_DIR / filename

    image = image.resize(IMAGE_SIZE)
    image.save(image_path, format="JPEG", quality=95)

    records = get_uploaded_images()

    record = {
        "filename": filename,
        "original_filename": image_file.filename,
        "label": disease_label
    }

    records.append(record)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=4)

    return record


def get_uploaded_images():
    if not MANIFEST_PATH.exists():
        return []

    with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# Prepare original training data for rehearsal
# --------------------------------------------------

def get_retraining_data(class_names):
    image_paths = []
    image_labels = []

    random_generator = random.Random(42)

    # Select a representative sample from every original class.
    for class_index, class_name in enumerate(class_names):
        class_directory = DATA_DIR / "train" / class_name

        if not class_directory.exists():
            raise FileNotFoundError(
                f"Original training class not found: {class_name}"
            )

        files = [
            path for path in class_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in ALLOWED_EXTENSIONS
        ]

        if not files:
            raise ValueError(f"No training images found for {class_name}")

        selected_files = random_generator.sample(
            files,
            min(len(files), MAX_ORIGINAL_IMAGES_PER_CLASS)
        )

        image_paths.extend([str(path) for path in selected_files])
        image_labels.extend([class_index] * len(selected_files))

    # Add the newly uploaded labeled images.
    uploaded_records = get_uploaded_images()

    new_image_count = 0

    for record in uploaded_records:
        image_path = INCOMING_DIR / record["filename"]

        if not image_path.exists():
            continue

        if record["label"] not in class_names:
            continue

        image_paths.append(str(image_path))
        image_labels.append(class_names.index(record["label"]))
        new_image_count += 1

    if new_image_count == 0:
        raise ValueError("Upload at least one labeled image before retraining.")

    return image_paths, image_labels, new_image_count


# --------------------------------------------------
# TensorFlow dataset preparation
# --------------------------------------------------

def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    image.set_shape([256, 256, 3])

    return image, label


def create_retraining_dataset(image_paths, image_labels):
    dataset = tf.data.Dataset.from_tensor_slices(
        (image_paths, image_labels)
    )

    dataset = dataset.shuffle(
        buffer_size=len(image_paths),
        seed=42,
        reshuffle_each_iteration=True
    )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_validation_dataset(class_names):
    return tf.keras.utils.image_dataset_from_directory(
        str(DATA_DIR / "validation"),
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_names=class_names,
        label_mode="int",
        shuffle=False
    ).prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# Incremental fine-tuning
# --------------------------------------------------

def run_incremental_retraining(current_model, class_names):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    candidate_path = (
        EXPERIMENTS_DIR / f"candidate_{timestamp}.keras"
    )

    validation_dataset = create_validation_dataset(class_names)

    # Measure the current production model first.
    baseline_loss, baseline_accuracy = current_model.evaluate(
        validation_dataset,
        verbose=0
    )

    # Combine original training samples with new labeled images.
    image_paths, image_labels, new_image_count = (
        get_retraining_data(class_names)
    )

    training_dataset = create_retraining_dataset(
        image_paths,
        image_labels
    )

    # Clone the existing model so the production model remains untouched.
    candidate_model = tf.keras.models.clone_model(current_model)
    candidate_model.set_weights(current_model.get_weights())

    candidate_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=RETRAINING_LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = candidate_model.fit(
        training_dataset,
        validation_data=validation_dataset,
        epochs=RETRAINING_EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )

    # Save the candidate separately before deciding whether to promote it.
    candidate_model.save(candidate_path)

    candidate_loss, candidate_accuracy = candidate_model.evaluate(
        validation_dataset,
        verbose=0
    )

    # Keep production performance within a small tolerance.
    accepted = (
        candidate_accuracy >= baseline_accuracy - 0.005
        and candidate_loss <= baseline_loss * 1.05
    )

    backup_path = None

    if accepted:
        # Preserve the previous production model before replacing it.
        backup_path = (
            EXPERIMENTS_DIR / f"previous_production_{timestamp}.keras"
        )

        if PRODUCTION_MODEL_PATH.exists():
            shutil.copy2(PRODUCTION_MODEL_PATH, backup_path)

        shutil.copy2(candidate_path, PRODUCTION_MODEL_PATH)

    result = {
        "status": "accepted" if accepted else "rejected",
        "timestamp": timestamp,
        "new_images": new_image_count,
        "training_images": len(image_paths),
        "baseline_validation_accuracy": float(baseline_accuracy),
        "candidate_validation_accuracy": float(candidate_accuracy),
        "baseline_validation_loss": float(baseline_loss),
        "candidate_validation_loss": float(candidate_loss),
        "epochs_completed": len(history.history["loss"]),
        "candidate_model": str(candidate_path.relative_to(BASE_DIR)),
        "backup_model": (
            str(backup_path.relative_to(BASE_DIR))
            if backup_path else None
        )
    }

    # Save an experiment record for reporting and auditing.
    record_path = EXPERIMENTS_DIR / f"experiment_{timestamp}.json"

    with open(record_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=4)

    return (candidate_model if accepted else current_model), result


from pathlib import Path
from datetime import datetime, timezone
from contextlib import closing
import json
import sqlite3

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from src.training.retraining import (
    save_labeled_image,
    get_uploaded_images,
    run_incremental_retraining
)



# --------------------------------------------------
# 1. Application configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
DATABASE_PATH = DATA_DIR / "predictions.db"

IMAGE_SIZE = (256, 256)
MAX_IMAGE_SIZE_MB = 10

DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE_MB * 1024 * 1024


# --------------------------------------------------
# 2. Database initialization
# --------------------------------------------------

def initialize_database():
    """
    Create the prediction history table if it does not exist.
    """

    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        with connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_created_at
                ON predictions(created_at)
            """)


def save_prediction(filename, prediction, prediction_type):
    """
    Persist a successful prediction in SQLite.
    """

    created_at = datetime.now(timezone.utc).isoformat()

    with closing(sqlite3.connect(DATABASE_PATH)) as connection:
        with connection:
            cursor = connection.execute("""
                INSERT INTO predictions (
                    filename,
                    predicted_class,
                    confidence,
                    probabilities,
                    prediction_type,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                filename,
                prediction["predicted_class"],
                prediction["confidence"],
                json.dumps(prediction["probabilities"]),
                prediction_type,
                created_at
            ))

            prediction_id = cursor.lastrowid

    return prediction_id


# Initialize database when the application starts.
initialize_database()


# --------------------------------------------------
# 3. Load trained model and class names
# --------------------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

if len(class_names) != model.output_shape[-1]:
    raise ValueError(
        "Number of class names does not match model output."
    )


# --------------------------------------------------
# 4. Image preprocessing
# --------------------------------------------------

def preprocess_image(image_file):
    """
    Convert an uploaded image into a model-compatible tensor.
    """

    image = Image.open(image_file).convert("RGB")

    image = image.resize(
        IMAGE_SIZE,
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(image, dtype=np.float32)

    # Add batch dimension: (256, 256, 3) -> (1, 256, 256, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# --------------------------------------------------
# 5. Prediction function
# --------------------------------------------------

def predict_image(image_file):
    image_array = preprocess_image(image_file)

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_class = class_names[predicted_index]

    confidence = float(probabilities[predicted_index])

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100, 2),
        "probabilities": {
            class_names[index]: round(float(probability), 4)
            for index, probability in enumerate(probabilities)
        }
    }


# --------------------------------------------------
# 6. API health endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": "Plant Disease Classification CNN",
        "number_of_classes": len(class_names)
    }), 200


# --------------------------------------------------
# 7. Single-image prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({
            "error": "No image uploaded. Use the field name 'image'."
        }), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({
            "error": "No image selected."
        }), 400

    try:
        result = predict_image(image_file)

        prediction_id = save_prediction(
            image_file.filename,
            result,
            "single"
        )

        return jsonify({
            "success": True,
            "prediction_id": prediction_id,
            "prediction": result
        }), 200

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "Invalid or unsupported image file."
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "An internal prediction error occurred."
        }), 500


# --------------------------------------------------
# 8. Bulk image prediction endpoint
# --------------------------------------------------

@app.route("/predict/batch", methods=["POST"])
def predict_batch():

    if "images" not in request.files:
        return jsonify({
            "error": "No images uploaded. Use the field name 'images'."
        }), 400

    image_files = request.files.getlist("images")

    if not image_files or all(
        image.filename == "" for image in image_files
    ):
        return jsonify({
            "error": "No valid images selected."
        }), 400

    results = []
    errors = []

    for image_file in image_files:

        if image_file.filename == "":
            continue

        try:
            prediction = predict_image(image_file)

            prediction_id = save_prediction(
                image_file.filename,
                prediction,
                "batch"
            )

            results.append({
                "prediction_id": prediction_id,
                "filename": image_file.filename,
                "success": True,
                "prediction": prediction
            })

        except (UnidentifiedImageError, OSError):
            errors.append({
                "filename": image_file.filename,
                "error": "Invalid or unsupported image file."
            })

        except Exception:
            app.logger.exception(
                "Bulk prediction failed for %s",
                image_file.filename
            )

            errors.append({
                "filename": image_file.filename,
                "error": "Prediction failed."
            })

    return jsonify({
        "success": len(results) > 0,
        "total_uploaded": len(image_files),
        "successful_predictions": len(results),
        "failed_predictions": len(errors),
        "results": results,
        "errors": errors
    }), 200 if results else 400


# --------------------------------------------------
# 9. Prediction history endpoint
# --------------------------------------------------

@app.route("/history", methods=["GET"])
def prediction_history():

    try:
        limit = request.args.get("limit", default=50, type=int)

        if limit is None or limit < 1:
            return jsonify({
                "error": "Limit must be a positive integer."
            }), 400

        limit = min(limit, 500)

        with closing(sqlite3.connect(DATABASE_PATH)) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute("""
                SELECT *
                FROM predictions
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()

        history = []

        for row in rows:
            history.append({
                "id": row["id"],
                "filename": row["filename"],
                "predicted_class": row["predicted_class"],
                "confidence": row["confidence"],
                "confidence_percentage": round(
                    row["confidence"] * 100, 2
                ),
                "probabilities": json.loads(row["probabilities"]),
                "prediction_type": row["prediction_type"],
                "created_at": row["created_at"]
            })

        return jsonify({
            "success": True,
            "total_returned": len(history),
            "history": history
        }), 200

    except Exception:
        app.logger.exception("Could not retrieve prediction history")

        return jsonify({
            "error": "Could not retrieve prediction history."
        }), 500


# --------------------------------------------------
# 10. Retraining dataset upload endpoints
# --------------------------------------------------


@app.route("/retraining/upload", methods=["POST"])
def upload_retraining_images():
    try:
        images = request.files.getlist("images")
        labels = request.form.getlist("labels")

        if not images:
            return jsonify({
                "success": False,
                "message": "Upload at least one image."
            }), 400

        if len(images) != len(labels):
            return jsonify({
                "success": False,
                "message": "Each image must have a corresponding disease label."
            }), 400

        uploaded_records = []
        errors = []

        for image_file, label in zip(images, labels):

            try:
                record = save_labeled_image(
                    image_file,
                    label
                )

                uploaded_records.append(record)

            except ValueError as error:
                errors.append({
                    "filename": image_file.filename,
                    "error": str(error)
                })

        return jsonify({
            "success": len(uploaded_records) > 0,
            "uploaded": len(uploaded_records),
            "failed": len(errors),
            "images": uploaded_records,
            "errors": errors
        }), 200

    except Exception as error:
        app.logger.exception("Retraining image upload failed")

        return jsonify({
            "success": False,
            "message": str(error)
        }), 500



@app.route("/retraining/images", methods=["GET"])
def list_retraining_images():

    records = get_uploaded_images()

    return jsonify({
        "success": True,
        "total_images": len(records),
        "images": records
    }), 200


@app.route("/retraining/train", methods=["POST"])
def retrain_model():
    global model

    try:
        uploaded_images = get_uploaded_images()

        if not uploaded_images:
            return jsonify({
                "success": False,
                "message": "Upload labeled images before starting retraining."
            }), 400

        # Fine-tune a copy of the current production model.
        updated_model, result = run_incremental_retraining(
            current_model=model,
            class_names=class_names
        )

        # Update the running API model only if the candidate was accepted.
        if result["status"] == "accepted":
            model = updated_model

        return jsonify({
            "success": True,
            "message": (
                "Retraining completed and the candidate was promoted."
                if result["status"] == "accepted"
                else "Retraining completed, but the candidate did not pass validation."
            ),
            "result": result
        }), 200

    except ValueError as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 400

    except Exception as error:
        app.logger.exception("Retraining failed")

        return jsonify({
            "success": False,
            "message": "An error occurred during model retraining.",
            "error": str(error)
        }), 500


# --------------------------------------------------
# 11. Start development server
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )

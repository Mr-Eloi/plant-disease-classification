# Tomato Plant Disease Classification

An end-to-end machine learning application that classifies tomato leaf images into 10 disease and health categories using a custom Convolutional Neural Network (CNN).

**Quick Links**

- **Live Dashboard:** [Plant Disease Classification](http://13.63.53.187:8501)
- **GitHub Repository:** [Plant Disease Classification](https://github.com/Mr-Eloi/plant-disease-classification)
- **Project Demonstration:** [Watch on YouTube](https://youtu.be/qHKWg1qqWGU)
- **Machine Learning Notebook:** [View Notebook](notebooks/plant_disease_classification.ipynb)

---

## 1. Project Overview

The system accepts tomato leaf images and predicts their corresponding disease or health category, returning the predicted class and confidence score.

The project demonstrates an end-to-end machine learning lifecycle, including data acquisition, preprocessing, model development, evaluation, interpretability, deployment, inference, and incremental retraining.

### Main Features

- **Image Classification:** Predicts one of 10 tomato leaf categories.
- **Single Image Prediction:** Classifies an individual uploaded image.
- **Batch Prediction:** Processes multiple images.
- **Prediction History:** Records predictions for later inspection.
- **Analytics Dashboard:** Displays prediction activity and model information.
- **Model Interpretability:** Uses Grad-CAM to visualize image regions influencing selected predictions.
- **Incremental Retraining:** Accepts newly labeled images, fine-tunes the existing model, evaluates the candidate, and promotes it when acceptance criteria are satisfied.
- **REST API:** Provides prediction, history, and retraining endpoints.
- **Docker Deployment:** Packages the Flask API and its dependencies into a Docker container.
- **Load Testing:** Uses Locust to evaluate API performance under concurrent requests.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Machine learning | TensorFlow / Keras |
| Image processing | Pillow |
| Numerical processing | NumPy |
| Data analysis | Pandas |
| Model evaluation | scikit-learn |
| API | Flask |
| API server | Gunicorn |
| User interface | Streamlit |
| Visualizations | Plotly, Matplotlib |
| Metadata storage | SQLite |
| Load testing | Locust |
| Containerization | Docker, Docker Compose |
| Cloud deployment | AWS EC2 |

## 3. Project Structure

```text
plant-disease-classification/
├── models/
│   ├── plant_disease_model.keras
│   └── class_names.json
├── notebooks/
│   └── plant_disease_classification.ipynb
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── dashboard.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── acquire_data.py
│   └── training/
│       ├── __init__.py
│       └── retraining.py
├── Dockerfile.api
├── docker-compose.yml
├── .dockerignore
├── .gitignore
├── .python-version
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

- `models/`: Contains the exported production model and class-name mapping.
- `notebooks/`: Contains the documented machine learning experiments.
- `src/api/`: Implements the Flask prediction and retraining API.
- `src/dashboard/`: Implements the Streamlit user interface.
- `src/data/`: Contains dataset acquisition and preparation scripts.
- `src/training/`: Implements incremental model retraining.
- `Dockerfile.api`: Defines the Flask API container image.
- `docker-compose.yml`: Defines the Docker service configuration.
- `.dockerignore`: Excludes unnecessary files from the Docker build context.

## 4. Dataset

The project uses a PlantVillage-derived tomato leaf image dataset containing 10 classes.

The categories include:

- Bacterial spot
- Early blight
- Healthy
- Late blight
- Leaf mold
- Septoria leaf spot
- Spider mites
- Target spot
- Tomato mosaic virus
- Tomato yellow leaf curl virus

The dataset contains **18,160 images**, divided into:

| Dataset split | Images |
|---|---:|
| Training | 12,712 |
| Validation | 2,724 |
| Testing | 2,724 |
| **Total** | **18,160** |

The dataset files are excluded from Git to keep the repository lightweight.

## 5. Machine Learning Pipeline

### Data preprocessing

- Images are resized to 256 × 256 pixels.
- Images are represented in RGB format.
- Labels are encoded as integer class indices.
- Training, validation, and test datasets are loaded in batches of 32.
- Pixel values are normalized to the 0–1 range inside the model.
- Class weights are used to address class imbalance.

### Model architecture

The baseline model is a custom CNN consisting of:

- Three convolutional layers with 32, 64, and 128 filters.
- ReLU activation and max pooling.
- Global average pooling.
- A 128-unit dense layer.
- Dropout regularization with a rate of 0.3.
- A 10-unit softmax output layer.

The model is trained using the Adam optimizer and sparse categorical cross-entropy loss.

### Model evaluation

The selected baseline model achieved the following test results:

| Metric | Result |
|---|---:|
| Accuracy | 90.82% |
| Weighted precision | 91.32% |
| Weighted recall | 90.82% |
| Weighted F1-score | 90.95% |
| Balanced accuracy | 90.70% |

The evaluation also includes a classification report and confusion matrix.

### Model interpretability

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize regions of selected input images that influence the model's predictions.

### Model optimization

Data augmentation and a reduced learning rate were evaluated as optimization experiments. These experiments did not outperform the baseline model on validation performance, so the baseline was retained as the production model.

## 6. Local Installation

### Prerequisites

- Python 3.13.2
- Git
- A compatible environment for TensorFlow

### Clone the repository

```bash
git clone -b refactor/lean-structure https://github.com/Mr-Eloi/plant-disease-classification.git
cd plant-disease-classification
```

### Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
python -m pip install --upgrade pip

# Application dependencies
pip install -r requirements.txt

# Development and notebook dependencies
pip install -r requirements-dev.txt
```

The dataset is not included in the repository. Prepare the dataset and its training, validation, and test directories before running experiments that depend on those files.

## 7. Running the Application

The application uses two components: a Flask API and a Streamlit dashboard.

### Start the Flask API

From the project root:

```bash
python -m src.api.app
```

The API runs locally on port `5000`.

### Start the Streamlit dashboard

Open another terminal, activate the same virtual environment, and run:

```bash
streamlit run src/dashboard/dashboard.py
```

The dashboard runs locally on port `8501`.

The dashboard communicates with the Flask API through the `API_URL` environment variable. Its default API address is `http://127.0.0.1:5000`.

## 8. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Checks API and model availability. |
| POST | `/predict` | Predicts the class of one uploaded image. |
| POST | `/predict/batch` | Predicts classes for multiple uploaded images. |
| GET | `/history` | Retrieves stored prediction history. |
| POST | `/retraining/upload` | Uploads newly labeled images. |
| GET | `/retraining/images` | Retrieves uploaded retraining images. |
| POST | `/retraining/train` | Starts incremental model retraining. |

### Prediction requests

The `/predict` endpoint expects an image uploaded using the multipart form field `image`.

The `/predict/batch` endpoint accepts multiple images using the multipart form field `images`.

## 9. Incremental Retraining

The retraining pipeline uses the existing production model as its starting point.

The workflow:

1. Accepts newly uploaded images with their class labels.
2. Validates and saves the uploaded images.
3. Combines uploaded images with a sample of the original training data.
4. Applies the required image preprocessing.
5. Fine-tunes a copy of the existing model.
6. Evaluates the candidate model using validation data.
7. Compares the candidate against the current production model.
8. Promotes the candidate only when the configured acceptance criteria are satisfied.

The existing production model is backed up before an accepted candidate replaces it.

## 10. Docker Containerization

The Flask API is containerized using Docker.

The repository includes:

- `Dockerfile.api`: Builds the API image using Python 3.12 and installs the required runtime dependencies.
- `docker-compose.yml`: Defines the API container, port mapping, persistent data and model volumes.
- `.dockerignore`: Excludes unnecessary project files from the Docker build context.

### Build the Docker image

```bash
docker build -f Dockerfile.api -t plant-disease-api:1.0 .
```

### Start the container

```bash
docker compose up -d
```

### Verify the running container

```bash
docker ps
```

### Check container logs

```bash
docker compose logs -f api
```

The containerized API is mapped to host port `5001`, bound to localhost in the deployment configuration.

## 11. Cloud Deployment

The application is deployed on an AWS EC2 instance running Ubuntu Server 24.04.

### Deployment architecture

- **Cloud infrastructure:** AWS EC2.
- **API service:** Flask served using Gunicorn.
- **Dashboard service:** Streamlit.
- **Containerized API:** Docker.
- **Service management:** systemd.
- **API communication:** Internal loopback interface.

### Live application

[Open the deployed Plant Disease Classification Dashboard](http://13.63.53.187:8501)

The dashboard is accessible through port `8501`. The Flask API runs internally on port `5000`.

## 12. Load Testing and Performance Evaluation

Locust was used to evaluate API performance under different concurrent user loads.

Each test ran for 60 seconds against the Dockerized prediction API.

### Performance results

| Metric | 1 Concurrent User | 5 Concurrent Users |
|---|---:|---:|
| Total requests | 36 | 177 |
| Failed requests | 0 | 0 |
| Average response time | 106.97 ms | 124.33 ms |
| Median response time | 110 ms | 120 ms |
| 95th percentile latency | 120 ms | 210 ms |
| Maximum response time | 125.11 ms | 318.90 ms |
| Throughput | 0.61 req/s | 3.00 req/s |

### Performance observations

- Throughput increased approximately 4.9 times when concurrency increased from one to five users.
- Average response time increased by approximately 16.2%.
- The 95th percentile latency increased from 120 ms to 210 ms.
- Both tests recorded zero failed requests.

### Test reports

The original Locust HTML reports are available in the EC2 project directory:

- `report_1_user.html`
- `report_5_users.html`

### Testing limitations

Locust and the API ran on the same EC2 instance, meaning the load generator also consumed server resources.

The experiments compare one and five concurrent simulated users against a single Docker API container. A multi-container replica comparison was not completed due to the memory limitations of the available EC2 instance.

These results should therefore be interpreted as a concurrency test rather than a multi-container scaling benchmark.

## 13. Project Demonstration

Watch the video demonstration covering the machine learning workflow, deployed application, prediction functionality, retraining process, Docker deployment and Locust performance evaluation.

[Watch the Project Demonstration on YouTube](https://youtu.be/qHKWg1qqWGU)

## 14. Machine Learning Documentation

The complete experimental workflow is documented in:

`notebooks/plant_disease_classification.ipynb`

It covers:

- Dataset exploration.
- Image preprocessing.
- CNN development and training.
- Model evaluation.
- Grad-CAM interpretability.
- Optimization experiments.
- Production model selection.
- Retraining experiments.

## 15. Limitations

- The model is trained specifically on tomato leaf categories and should not be assumed to classify other plant species reliably.
- Prediction confidence is not a guarantee that a classification is correct.
- Grad-CAM visualizations indicate model activation patterns and do not establish biological causation.
- Performance on images captured under conditions substantially different from the training dataset requires further validation.
- Retraining quality depends on the accuracy, diversity, and representativeness of newly labeled images.
- The load-testing experiments were performed on a single Docker container and do not establish multi-container scaling performance.

## 16. License

Add the project's chosen license before distributing or reusing the repository's code and model.

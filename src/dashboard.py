
import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide"
)

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:5000"
).rstrip("/")


# --------------------------------------------------
# 2. API communication functions
# --------------------------------------------------

def check_api_health():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException:
        return None


def get_prediction_history(limit=500):
    try:
        response = requests.get(
            f"{API_URL}/history",
            params={"limit": limit},
            timeout=15
        )

        response.raise_for_status()

        return response.json().get("history", [])

    except requests.RequestException:
        return None


def predict_single_image(image):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={
                "image": (
                    image.name,
                    image.getvalue(),
                    image.type
                )
            },
            timeout=60
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(f"Prediction request failed: {error}")
        return None


def predict_multiple_images(images):
    try:
        files = [
            (
                "images",
                (image.name, image.getvalue(), image.type)
            )
            for image in images
        ]

        response = requests.post(
            f"{API_URL}/predict/batch",
            files=files,
            timeout=180
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(f"Bulk prediction request failed: {error}")
        return None


def get_retraining_images():
    try:
        response = requests.get(
            f"{API_URL}/retraining/images",
            timeout=15
        )

        response.raise_for_status()
        return response.json().get("images", [])

    except requests.RequestException:
        return None


def upload_retraining_images(images, disease_label):
    try:
        files = [
            (
                "images",
                (image.name, image.getvalue(), image.type)
            )
            for image in images
        ]

        data = [
            ("labels", disease_label)
            for _ in images
        ]

        response = requests.post(
            f"{API_URL}/retraining/upload",
            files=files,
            data=data,
            timeout=120
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(f"Image upload failed: {error}")
        return None


def trigger_retraining():
    try:
        response = requests.post(
            f"{API_URL}/retraining/train",
            timeout=1800
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as error:
        st.error(f"Retraining request failed: {error}")
        return None


# --------------------------------------------------
# 3. Sidebar navigation
# --------------------------------------------------

st.sidebar.title("🌿 Plant Disease AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Single Prediction",
        "Bulk Prediction",
        "History & Analytics",
        "Model Retraining"
    ]
)

health = check_api_health()

if health:
    st.sidebar.success("API Online")
    st.sidebar.caption(
        f"Model: {health['model_name']}"
    )
else:
    st.sidebar.error("API Offline")
    st.sidebar.caption(
        "Start the Flask API using: python src/app.py"
    )


# --------------------------------------------------
# 4. Retrieve prediction history
# --------------------------------------------------

history = get_prediction_history()

if history is None:
    history = []

history_df = pd.DataFrame(history)

if not history_df.empty:
    history_df["created_at"] = pd.to_datetime(
        history_df["created_at"],
        utc=True
    )

    history_df["confidence_percentage"] = (
        history_df["confidence"] * 100
    )


# --------------------------------------------------
# 5. Overview page
# --------------------------------------------------

if page == "Overview":

    st.title("Plant Disease Classification Dashboard")

    st.write(
        "Monitor plant disease predictions, model activity, "
        "and classification results."
    )

    if health is None:
        st.warning(
            "The Flask API is unavailable. Start the backend "
            "to enable predictions and monitoring."
        )

    if not history_df.empty:

        total_predictions = len(history_df)

        average_confidence = (
            history_df["confidence"].mean() * 100
        )

        most_common_disease = (
            history_df["predicted_class"].mode().iloc[0]
        )

        single_predictions = (
            history_df["prediction_type"] == "single"
        ).sum()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Predictions",
            total_predictions
        )

        col2.metric(
            "Average Confidence",
            f"{average_confidence:.2f}%"
        )

        col3.metric(
            "Most Detected Disease",
            most_common_disease.replace("Tomato___", "")
        )

        col4.metric(
            "Single Predictions",
            int(single_predictions)
        )

        st.subheader("Recent Predictions")

        recent = history_df.head(10).copy()

        recent["created_at"] = (
            recent["created_at"]
            .dt.strftime("%Y-%m-%d %H:%M UTC")
        )

        st.dataframe(
            recent[
                [
                    "filename",
                    "predicted_class",
                    "confidence_percentage",
                    "prediction_type",
                    "created_at"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info(
            "No predictions recorded yet. "
            "Upload an image to get started."
        )


# --------------------------------------------------
# 6. Single prediction page
# --------------------------------------------------

elif page == "Single Prediction":

    st.title("Single Image Prediction")

    st.write(
        "Upload a tomato leaf image to identify its "
        "most likely disease."
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Uploaded leaf image",
            use_container_width=True
        )

        if st.button(
            "Analyze Image",
            type="primary",
            disabled=health is None
        ):

            with st.spinner("Analyzing image..."):

                result = predict_single_image(uploaded_image)

            if result and result.get("success"):

                prediction = result["prediction"]

                st.success("Prediction completed!")

                st.subheader(
                    prediction["predicted_class"].replace(
                        "Tomato___", ""
                    ).replace("_", " ")
                )

                st.metric(
                    "Confidence",
                    f"{prediction['confidence_percentage']}%"
                )

                probabilities = pd.DataFrame(
                    [
                        {
                            "Disease": name.replace(
                                "Tomato___", ""
                            ).replace("_", " "),
                            "Probability": probability * 100
                        }
                        for name, probability
                        in prediction["probabilities"].items()
                    ]
                ).sort_values(
                    "Probability",
                    ascending=False
                )

                figure = px.bar(
                    probabilities,
                    x="Probability",
                    y="Disease",
                    orientation="h",
                    title="Class Probability Distribution"
                )

                figure.update_layout(
                    yaxis={"categoryorder": "total ascending"}
                )

                st.plotly_chart(
                    figure,
                    use_container_width=True
                )

                st.caption(
                    f"Prediction ID: {result['prediction_id']}"
                )


# --------------------------------------------------
# 7. Bulk prediction page
# --------------------------------------------------

elif page == "Bulk Prediction":

    st.title("Bulk Image Prediction")

    st.write(
        "Upload multiple tomato leaf images for "
        "simultaneous classification."
    )

    uploaded_images = st.file_uploader(
        "Choose multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_images:

        st.write(
            f"{len(uploaded_images)} images selected."
        )

        if st.button(
            "Analyze All Images",
            type="primary",
            disabled=health is None
        ):

            with st.spinner("Processing images..."):

                result = predict_multiple_images(
                    uploaded_images
                )

            if result:

                st.success(
                    f"Processed {result['successful_predictions']} "
                    "images successfully."
                )

                st.metric(
                    "Failed Predictions",
                    result["failed_predictions"]
                )

                results = result.get("results", [])

                if results:

                    records = []

                    for item in results:

                        prediction = item["prediction"]

                        records.append({
                            "Filename": item["filename"],
                            "Predicted Disease": (
                                prediction["predicted_class"]
                                .replace("Tomato___", "")
                                .replace("_", " ")
                            ),
                            "Confidence (%)": (
                                prediction["confidence_percentage"]
                            )
                        })

                    st.dataframe(
                        pd.DataFrame(records),
                        use_container_width=True,
                        hide_index=True
                    )

                if result.get("errors"):
                    st.warning("Some images could not be processed.")

                    st.dataframe(
                        pd.DataFrame(result["errors"]),
                        use_container_width=True,
                        hide_index=True
                    )


# --------------------------------------------------
# 8. History and analytics page
# --------------------------------------------------

elif page == "History & Analytics":

    st.title("Prediction History & Analytics")

    if history_df.empty:

        st.info(
            "No prediction records available yet."
        )

    else:

        st.subheader("Disease Distribution")

        disease_counts = (
            history_df["predicted_class"]
            .str.replace("Tomato___", "", regex=False)
            .value_counts()
            .reset_index()
        )

        disease_counts.columns = [
            "Disease",
            "Number of Predictions"
        ]

        figure = px.pie(
            disease_counts,
            names="Disease",
            values="Number of Predictions",
            title="Predictions by Disease"
        )

        st.plotly_chart(
            figure,
            use_container_width=True
        )

        st.subheader("Prediction Confidence")

        confidence_figure = px.histogram(
            history_df,
            x="confidence_percentage",
            nbins=10,
            title="Distribution of Prediction Confidence",
            labels={
                "confidence_percentage": "Confidence (%)"
            }
        )

        st.plotly_chart(
            confidence_figure,
            use_container_width=True
        )

        st.subheader("Prediction Activity Over Time")

        activity = (
            history_df
            .assign(
                date=history_df["created_at"].dt.date
            )
            .groupby("date")
            .size()
            .reset_index(name="Predictions")
        )

        activity_figure = px.line(
            activity,
            x="date",
            y="Predictions",
            markers=True,
            title="Daily Prediction Activity"
        )

        st.plotly_chart(
            activity_figure,
            use_container_width=True
        )

        st.subheader("Complete Prediction History")

        display_history = history_df.copy()

        display_history["predicted_class"] = (
            display_history["predicted_class"]
            .str.replace("Tomato___", "", regex=False)
            .str.replace("_", " ", regex=False)
        )

        display_history["created_at"] = (
            display_history["created_at"]
            .dt.strftime("%Y-%m-%d %H:%M UTC")
        )

        st.dataframe(
            display_history[
                [
                    "id",
                    "filename",
                    "predicted_class",
                    "confidence_percentage",
                    "prediction_type",
                    "created_at"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        csv = display_history.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Prediction History (CSV)",
            data=csv,
            file_name="plant_disease_prediction_history.csv",
            mime="text/csv"
        )


# --------------------------------------------------
# 9. Model retraining page
# --------------------------------------------------

elif page == "Model Retraining":

    st.title("Incremental Model Retraining")

    st.write(
        "Upload newly labeled tomato leaf images and "
        "fine-tune the existing model with additional training data."
    )

    if health is None:

        st.warning("The Flask API is offline.")

    else:

        # ------------------------------------------
        # A. Upload labeled images
        # ------------------------------------------

        st.subheader("1. Upload Labeled Images")

        st.info(
            "Select images belonging to the same disease class. "
            "You can repeat the process for different classes."
        )

        class_names = [
            "Tomato___Bacterial_spot",
            "Tomato___Early_blight",
            "Tomato___healthy",
            "Tomato___Late_blight",
            "Tomato___Leaf_Mold",
            "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite",
            "Tomato___Target_Spot",
            "Tomato___Tomato_mosaic_virus",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
        ]

        selected_label = st.selectbox(
            "Select the correct disease label",
            class_names,
            format_func=lambda name: (
                name.replace("Tomato___", "").replace("_", " ")
            )
        )

        new_images = st.file_uploader(
            "Upload labeled leaf images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="retraining_uploader"
        )

        if new_images:

            st.write(
                f"{len(new_images)} images selected."
            )

            if st.button(
                "Upload Training Images",
                type="primary"
            ):

                with st.spinner("Uploading labeled images..."):

                    upload_result = upload_retraining_images(
                        new_images,
                        selected_label
                    )

                if upload_result:

                    st.success(
                        f"{upload_result['uploaded']} images uploaded successfully."
                    )

                    if upload_result["failed"] > 0:

                        st.warning(
                            f"{upload_result['failed']} images could not be uploaded."
                        )

                        st.dataframe(
                            pd.DataFrame(upload_result["errors"]),
                            use_container_width=True
                        )

        st.divider()

        # ------------------------------------------
        # B. View uploaded training images
        # ------------------------------------------

        st.subheader("2. Uploaded Training Images")

        uploaded_records = get_retraining_images()

        if uploaded_records is None:

            st.error("Could not retrieve uploaded training images.")

        elif uploaded_records:

            uploaded_df = pd.DataFrame(uploaded_records)

            st.metric(
                "Total Labeled Images",
                len(uploaded_records)
            )

            if "label" in uploaded_df.columns:

                label_counts = (
                    uploaded_df["label"]
                    .value_counts()
                    .reset_index()
                )

                label_counts.columns = [
                    "Disease",
                    "Image Count"
                ]

                label_counts["Disease"] = (
                    label_counts["Disease"]
                    .str.replace("Tomato___", "", regex=False)
                    .str.replace("_", " ", regex=False)
                )

                st.dataframe(
                    label_counts,
                    use_container_width=True,
                    hide_index=True
                )

            with st.expander("View uploaded image records"):

                st.dataframe(
                    uploaded_df,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "No labeled training images have been uploaded."
            )

        st.divider()

        # ------------------------------------------
        # C. Trigger manual retraining
        # ------------------------------------------

        st.subheader("3. Start Model Retraining")

        st.write(
            "The system will combine uploaded images with "
            "representative samples from the original training "
            "dataset and fine-tune the existing model."
        )

        can_retrain = bool(uploaded_records)

        if st.button(
            "Start Incremental Retraining",
            type="primary",
            disabled=not can_retrain
        ):

            with st.spinner(
                "Fine-tuning model. This may take several minutes..."
            ):

                retraining_result = trigger_retraining()

            if retraining_result:

                result = retraining_result["result"]

                if result["status"] == "accepted":

                    st.success(
                        "The candidate model passed validation "
                        "and was promoted to production."
                    )

                else:

                    st.warning(
                        "The candidate did not pass validation. "
                        "The previous production model remains active."
                    )

                # ----------------------------------
                # D. Display retraining results
                # ----------------------------------

                st.subheader("Retraining Results")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "New Images",
                    result["new_images"]
                )

                col2.metric(
                    "Training Images",
                    result["training_images"]
                )

                col3.metric(
                    "Epochs Completed",
                    result["epochs_completed"]
                )

                col1, col2 = st.columns(2)

                col1.metric(
                    "Validation Accuracy",
                    f"{result['candidate_validation_accuracy'] * 100:.2f}%",
                    delta=(
                        f"{(
                            result['candidate_validation_accuracy'] -
                            result['baseline_validation_accuracy']
                        ) * 100:+.2f} percentage points"
                    )
                )

                col2.metric(
                    "Validation Loss",
                    f"{result['candidate_validation_loss']:.4f}",
                    delta=(
                        f"{(
                            result['candidate_validation_loss'] -
                            result['baseline_validation_loss']
                        ):+.4f}"
                    ),
                    delta_color="inverse"
                )

                comparison = pd.DataFrame({
                    "Metric": [
                        "Validation Accuracy",
                        "Validation Loss"
                    ],
                    "Before Retraining": [
                        result["baseline_validation_accuracy"],
                        result["baseline_validation_loss"]
                    ],
                    "After Retraining": [
                        result["candidate_validation_accuracy"],
                        result["candidate_validation_loss"]
                    ]
                })

                st.dataframe(
                    comparison,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    f"Experiment: {result['timestamp']}"
                )

                st.caption(
                    f"Candidate model: {result['candidate_model']}"
                )

# --------------------------------------------------
# 10. Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Plant Disease Classification System | "
    "TensorFlow + Flask + Streamlit"
)

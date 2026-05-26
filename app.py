import streamlit as st
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image, ImageOps

st.title("YOLO-Based Object Detection Dashboard with Confidence Threshold Analysis")
st.write(
    "Upload an image to detect objects, inspect bounding boxes, and run a confidence-threshold experiment."
)

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model


model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    current_file_name = uploaded_file.name

    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = current_file_name

    if st.session_state.last_uploaded_file != current_file_name:
        st.session_state.experiment_df = None
        st.session_state.last_uploaded_file = current_file_name

    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.05
    )

    results = model(image, conf=confidence)

    detected_class_names = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        detected_class_names.append(class_name)

    unique_detected_classes = sorted(list(set(detected_class_names)))

    selected_classes = st.multiselect(
        "Filter by object class",
        options=unique_detected_classes,
        default=[]
    )

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        confidence_score = float(box.conf[0])

        if selected_classes and class_name not in selected_classes:
            continue
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height

        detections.append({
            "Object": class_name,
            "Confidence": round(confidence_score, 2),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "Box Width": box_width,
            "Box Height": box_height,
            "Box Area": box_area
        })

    total_detections = len(detections)
    unique_objects = len(set([item["Object"] for item in detections]))

    col1, col2 = st.columns(2)

    col1.metric("Total Detections", total_detections)
    col2.metric("Unique Object Types", unique_objects)

    if detections:
        st.subheader("Detection Summary")

        detection_df = pd.DataFrame(detections)
        st.dataframe(detection_df, use_container_width=True)

        detection_csv = detection_df.to_csv(index=False)

        st.download_button(
            label="Download Detection Summary as CSV",
            data=detection_csv,
            file_name="detection_summary.csv",
            mime="text/csv"
        )

        count_df = pd.DataFrame(detections)["Object"].value_counts().reset_index()
        count_df.columns = ["Object", "Count"]

        st.subheader("Object Counts")
        st.dataframe(count_df, use_container_width=True)
    else:
        st.warning("No objects detected. Try lowering the confidence threshold.")

    image_array = np.array(image).copy()

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        confidence_score = float(box.conf[0])

        if selected_classes and class_name not in selected_classes:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        box_color = (255, 0, 0)
        text_color = (255, 255, 255)

        label = f"{class_name} {confidence_score:.2f}"

        cv2.rectangle(image_array, (x1, y1), (x2, y2), box_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 4

        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            font_thickness
        )

        label_y = max(y1, text_height + 10)

        cv2.rectangle(
            image_array,
            (x1, label_y - text_height - 10),
            (x1 + text_width + 10, label_y + baseline - 5),
            box_color,
            -1
        )

        cv2.putText(
            image_array,
            label,
            (x1 + 5, label_y - 7),
            font,
            font_scale,
            text_color,
            font_thickness
        )

    st.subheader("Detected Objects")
    st.image(image_array, use_container_width=True)

    st.subheader("Confidence Threshold Experiment")

    experiment_thresholds = [0.25, 0.40, 0.50, 0.60, 0.75, 0.90]

    if "experiment_df" not in st.session_state:
        st.session_state.experiment_df = None

    if st.button("Run Threshold Experiment"):
        experiment_results = []

        for threshold in experiment_thresholds:
            exp_results = model(image, conf=threshold)

            exp_detections = []

            for box in exp_results[0].boxes:
                class_id = int(box.cls[0])
                class_name = exp_results[0].names[class_id]
                confidence_score = float(box.conf[0])

                exp_detections.append({
                    "Object": class_name,
                    "Confidence": confidence_score
                })

            total_exp_detections = len(exp_detections)
            unique_classes = len(set([item["Object"] for item in exp_detections]))

            if total_exp_detections > 0:
                average_confidence = (
                    sum([item["Confidence"] for item in exp_detections])
                    / total_exp_detections
                )
            else:
                average_confidence = 0

            experiment_results.append({
                "Threshold": threshold,
                "Total Detections": total_exp_detections,
                "Unique Object Types": unique_classes,
                "Average Confidence": round(average_confidence, 2)
            })

        st.session_state.experiment_df = pd.DataFrame(experiment_results)
    
    if st.session_state.experiment_df is not None:
        experiment_df = st.session_state.experiment_df

        st.subheader("Experiment Results")
        st.dataframe(experiment_df, use_container_width=True)

        csv_data = experiment_df.to_csv(index=False)

        st.download_button(
            label="Download Experiment Results as CSV",
            data=csv_data,
            file_name="threshold_experiment_results.csv",
            mime="text/csv"
        )

        st.subheader("Threshold vs Total Detections")

        chart_df = experiment_df.set_index("Threshold")
        st.line_chart(chart_df["Total Detections"])

        st.subheader("Threshold vs Average Confidence")
        st.line_chart(chart_df["Average Confidence"])

        lowest_threshold = experiment_df.iloc[0]
        highest_threshold = experiment_df.iloc[-1]

        st.subheader("Experiment Interpretation")

        st.write(
            f"When the confidence threshold increased from "
            f"{lowest_threshold['Threshold']} to {highest_threshold['Threshold']}, "
            f"the total number of detections changed from "
            f"{lowest_threshold['Total Detections']} to {highest_threshold['Total Detections']}."
        )

        st.write(
            "This shows that increasing the confidence threshold makes the detector more selective. "
            "Lower thresholds may detect more objects, but they can also include weaker or noisy predictions. "
            "Higher thresholds usually produce cleaner detections, but some valid objects may be missed."
        )
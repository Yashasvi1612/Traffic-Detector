import cv2
import time
import threading
import winsound
import gradio as gr
import numpy as np
import csv
import os
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=15, n_init=3, nms_max_overlap=0.5)

# Classes we care about
TARGET_CLASSES = {0: "Person", 2: "Car"}

# Alert settings
PEOPLE_THRESHOLD = 15
alert_cooldown = 0

# Shared state
is_running = False
cap = None
frame_counter = 0

# Unique ID tracking
counted_ids = {"Person": set(), "Car": set()}
total_people = 0
total_cars = 0

# CSV setup
CSV_FILE = "D:\\TrafficDetector\\detection_log.csv"

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "People_Live", "Cars_Live", "Total_People", "Total_Cars", "Alert"])

def log_to_csv(person_count, car_count, total_people, total_cars, alert_status):
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            person_count,
            car_count,
            total_people,
            total_cars,
            alert_status
        ])

def play_alert():
    winsound.Beep(1000, 500)

def get_frame():
    global alert_cooldown, total_people, total_cars, cap, tracker, frame_counter

    if cap is None or not cap.isOpened():
        return None, 0, 0, "Camera not started"

    ret, frame = cap.read()
    if not ret:
        return None, 0, 0, "Frame read failed"

    frame_counter += 1

    results = model(frame, verbose=False)[0]

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if cls_id not in TARGET_CLASSES or conf < 0.4:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, TARGET_CLASSES[cls_id]))

    tracks = tracker.update_tracks(detections, frame=frame)

    person_count = 0
    car_count = 0

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        label = track.det_class
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        if label == "Person":
            person_count += 1
            color = (0, 255, 0)
            if track_id not in counted_ids["Person"]:
                counted_ids["Person"].add(track_id)
                total_people += 1
        elif label == "Car":
            car_count += 1
            color = (0, 0, 255)
            if track_id not in counted_ids["Car"]:
                counted_ids["Car"].add(track_id)
                total_cars += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} #{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Crowd alert
    alert_cooldown = max(0, alert_cooldown - 1)
    if person_count >= PEOPLE_THRESHOLD and alert_cooldown == 0:
        threading.Thread(target=play_alert, daemon=True).start()
        alert_cooldown = 30

    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame, f"People (live): {person_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Cars (live): {car_count}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Total People: {total_people}", (10, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(frame, f"Total Cars: {total_cars}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

    if person_count >= PEOPLE_THRESHOLD:
        cv2.putText(frame, "!! CROWD ALERT !!", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    alert_status = "CROWD ALERT!" if person_count >= PEOPLE_THRESHOLD else "Normal"

    if frame_counter % 10 == 0:
        log_to_csv(person_count, car_count, total_people, total_cars, alert_status)

    return frame_rgb, person_count, car_count, alert_status


def start_camera():
    global cap, is_running, counted_ids, total_people, total_cars, tracker, frame_counter
    counted_ids = {"Person": set(), "Car": set()}
    total_people = 0
    total_cars = 0
    frame_counter = 0
    tracker = DeepSort(max_age=15, n_init=3, nms_max_overlap=0.5)
    cap = cv2.VideoCapture(0)
    is_running = True
    init_csv()
    return "Camera Started!"

def stop_camera():
    global cap, is_running
    is_running = False
    if cap:
        cap.release()
    return "Camera Stopped!"

def stream():
    while is_running:
        result = get_frame()
        if result[0] is not None:
            yield result
        time.sleep(0.05)

def load_dashboard():
    if not os.path.exists(CSV_FILE):
        return None, None, "No data yet! Run the detector first."

    df = pd.read_csv(CSV_FILE, encoding="utf-8")

    if df.empty:
        return None, None, "CSV is empty! Run the detector first."

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # People over time chart
    fig_people = go.Figure()
    fig_people.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["People_Live"],
        mode="lines+markers",
        name="People (live)",
        line=dict(color="green", width=2),
        marker=dict(size=4)
    ))

    # Highlight alert moments
    alerts = df[df["Alert"] == "CROWD ALERT!"]
    if not alerts.empty:
        fig_people.add_trace(go.Scatter(
            x=alerts["Timestamp"],
            y=alerts["People_Live"],
            mode="markers",
            name="Alert",
            marker=dict(color="red", size=10, symbol="x")
        ))

    fig_people.update_layout(
        title="People Count Over Time",
        xaxis_title="Time",
        yaxis_title="People Count",
        template="plotly_dark",
        height=350
    )

    # Cars over time chart
    fig_cars = go.Figure()
    fig_cars.add_trace(go.Scatter(
        x=df["Timestamp"],
        y=df["Cars_Live"],
        mode="lines+markers",
        name="Cars (live)",
        line=dict(color="red", width=2),
        marker=dict(size=4)
    ))
    fig_cars.update_layout(
        title="Car Count Over Time",
        xaxis_title="Time",
        yaxis_title="Car Count",
        template="plotly_dark",
        height=350
    )

    # Summary stats
    summary = f"""
### Session Summary
- **Duration:** {df['Timestamp'].min()} to {df['Timestamp'].max()}
- **Peak People:** {df['People_Live'].max()}
- **Peak Cars:** {df['Cars_Live'].max()}
- **Total Alerts:** {len(alerts)}
- **Total Logs:** {len(df)}
    """

    return fig_people, fig_cars, summary


# Gradio UI
with gr.Blocks(title="Traffic Detector") as app:

    gr.Markdown("# Real-Time People & Car Detector")
    gr.Markdown("YOLOv8 + DeepSORT — Live detection with crowd alerts & analytics")

    with gr.Tabs():

        # Tab 1: Live Detection
        with gr.Tab("Live Detection"):
            with gr.Row():
                start_btn = gr.Button("Start Camera", variant="primary")
                stop_btn = gr.Button("Stop Camera", variant="stop")

            status_box = gr.Textbox(label="Status", value="Click Start to begin")

            with gr.Row():
                with gr.Column(scale=2):
                    output_frame = gr.Image(label="Live Detection Feed")
                with gr.Column(scale=1):
                    gr.Markdown("### Live Stats")
                    people_out = gr.Number(label="People (live)")
                    cars_out = gr.Number(label="Cars (live)")
                    alert_out = gr.Textbox(label="Alert Status", value="Normal")
                    gr.Markdown("---")
                    gr.Markdown(f"Logs saved to: `{CSV_FILE}`")

            start_btn.click(fn=start_camera, outputs=status_box).then(
                fn=stream,
                outputs=[output_frame, people_out, cars_out, alert_out]
            )
            stop_btn.click(fn=stop_camera, outputs=status_box)

        # Tab 2: Analytics Dashboard
        with gr.Tab("Analytics Dashboard"):
            gr.Markdown("### Detection Analytics")
            gr.Markdown("Click **Refresh** after running the detector to see your data.")

            refresh_btn = gr.Button("Refresh Dashboard", variant="primary")
            summary_out = gr.Markdown("Click Refresh to load data.")

            with gr.Row():
                people_chart = gr.Plot(label="People Over Time")
                cars_chart = gr.Plot(label="Cars Over Time")

            refresh_btn.click(
                fn=load_dashboard,
                outputs=[people_chart, cars_chart, summary_out]
            )

app.launch(theme=gr.themes.Soft())
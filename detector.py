import cv2
import time
import threading
import winsound
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("yolov8n.pt")

# ✅ Initialize DeepSORT tracker
tracker = DeepSort(max_age=15, n_init=3,nms_max_overlap=0.5)

# Classes we care about
TARGET_CLASSES = {0: "Person", 2: "Car"}

# Open webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

# FPS tracking
prev_time = time.time()

# ✅ Unique ID tracking
counted_ids = {"Person": set(), "Car": set()}
total_people = 0
total_cars = 0

# Alert settings
PEOPLE_THRESHOLD = 15
alert_cooldown = 0

def play_alert():
    winsound.Beep(1000, 500)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame, verbose=False)[0]

    # ✅ Prepare detections for DeepSORT
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id not in TARGET_CLASSES or conf < 0.4:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        detections.append(([x1, y1, w, h], conf, TARGET_CLASSES[cls_id]))

    # ✅ Update tracker
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
            # ✅ Count only new unique IDs
            if track_id not in counted_ids["Person"]:
                counted_ids["Person"].add(track_id)
                total_people += 1
        elif label == "Car":
            car_count += 1
            color = (0, 0, 255)
            if track_id not in counted_ids["Car"]:
                counted_ids["Car"].add(track_id)
                total_cars += 1

        # Draw box + ID label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} #{track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Crowd alert
    alert_cooldown = max(0, alert_cooldown - 1)
    if person_count >= PEOPLE_THRESHOLD and alert_cooldown == 0:
        threading.Thread(target=play_alert, daemon=True).start()
        alert_cooldown = 30

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Semi-transparent panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # Live counts
    cv2.putText(frame, f"People (live): {person_count}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Cars (live): {car_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Cumulative counts (unique IDs)
    cv2.putText(frame, f"Total People: {total_people}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    cv2.putText(frame, f"Total Cars: {total_cars}", (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)

    # Crowd alert text
    if person_count >= PEOPLE_THRESHOLD:
        cv2.putText(frame, f"!! CROWD ALERT: {person_count} PEOPLE !!", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Traffic Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
from ultralytics import YOLO
import cv2
import time
from collections import defaultdict

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- FPS ----------------
class FPSCounter:
    def __init__(self):
        self.prev_time = 0

    def get_fps(self):
        curr = time.time()
        fps = 1 / (curr - self.prev_time) if self.prev_time else 0
        self.prev_time = curr
        return int(fps)

# ---------------- TRACKING MEMORY ----------------
track_history = defaultdict(list)

# ---------------- COLORS ----------------
def get_color(label):
    if label == "person":
        return (0,255,0)
    elif label == "car":
        return (255,0,0)
    return (255,255,255)

# ---------------- ZONE ----------------
def draw_zone(frame):
    h, w = frame.shape[:2]
    zone = [(int(w*0.2), int(h*0.2)), (int(w*0.8), int(h*0.8))]
    cv2.rectangle(frame, zone[0], zone[1], (0,255,255), 2)
    return zone

# ---------------- CORE PROCESS ----------------
def process_frame(frame, confidence=0.5):

    results = model.track(frame, conf=confidence, persist=True)

    boxes = results[0].boxes
    names = results[0].names

    counts = {}
    alerts = []
    zone = draw_zone(frame)

    for box in boxes:
        cls = int(box.cls[0])
        label = names[cls]

        counts[label] = counts.get(label, 0) + 1

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1+x2)//2, (y1+y2)//2

        color = get_color(label)

        # Draw box
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
        cv2.putText(frame, label, (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

        # Zone detection
        if zone[0][0] < cx < zone[1][0] and zone[0][1] < cy < zone[1][1]:
            alerts.append(f"{label} in zone")

    # Behavior detection
    behavior = "Normal"
    if counts.get("person",0) > 5:
        behavior = "⚠ Crowd detected"
    if counts.get("person",0) > 2 and counts.get("person",0) < 5:
        behavior = "Group activity"

    return frame, counts, behavior, alerts
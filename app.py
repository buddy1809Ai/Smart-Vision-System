# ============================================================
#   🤖 AI VISION ANALYTICS SYSTEM 3.0 — Ultimate Pro Edition
# ============================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import time
import pandas as pd
import psutil
import pyttsx3
import threading
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pymongo import MongoClient
import tempfile
import os
import torch
import platform
import json

# --------------------------------------------------------------
# Streamlit Page Setup
# --------------------------------------------------------------
st.set_page_config(page_title="AI Vision Analytics System 3.0",
                   layout="wide",
                   page_icon="🤖")

st.markdown("""
<style>
body {
    background: linear-gradient(140deg,#0f2027,#203a43,#2c5364);
    color: white;
}
h1,h2,h3 {
    color:#00FFC6;
    text-align:center;
}
.stButton>button {
    background: linear-gradient(90deg,#00FFC6,#007BFF);
    color:black;
    font-weight:bold;
    border-radius:10px;
}
div[data-testid="stMetricValue"], .stProgress > div > div {
    color:#00FFD1;
}
.main {
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Vision Analytics Dashboard 3.0")
st.markdown("A futuristic real-time object detection, analytics & monitoring suite enriched with advanced features and full resilience.")

# --------------------------------------------------------------
# Sidebar Controls
# --------------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")
model_choice = st.sidebar.selectbox("Select YOLO Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"])
confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)
source_mode = st.sidebar.radio("Detection Source", ["📸 Image", "🎥 Webcam", "🎞️ Video", "📹 Multi-Cam"])
camera_index = st.sidebar.number_input("Camera Index", 0, 3, step=1)
voice_enabled = st.sidebar.toggle("Enable Voice Alerts", True)
save_detections = st.sidebar.toggle("Auto-Save Detections", True)
log_to_mongo = st.sidebar.toggle("Enable MongoDB Logging", False)
selected_classes = st.sidebar.multiselect("🎯 Filter Classes", ["person", "car", "dog", "cat", "bicycle", "motorbike", "truck"])
export_format = st.sidebar.radio("📦 Export Format", ["CSV", "JSON", "Both"])
st.sidebar.markdown("---")

# --------------------------------------------------------------
# Voice Engine
# --------------------------------------------------------------
engine = None
try:
    engine = pyttsx3.init()
    voice_speed = st.sidebar.slider("Voice Speed", 100, 250, 170)
    engine.setProperty("rate", voice_speed)
except Exception:
    voice_enabled = False
    st.sidebar.warning("⚠️ Voice engine unavailable")

def speak(text):
    if not voice_enabled or engine is None:
        return
    def _run():
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
    threading.Thread(target=_run).start()

# --------------------------------------------------------------
# MongoDB Init
# --------------------------------------------------------------
mongo_client = None
collection = None
if log_to_mongo:
    try:
        mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
        mongo_client.server_info()  # Test connection
        db = mongo_client["vision_analytics"]
        collection = db["detections"]
        st.sidebar.success("✅ MongoDB Connected")
    except Exception as e:
        st.sidebar.error(f"❌ MongoDB Error: {e}")
        collection = None
        log_to_mongo = False

# --------------------------------------------------------------
# Memory State
# --------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "analytics" not in st.session_state:
    st.session_state.analytics = []

# --------------------------------------------------------------
# Load YOLO Model
# --------------------------------------------------------------
with st.spinner(f"🔄 Loading {model_choice}..."):
    model = YOLO(model_choice)
speak("AI Vision Analytics System version 3 initialized successfully")

# --------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------
def extract_counts(results):
    boxes = results[0].boxes
    names = results[0].names
    counts = {}
    for box in boxes:
        cls = int(box.cls[0])
        label = names[cls]
        counts[label] = counts.get(label, 0) + 1
    return counts

def detect_anomalies(counts):
    if counts.get("person", 0) > 8:
        return "⚠️ Crowd Alert"
    elif counts.get("car", 0) > 15:
        return "🚨 Heavy Traffic"
    return "✅ Normal Condition"

def log_to_database(entry):
    # FIX: Use 'is not None' instead of boolean check on collection
    if log_to_mongo and collection is not None:
        try:
            collection.insert_one(entry)
        except Exception:
            pass

def system_monitor():
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    gpu = 0.0
    try:
        if torch.cuda.is_available():
            gpu = torch.cuda.memory_allocated() / 1024 ** 3
    except Exception:
        pass
    gpu_temp = None
    try:
        temp = psutil.sensors_temperatures()
        if temp and "coretemp" in temp and len(temp["coretemp"]) > 0:
            gpu_temp = temp["coretemp"][0].current
    except Exception:
        pass
    return cpu, ram, gpu, gpu_temp

# --------------------------------------------------------------
# Image Detection Mode
# --------------------------------------------------------------
def image_mode():
    st.subheader("📸 Image Detection Mode")
    upload = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if upload:
        img = Image.open(upload).convert("RGB")
        img_np = np.array(img)
        results = model(img_np, conf=confidence)
        annotated = results[0].plot()
        counts = extract_counts(results)

        if selected_classes:
            counts = {k: v for k, v in counts.items() if k in selected_classes}

        st.image(annotated, caption="Detected Objects", use_container_width=True)
        detected_summary = pd.DataFrame(list(counts.items()), columns=["Object", "Count"])
        st.dataframe(detected_summary, use_container_width=True)

        condition = detect_anomalies(counts)
        cpu, ram, gpu, temp = system_monitor()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Condition", condition)
        col2.metric("CPU Usage", f"{cpu}%")
        col3.metric("Memory Usage", f"{ram}%")
        col4.metric("GPU Memory", f"{gpu:.2f} GB")

        if "Alert" in condition:
            speak(condition)

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "image",
            "counts": counts,
            "condition": condition,
            "cpu": cpu,
            "ram": ram,
            "gpu": gpu
        }
        st.session_state.history.append(record)
        st.session_state.analytics.append(record)
        log_to_database(record)

        # Save & Download
        if save_detections:
            try:
                temp_path = os.path.join(tempfile.gettempdir(), "annotated.jpg")
                cv2.imwrite(temp_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                with open(temp_path, "rb") as file:
                    st.download_button("⬇ Download Annotated Image", data=file, file_name="annotated_result.jpg", mime="image/jpeg")
            except Exception:
                pass

# --------------------------------------------------------------
# Webcam Detection Mode
# --------------------------------------------------------------
def webcam_mode():
    st.subheader("🎥 Live Webcam Detection")
    start = st.button("▶ Start Detection")
    stop_button = st.button("⏹ Stop Detection")
    frame_placeholder = st.empty()
    chart_placeholder = st.empty()
    detection_trend = []

    if start and not stop_button:
        cap = cv2.VideoCapture(int(camera_index))
        if not cap.isOpened():
            st.error("❌ Cannot open camera. Check camera index.")
            return
        
        frame_count = 0
        max_frames = 500  # Safety limit
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to read frame")
                break
            
            frame_count += 1
            results = model(frame, conf=confidence)
            annotated = results[0].plot()
            counts = extract_counts(results)
            
            if selected_classes:
                counts = {k: v for k, v in counts.items() if k in selected_classes}

            condition = detect_anomalies(counts)
            cpu, ram, gpu, temp = system_monitor()
            detection_trend.append(sum(counts.values()))

            st.sidebar.metric("CPU", f"{cpu}%")
            st.sidebar.metric("RAM", f"{ram}%")
            st.sidebar.metric("GPU", f"{gpu:.2f} GB")
            frame_placeholder.image(annotated, channels="BGR")
            
            if len(detection_trend) > 1:
                chart_placeholder.line_chart(pd.DataFrame(detection_trend, columns=["Detections"]))

            log_to_database({"time": time.strftime("%H:%M:%S"), "counts": counts, "condition": condition})
            
            if "Alert" in condition:
                speak(condition)

            time.sleep(0.03)  # ~30 FPS cap
            
        cap.release()
        speak("Webcam detection stopped")

# --------------------------------------------------------------
# Video Detection Mode
# --------------------------------------------------------------
def video_mode():
    st.subheader("🎞️ Video Detection Mode")
    upload = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])
    if upload:
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(upload.read())
        temp_video.close()
        
        cap = cv2.VideoCapture(temp_video.name)
        if not cap.isOpened():
            st.error("❌ Cannot open video file")
            return
            
        frame_placeholder = st.empty()
        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            frame_count = 1
        idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            results = model(frame, conf=confidence)
            annotated = results[0].plot()
            frame_placeholder.image(annotated, channels="BGR", use_container_width=True)
            counts = extract_counts(results)
            progress.progress(min(idx / frame_count, 1.0))
            
        cap.release()
        try:
            os.unlink(temp_video.name)
        except Exception:
            pass
        speak("Video processing complete")

# --------------------------------------------------------------
# Multi-Camera Mode (Simulated Dual Feed)
# --------------------------------------------------------------
def multi_cam_mode():
    st.subheader("📹 Multi-Camera Detection Mode (Simulated)")
    cams = [0, 1]
    columns = st.columns(len(cams))
    for i, cam_index_local in enumerate(cams):
        columns[i].markdown(f"### 🎥 Camera {cam_index_local}")
        try:
            cap = cv2.VideoCapture(cam_index_local)
            ret, frame = cap.read()
            if ret:
                results = model(frame, conf=confidence)
                annotated = results[0].plot()
                columns[i].image(annotated, channels="BGR")
            else:
                columns[i].warning(f"Camera {cam_index_local} not available")
            cap.release()
        except Exception as e:
            columns[i].error(f"Error: {e}")

# --------------------------------------------------------------
# Mode Handling
# --------------------------------------------------------------
if source_mode == "📸 Image":
    image_mode()
elif source_mode == "🎥 Webcam":
    webcam_mode()
elif source_mode == "🎞️ Video":
    video_mode()
else:
    multi_cam_mode()

# --------------------------------------------------------------
# Analytics / Logs
# --------------------------------------------------------------
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.subheader("📊 Detection Logs & Analytics Dashboard")
    col1, col2 = st.columns(2)
    col1.dataframe(df, use_container_width=True)

    x_vals = list(range(len(df)))
    y_vals = [sum(v for v in rec["counts"].values()) for rec in st.session_state.history]
    fig = px.bar(x=x_vals, y=y_vals, labels={'x': 'Frame Index', 'y': 'Total Objects'})
    col2.plotly_chart(fig, use_container_width=True)

    # Export Logs
    csv_data = df.to_csv(index=False).encode()
    json_data = json.dumps(st.session_state.history, indent=4).encode()

    if export_format in ["CSV", "Both"]:
        st.download_button("⬇ Download Logs (CSV)", data=csv_data,
                           file_name="analytics_logs.csv", mime="text/csv")
    if export_format in ["JSON", "Both"]:
        st.download_button("⬇ Download Logs (JSON)", data=json_data,
                           file_name="analytics_logs.json", mime="application/json")

# --------------------------------------------------------------
# Footer
# --------------------------------------------------------------
st.markdown("""
---
#### 💡 Pro Tips & Features
- Load **`yolov8l.pt`** for best accuracy (requires GPU)
- Enable **MongoDB logging** for permanent analytics storage  
- **Voice alerts** automatically trigger for crowd or traffic anomalies  
- Download logs in CSV/JSON and use Plotly Dashboard for deeper trends  
---
⚙️ Running on: **{} | Python {}**  
Torch device: **{}**  
""".format(platform.system(), platform.python_version(), "GPU" if torch.cuda.is_available() else "CPU"))










 
# ============================================================
#   🤖 AI VISION ANALYTICS SYSTEM — DEPLOYABLE VERSION
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image

# ✅ ONLY ONCE
st.set_page_config(page_title="Smart Vision System", layout="wide")

st.title("🤖 AI Vision Analytics System 🚀")
st.success("App deployed successfully!")

# --------------------------------------------------------------
# SIDEBAR (SAFE)
# --------------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select YOLO Model",
    ["yolov8n.pt", "yolov8s.pt"]
)

confidence = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)

source_mode = st.sidebar.radio(
    "Detection Source",
    ["📸 Image", "🎥 Webcam"]
)

# --------------------------------------------------------------
# LAZY LOAD YOLO (IMPORTANT)
# --------------------------------------------------------------
@st.cache_resource
def load_model(model_name):
    from ultralytics import YOLO
    return YOLO(model_name)

model = load_model(model_choice)

# --------------------------------------------------------------
# IMAGE MODE (WORKING)
# --------------------------------------------------------------
def image_mode():
    st.subheader("📸 Image Detection")

    uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)

        results = model(img_np, conf=confidence)
        annotated = results[0].plot()

        st.image(annotated, caption="Detection Result", use_container_width=True)

# --------------------------------------------------------------
# WEBCAM MODE (LIMITED SUPPORT)
# --------------------------------------------------------------
def webcam_mode():
    st.subheader("🎥 Webcam Detection")

    st.warning("⚠️ Webcam may not work on Streamlit Cloud")

    if st.button("Start Camera"):
        import cv2

        cap = cv2.VideoCapture(0)

        frame_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence)
            annotated = results[0].plot()

            frame_placeholder.image(annotated, channels="BGR")

        cap.release()

# --------------------------------------------------------------
# MODE HANDLER
# --------------------------------------------------------------
if source_mode == "📸 Image":
    image_mode()
else:
    webcam_mode()

# --------------------------------------------------------------
# PLACEHOLDER FEATURES (NOT REMOVED)
# --------------------------------------------------------------
st.markdown("---")

st.subheader("📊 Upcoming Features")

st.info("""
🔜 MongoDB Logging  
🔜 Voice Alerts  
🔜 Analytics Dashboard  
🔜 Multi-Camera Support  
""")

# --------------------------------------------------------------
# FOOTER
# --------------------------------------------------------------
st.markdown("""
---
💡 Built with Streamlit + YOLOv8  
⚡ Optimized for deployment  
""")










 
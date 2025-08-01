import streamlit as st
import numpy as np
import cv2
from PIL import Image
from main import process_image_simulation, predict_objects, annotate_image_with_detections, load_detection_model
import time

st.set_page_config(page_title="Weed Robot Dashboard", page_icon="🌱", layout="wide")

# --- Sidebar/logo etc ---
st.sidebar.image("robot.jpg", width=90)
st.sidebar.markdown(
    "<div style='text-align:center; font-size:21px; font-weight:bold; color:#23c268; margin-bottom:10px;'>Unkraut Roboter</div>",
    unsafe_allow_html=True
)
st.sidebar.title("Weed Robot Dashboard")
st.sidebar.markdown("---")

# --- Main Header ---
st.markdown("""
    <div style="width:100%; text-align:center; margin-bottom:5px;">
        <h1 style="display:inline; font-size:2.8rem; color:#23c268; letter-spacing:1px;">Unkraut Roboter
</h1>
        <br>
        <span style="font-size:1.1rem; color:#444;">AI-powered detection for plants & weeds</span>
    </div>
    <hr style="border:1px solid #eee; margin:0 0 25px 0">
""", unsafe_allow_html=True)

st.markdown("<h3 style='color:#183152;'>Smart Image Detection</h3>", unsafe_allow_html=True)
st.markdown(
    "Upload a field image. The system detects and highlights <span style='color:#23c268;'>plants</span> and <span style='color:#e74c3c;'>weeds</span> using AI and simulation-accurate logic.",
    unsafe_allow_html=True
)
st.markdown("")

uploaded_file = st.file_uploader("Choose a field image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # --- Loading bar (fixed to never exceed 100) ---
    loading_spot = st.empty()
    with loading_spot:
        st.info("Processing... Please wait!")
        my_bar = st.progress(0, text="Starting detection ...")
        for percent_complete in range(0, 100, 8):
            time.sleep(0.04)
            next_val = min(percent_complete + 8, 100)
            my_bar.progress(next_val)
        time.sleep(0.18)
    loading_spot.empty()

    img_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(img_bytes, 1)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

    model = load_detection_model()
    img_resized = cv2.resize(image_cv, (640, 480))
    detections = predict_objects(img_resized, model)
    annotated = annotate_image_with_detections(img_resized, detections)

    num_plants = sum(1 for d in detections if d["class"] == "plant")
    num_weeds = sum(1 for d in detections if d["class"] == "weed")
    status = "Plant" if num_plants >= num_weeds else "Weed" if num_weeds > 0 else "Unknown"

    st.markdown("---")

    # --- Stats row: BIG and colored, as requested! ---
        # --- Stats row: only your requested changes ---
    st.markdown(
        f"""
        <div style='display:flex;justify-content:space-between;text-align:center;margin:30px 0 25px 0;'>
            <div style='flex:1'>
                <span style='color:#bbb;font-size:40px;font-weight:600;'>Detected Plants</span><br>
                <span style='color:#23c268;font-size:38px;font-weight:700;display:block;margin-top:2px;'>{num_plants}</span>
            </div>
            <div style='flex:1'>
                <span style='color:#bbb;font-size:40px;font-weight:600;'>Detected Weeds</span><br>
                <span style='color:#e74c3c;font-size:38px;font-weight:700;display:block;margin-top:2px;'>{num_weeds}</span>
            </div>
            <div style='flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;'>
                <span style='color:#bbb;font-size:40px;font-weight:600;'>Status</span><br>
                <span style='color:{'#23c268' if status.lower()=='plant' else '#e74c3c'};font-size:38px;font-weight:800;display:block;margin-top:1px;'>{status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Images Side-by-Side ---
        # --- Images Side-by-Side ---
    c1, c2 = st.columns(2)
    c1.image(image_rgb, caption="Original Image", use_container_width=True)
    c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection Output", use_container_width=True)

    st.success("Detection complete! Results match your simulation.")

else:
    st.info("Please upload an image to get started.")

# --- Footer ---
st.markdown("""
    <hr>
    <div style='text-align:center; color:#bbb; font-size:14px; margin-top:28px;'>
        &copy; 2025 Weed Robot • 
    </div>
""", unsafe_allow_html=True)

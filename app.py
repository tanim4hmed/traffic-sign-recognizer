import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Traffic Sign Recognizer", page_icon="🚦", layout="wide")

@st.cache_resource
def load_traffic_model():
    return tf.keras.models.load_model("gtsrb_model.keras")

model = load_traffic_model()

LABELS = [
    "20 km/h Limit", "30 km/h Limit", "50 km/h Limit", "60 km/h Limit",
    "70 km/h Limit", "80 km/h Limit", "End 80 km/h Limit", "100 km/h Limit",
    "120 km/h Limit", "No Overtaking", "No Overtaking (Trucks)", "Right of Way Next Crossroads",
    "Priority Road", "Yield / Give Way", "Stop Sign", "No Vehicles",
    "No Trucks", "No Entry / One-Way", "General Hazard Warning", "Left Curve Ahead",
    "Right Curve Ahead", "Double Curve", "Bumpy Road", "Slippery Road",
    "Road Narrows on Right", "Roadworks Ahead", "Traffic Signals Ahead", "Pedestrian Crossing",
    "Children / School Zone", "Bicycle Crossing", "Icy / Snowy Road", "Wild Animals Crossing",
    "End All Speed & Passing Limits", "Turn Right Ahead", "Turn Left Ahead", "Ahead Only",
    "Go Straight or Right", "Go Straight or Left", "Keep Right", "Keep Left",
    "Roundabout Mandatory", "End of No Overtaking", "End No Overtaking (Trucks)"
]

st.title("🚦 AI Traffic Sign Recognition Dashboard")
st.write("Upload any road or traffic sign photo to see how accurately the model recognizes it.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Traffic Sign Image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Traffic Sign", use_container_width=True)

with col2:
    if uploaded_file:
        resized = image.resize((64, 64))
        arr = np.expand_dims(np.array(resized), axis=0)
        preds = model.predict(arr, verbose=0)[0]
        
        top_5_idx = np.argsort(preds)[::-1][:5]
        best_label = LABELS[top_5_idx[0]]
        best_conf = float(preds[top_5_idx[0]]) * 100
        
        st.subheader(f"🎯 Prediction: **{best_label}**")
        st.metric(label="Recognition Confidence", value=f"{best_conf:.1f}%")
        
        st.write("#### Top 5 Matches:")
        for idx in top_5_idx:
            pct = float(preds[idx])
            st.progress(pct, text=f"{LABELS[idx]} — {pct*100:.1f}%")
    else:
        st.info("👈 Upload an image on the left to view recognition confidence scores.")

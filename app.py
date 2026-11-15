import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

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

def get_sign_category_and_advice(idx):
    if idx in [0, 1, 2, 3, 4, 5, 7, 8]:
        return "🔴 Speed Limit Sign", "Maintain vehicle speed at or below the posted limit."
    elif idx in [9, 10, 14, 15, 16, 17]:
        return "⛔ Prohibitory / Mandatory Stop", "Strict road restriction. Obey right-of-way and entry rules immediately."
    elif idx in [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]:
        return "⚠️ Hazard Warning Sign", "Slow down and stay alert for upcoming road hazards or crossings."
    elif idx in [33, 34, 35, 36, 37, 38, 39, 40]:
        return "🔵 Mandatory Direction Sign", "You must follow the arrow direction indicated on the sign."
    else:
        return "ℹ️ Priority / End of Restriction", "Standard highway priority or end of previous speed/passing zone."

def check_out_of_distribution(pil_img, preds):
    """Detects novelty yellow diamond signs or high-uncertainty random photos."""
    arr = np.array(pil_img.resize((64, 64))).astype(int)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    
    # Detect solid yellow US/novelty signs (like the mosquito sign)
    yellow_pixels = (r > 150) & (g > 120) & (b < 80) & (np.abs(r - g) < 60)
    red_pixels = (r > 150) & (g < 80) & (b < 80)
    if np.mean(yellow_pixels) > 0.22 and np.mean(red_pixels) < 0.05:
        if np.argmax(preds) != 12:  # Class 12 is Priority Road (yellow diamond with white border)
            return True, "Solid Yellow Diamond Sign detected (US / Novelty sign not in European GTSRB dataset)."
            
    # Calculate Shannon Entropy of predictions (high entropy = model is guessing blindly)
    entropy = -np.sum(preds * np.log(preds + 1e-10))
    if entropy > 1.35 or np.max(preds) < 0.60:
        return True, "Low visual match or high uncertainty across the 43 trained classes."
        
    return False, ""

st.title("🚦 AI Traffic Sign Recognition Dashboard")
st.markdown("Upload an image or take a photo. **Drag the crop box** tightly around the sign (excluding text labels or wide backgrounds) for maximum accuracy.")

with st.expander("📋 Click to view all 43 Traffic Signs supported by this model"):
    st.write(", ".join([f"**{i+1}.** {name}" for i, name in enumerate(LABELS)]))

col1, col2 = st.columns([1.1, 1])

with col1:
    input_mode = st.radio("Select Input Source:", ["📁 Upload Image File", "📷 Use Camera"], horizontal=True)
    
    if input_mode == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Traffic Sign Photo", type=["png", "jpg", "jpeg", "webp"])
    else:
        uploaded_file = st.camera_input("Take a photo of a traffic sign")

    enable_cropper = st.checkbox("✂️ Enable Interactive Cropper (Crop out bottom text & background)", value=True)

    final_img = None
    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        if enable_cropper:
            st.caption("👇 Drag the corners of the red box to frame ONLY the traffic sign:")
            final_img = st_cropper(raw_img, realtime_update=True, box_color="#FF4B4B", aspect_ratio=(1, 1))
        else:
            st.image(raw_img, caption="Uploaded Image", use_container_width=True)
            final_img = raw_img

with col2:
    if final_img is not None:
        st.write("#### 🔍 Sign Sent to Neural Network (64×64):")
        st.image(final_img.resize((120, 120)), width=120)
        
        resized = final_img.resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.array(resized), axis=0)
        preds = model.predict(arr, verbose=0)[0]
        
        top_5_idx = np.argsort(preds)[::-1][:5]
        best_idx = int(top_5_idx[0])
        best_label = LABELS[best_idx]
        best_conf = float(preds[best_idx]) * 100
        
        is_unknown, reason = check_out_of_distribution(final_img, preds)
        
        if is_unknown:
            st.error("⚠️ **Unrecognized / Out-of-Dataset Image**")
            st.write(f"**Reason:** {reason}")
            st.caption(f"Closest visual pattern in dataset: *{best_label}* ({best_conf:.1f}%)")
        else:
            category, advice = get_sign_category_and_advice(best_idx)
            st.success(f"### 🎯 Prediction: **{best_label}**")
            
            m1, m2 = st.columns(2)
            m1.metric("Recognition Confidence", f"{best_conf:.1f}%")
            m2.metric("Sign Category", category.split()[1])
            
            st.info(f"**{category}**\n\n**Driver Action:** {advice}")
            
        st.write("#### 📊 Top 5 Probability Breakdown:")
        for idx in top_5_idx:
            pct = float(preds[idx])
            st.progress(pct, text=f"{LABELS[idx]} — {pct*100:.1f}%")
    else:
        st.info("👈 Upload an image on the left to start analyzing.")

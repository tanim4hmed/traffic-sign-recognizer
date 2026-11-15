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

def is_not_recognizable(pil_img, preds, min_confidence):
    """Returns (True, Reason) if the image is not recognizable as one of the 43 GTSRB signs."""
    arr = np.array(pil_img.resize((64, 64))).astype(int)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 1. Check for standard traffic sign colors (Red, Blue, or Yellow)
    red_mask = (r > 135) & (r - g > 40) & (r - b > 40)
    blue_mask = (b > 110) & (b - r > 35) & (b - g > 15)
    yellow_mask = (r > 150) & (g > 120) & (b < 85) & (np.abs(r - g) < 60)

    red_ratio = np.mean(red_mask)
    blue_ratio = np.mean(blue_mask)
    yellow_ratio = np.mean(yellow_mask)

    best_idx = int(np.argmax(preds))
    best_conf = float(preds[best_idx]) * 100

    # Reject solid yellow US/novelty diamond signs (only Class 12 "Priority Road" has yellow in GTSRB)
    if yellow_ratio > 0.20 and red_ratio < 0.05 and best_idx != 12:
        return True, "This image uses a solid yellow sign format that does not exist in the 43 trained classes."

    # Reject random photos/objects that lack red, blue, or yellow sign colors
    # (Except black-and-white end-of-limit signs: classes 6, 32, 41, 42)
    if best_idx not in [6, 32, 41, 42]:
        if (red_ratio + blue_ratio + yellow_ratio) < 0.03:
            return True, "No valid traffic sign color pattern (red/blue/yellow border) detected in the image."

    # 2. Check if top confidence meets the required threshold
    if best_conf < min_confidence:
        return True, f"Recognition score ({best_conf:.1f}%) is below the {min_confidence}% minimum threshold."

    # 3. Check if the model is confused between its top 2 guesses
    sorted_preds = np.sort(preds)[::-1]
    margin = (sorted_preds[0] - sorted_preds[1]) * 100
    if margin < 25.0:
        return True, "The model is split between multiple classes and cannot recognize this image reliably."

    return False, ""

st.title("🚦 AI Traffic Sign Recognition Dashboard")
st.markdown("Upload a traffic sign image. If the image is outside the 43 trained classes or unclear, the app will display **Not Recognizable**.")

with st.expander("📋 Click to view all 43 Traffic Signs supported by this model"):
    st.write(", ".join([f"**{i+1}.** {name}" for i, name in enumerate(LABELS)]))

col1, col2 = st.columns([1.1, 1])

with col1:
    input_mode = st.radio("Select Input Source:", ["📁 Upload Image File", "📷 Use Camera"], horizontal=True)
    
    if input_mode == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload Traffic Sign Photo", type=["png", "jpg", "jpeg", "webp"])
    else:
        uploaded_file = st.camera_input("Take a photo of a traffic sign")

    enable_cropper = st.checkbox("✂️ Enable Interactive Cropper (Crop tightly around the sign)", value=True)
    min_conf_slider = st.slider("Minimum Confidence Required (%)", min_value=50, max_value=95, value=75, step=5)

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
        st.write("#### 🔍 Cropped Sign Sent to AI (64×64):")
        st.image(final_img.resize((120, 120)), width=120)
        
        resized = final_img.resize((64, 64), Image.Resampling.BILINEAR)
        arr = np.expand_dims(np.array(resized), axis=0)
        preds = model.predict(arr, verbose=0)[0]
        
        top_5_idx = np.argsort(preds)[::-1][:5]
        best_idx = int(top_5_idx[0])
        best_label = LABELS[best_idx]
        best_conf = float(preds[best_idx]) * 100
        
        not_recog, reason = is_not_recognizable(final_img, preds, min_conf_slider)
        
        if not_recog:
            st.error("## 🚫 Not Recognizable")
            st.warning(f"**Why:** {reason}")
            st.caption("Please upload a clear photo of one of the 43 supported traffic signs and crop out any extra background or text.")
            with st.expander("See raw internal probabilities (for debugging)"):
                for idx in top_5_idx:
                    pct = float(preds[idx])
                    st.progress(pct, text=f"{LABELS[idx]} — {pct*100:.1f}%")
        else:
            category, advice = get_sign_category_and_advice(best_idx)
            st.success(f"## ✅ Recognized: **{best_label}**")
            
            m1, m2 = st.columns(2)
            m1.metric("Recognition Confidence", f"{best_conf:.1f}%")
            m2.metric("Sign Type", category.split()[1])
            
            st.info(f"**{category}**\n\n**Driver Action:** {advice}")
            
            st.write("#### 📊 Top 5 Probability Breakdown:")
            for idx in top_5_idx:
                pct = float(preds[idx])
                st.progress(pct, text=f"{LABELS[idx]} — {pct*100:.1f}%")
    else:
        st.info("👈 Upload an image on the left to start analyzing.")

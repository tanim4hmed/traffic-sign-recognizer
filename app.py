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

def smart_crop_white_border(pil_img):
    """Only trims pure white borders/text on web icons; leaves real camera photos untouched."""
    arr = np.array(pil_img)
    # Check if the corners of the image are pure white (like a screenshot with text)
    corners = [arr[0, 0], arr[0, -1], arr[-1, 0], arr[-1, -1]]
    is_white_bg = all(np.mean(c) > 240 for c in corners)
    if not is_white_bg:
        return pil_img

    max_rgb = np.max(arr, axis=2).astype(int)
    min_rgb = np.min(arr, axis=2).astype(int)
    color_mask = (max_rgb - min_rgb) > 30
    if np.sum(color_mask) > 100:
        coords = np.argwhere(color_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        h, w = y_max - y_min, x_max - x_min
        pad = int(max(h, w) * 0.08)
        y_min, y_max = max(0, y_min - pad), min(arr.shape[0], y_max + pad)
        x_min, x_max = max(0, x_min - pad), min(arr.shape[1], x_max + pad)
        return pil_img.crop((x_min, y_min, x_max, y_max))
    return pil_img

def is_not_recognizable(pil_img, preds, min_confidence):
    """Simple, reliable check that doesn't block valid dark or faded traffic signs."""
    best_idx = int(np.argmax(preds))
    best_conf = float(preds[best_idx]) * 100

    # 1. Only reject solid yellow US/novelty diamond signs (like the mosquito sign)
    arr = np.array(pil_img.resize((64, 64))).astype(int)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    yellow_mask = (r > 160) & (g > 130) & (b < 70) & (np.abs(r - g) < 50)
    red_mask = (r > 150) & (g < 80) & (b < 80)
    if np.mean(yellow_mask) > 0.25 and np.mean(red_mask) < 0.03 and best_idx != 12:
        return True, "Solid yellow diamond sign detected (not part of the 43 GTSRB classes)."

    # 2. Reject if model confidence is lower than the slider threshold
    if best_conf < min_confidence:
        return True, f"Model confidence ({best_conf:.1f}%) is below the {min_confidence}% threshold."

    return False, ""

st.title("🚦 AI Traffic Sign Recognition Dashboard")
st.write("Upload any road or traffic sign photo to see how accurately the model recognizes it.")

with st.expander("📋 Click to view all 43 Traffic Signs supported by this model"):
    st.write(", ".join([f"**{i+1}.** {name}" for i, name in enumerate(LABELS)]))

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Traffic Sign Photo", type=["png", "jpg", "jpeg", "webp"])
    min_conf_slider = st.slider(
        "Minimum Confidence Required to Recognize (%)",
        min_value=30, max_value=90, value=55, step=5,
        help="If the model's top score is below this percentage, it will display 'Not Recognizable'."
    )

    if uploaded_file:
        raw_img = Image.open(uploaded_file).convert("RGB")
        final_img = smart_crop_white_border(raw_img)
        st.image(raw_img, caption="Uploaded Traffic Sign", use_container_width=True)

with col2:
    if uploaded_file:
        resized = final_img.resize((64, 64))
        arr = np.expand_dims(np.array(resized), axis=0)
        preds = model.predict(arr, verbose=0)[0]
        
        top_5_idx = np.argsort(preds)[::-1][:5]
        best_idx = int(top_5_idx[0])
        best_label = LABELS[best_idx]
        best_conf = float(preds[best_idx]) * 100
        
        not_recog, reason = is_not_recognizable(final_img, preds, min_conf_slider)
        
        if not_recog:
            st.error("## 🚫 Not Recognizable")
            st.warning(f"**Reason:** {reason}")
            st.caption(f"Closest guess was **{best_label}** ({best_conf:.1f}%), which did not pass verification.")
        else:
            category, advice = get_sign_category_and_advice(best_idx)
            st.success(f"## 🎯 Prediction: **{best_label}**")
            st.metric("Recognition Confidence", f"{best_conf:.1f}%")
            st.info(f"**{category}**\n\n**Driver Action:** {advice}")
            
        st.write("#### 📊 Top 5 Matches:")
        for idx in top_5_idx:
            pct = float(preds[idx])
            st.progress(pct, text=f"{LABELS[idx]} — {pct*100:.1f}%")
    else:
        st.info("👈 Upload an image on the left to view recognition confidence scores.")

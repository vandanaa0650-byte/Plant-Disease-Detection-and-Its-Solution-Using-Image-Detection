import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Plant AI Dashboard", layout="wide")

# 🌿 Background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #e6ffe6, #ccffcc);
    }
    </style>
""", unsafe_allow_html=True)

# 🌈 Header
st.markdown("""
    <h1 style='text-align:center;
    background: linear-gradient(to right, #2E8B57, #66CDAA);
    -webkit-background-clip: text;
    color: transparent;'>
    🌱 AI-Based Plant Health Dashboard
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="🌿 Original Leaf", width=300)

    # -------------------------
    # PREPROCESS
    # -------------------------
    img = cv2.resize(img, (300, 300))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # -------------------------
    # GREEN MASK
    # -------------------------
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # -------------------------
    # DISEASE MASK
    # -------------------------
    lower_yellow = np.array([10, 50, 50])
    upper_yellow = np.array([40, 255, 255])

    lower_brown = np.array([0, 50, 20])
    upper_brown = np.array([20, 255, 200])

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)

    # -------------------------
    # DISEASE NAME PREDICTION
    # -------------------------
    yellow_ratio = np.sum(yellow_mask > 0) / (300 * 300)
    brown_ratio = np.sum(brown_mask > 0) / (300 * 300)

    if brown_ratio > yellow_ratio and brown_ratio > 0.02:
        disease_name = "🍂 Leaf Blight"
    elif yellow_ratio > 0.03:
        disease_name = "🟡 Leaf Spot"
    elif (yellow_ratio + brown_ratio) > 0.01:
        disease_name = "⚠ Early Infection"
    else:
        disease_name = "🌿 Healthy Leaf"

    # -------------------------
    # OVERLAY
    # -------------------------
    overlay = img.copy()
    overlay[disease_mask > 0] = [255, 0, 0]
    final_output = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    with col2:
        st.image(final_output, caption="🔴 Diseased Area Highlighted", width=300)

    st.markdown("---")

    # -------------------------
    # PIXEL CALCULATION
    # -------------------------
    total = img.shape[0] * img.shape[1]
    green_pixels = np.sum(green_mask == 255)
    disease_pixels = np.sum(disease_mask == 255)

    health = (green_pixels / total) * 100
    disease = (disease_pixels / total) * 100

    # -------------------------
    # RESULT
    # -------------------------
    st.markdown("## 📊 AI Prediction Result")
    st.markdown(f"### 🧠 Disease Name: **{disease_name}**")

    st.markdown("---")

    st.progress(int(health))
    st.markdown(f"🌿 Health: {health:.2f}%")

    st.progress(int(disease))
    st.markdown(f"🦠 Disease: {disease:.2f}%")

    # -------------------------
    # SEVERITY
    # -------------------------
    st.subheader("⚠ Severity Level")

    if disease < 5:
        st.success("LOW SEVERITY")
    elif disease < 20:
        st.warning("MEDIUM SEVERITY")
    else:
        st.error("HIGH SEVERITY")

    st.markdown("---")

    # -------------------------
    # SMALL PIE CHART (FIXED)
    # -------------------------
    st.subheader("📊 Health vs Disease")

    fig, ax = plt.subplots(figsize=(2.5, 2.5))  # 🔥 SMALL SIZE FIX

    ax.pie(
        [health, disease],
        labels=["Healthy", "Diseased"],
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 8}
    )

    ax.axis("equal")

    st.pyplot(fig, use_container_width=False)

    st.markdown("---")

    # -------------------------
    # FINAL RESULT
    # -------------------------
    if disease > 5:
        st.error(f"⚠ Plant is Diseased → {disease_name}")
    else:
        st.success("✅ Plant is Healthy")

    # -------------------------
    # RECOMMENDATION
    # -------------------------
    st.subheader("💊 Recommendation")

    if disease_name == "🍂 Leaf Blight":
        st.error("Use fungicide spray immediately.")
    elif disease_name == "🟡 Leaf Spot":
        st.warning("Remove infected leaves and monitor growth.")
    elif disease_name == "⚠ Early Infection":
        st.info("Apply preventive treatment.")
    else:
        st.success("No treatment needed. Plant is healthy.")

# FOOTER
st.markdown("""
<hr>
<h4 style='text-align:center; color:green;'>
🍃 Smart Plant Disease AI System 🍃
</h4>
""", unsafe_allow_html=True)
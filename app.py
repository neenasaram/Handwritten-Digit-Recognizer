import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image, ImageOps

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognizer")

# Sidebar - Drawing options
st.sidebar.title("Drawing Settings")
mode = st.sidebar.selectbox("Drawing Tool", ("freedraw", "line"))
stroke_width = st.sidebar.slider("Stroke width", 5, 25, 15)
stroke_color = st.sidebar.color_picker("Stroke color", "#000000")
bg_color = st.sidebar.color_picker("Background color", "#FFFFFF")

# Load model
@st.cache_resource
def load_mnist_model():
    return load_model("final_model.keras")

model = load_mnist_model()

# Preprocessing Function
def preprocess(img):
    img = ImageOps.grayscale(img)
    img = np.array(img)

    if np.mean(img) > 127:
        img = 255 - img

    _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(img)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    else:
        img = np.zeros((28, 28), dtype=np.uint8)

    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
    img = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28)
    return img

# Image upload
st.subheader("📤 Upload an Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])


# Draw canvas
st.subheader("🖌️Draw a Digit")
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    drawing_mode=mode,
    key="canvas",
)


# Prediction
input_img = None
if uploaded_file:
    input_img = Image.open(uploaded_file).convert("RGB")
elif canvas_result.image_data is not None:
    input_img = Image.fromarray(canvas_result.image_data.astype("uint8"))

if input_img:
    st.image(input_img, caption="Input Image", width=150)

    if st.button("🔍 Predict"):
        processed = preprocess(input_img)
        prediction = model.predict(processed, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        st.success(f"🧠 Predicted Digit: **{digit}** with **{confidence:.2f}%** confidence")

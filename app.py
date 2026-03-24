import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Page Config
st.set_page_config(page_title="Digit AI", layout="wide")

st.title("🔢 AI Digit Classifier")
st.write("Project for HR Interview Demo")

# Sidebar
st.sidebar.header("Instructions")
st.sidebar.info("Upload a clear image of a single handwritten digit (0-9).")

@st.cache_resource
def get_model():
    # Adding a spinner so the user knows the AI is 'thinking'
    with st.spinner("Loading Neural Network..."):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

# Initialize model
try:
    model = get_model()
    st.success("✅ Model Ready")
except Exception as e:
    st.error(f"Error loading model: {e}")

# File Uploader
uploaded_file = st.file_uploader("Upload your handwritten digit (PNG/JPG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Processing
    image = Image.open(uploaded_file).convert('L')
    image = ImageOps.invert(image) 
    image = image.resize((28, 28))
    
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image, width=200, caption="Pre-processed for CNN")

    with col2:
        st.subheader("AI Result")
        prediction = model.predict(img_array)
        result = np.argmax(prediction)
        st.metric("Predicted Digit", result)
        st.progress(int(np.max(prediction) * 100))
        st.write(f"Confidence: {np.max(prediction)*100:.1f}%")
else:
    st.warning("Waiting for an image upload...")
  

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model # <<< Import load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input as mobilenet_v2_preprocess_input,
)
import io
import os

st.set_page_config(layout="wide")

# --- Model Architecture ---
def create_inference_model():
    base_model = MobileNetV2(
        input_shape=(160, 160, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    inputs = Input(shape=(160, 160, 3), name="input_layer")
    x = mobilenet_v2_preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D(name="global_average_pooling2d")(x)
    x = Dropout(0.3, name="dropout")(x)
    x = Dense(128, activation="relu", name="dense")(x)
    x = Dropout(0.4, name="dropout_1")(x)
    x = Dense(64, activation="relu", name="dense_1")(x)
    x = Dropout(0.3, name="dropout_2")(x)
    outputs = Dense(10, activation="softmax", name="dense_2")(x)
    inference_model = Model(inputs, outputs)
    return inference_model

def get_prediction(img_pil):
    selected_model_name = st.session_state.selected_model

    #Select the correct model
    if selected_model_name == "transfer model":
        model_to_use = transfer_model
    elif selected_model_name == "basic model":
        model_to_use = basic_model
    else:
        st.error("Invalid model selected.")
        return

    preprocessed_img = preprocess_pil_image(img_pil)

    if preprocessed_img is not None:
        # Get prediction
        try:
            prediction_array = model_to_use.predict(preprocessed_img)
            predicted_class = np.argmax(prediction_array)
            confidence = np.max(prediction_array) * 100
            st.info(
                f"Prediction: {CLASSES[predicted_class]} (Confidence: {confidence:.2f}%)"
            )
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# --- Model Loading ---
@st.cache_resource
def load_tf_model(weights_path):
    """
    Loads the Keras model weights into the defined architecture.
    """
    try:
        if not os.path.exists(weights_path):
            st.error(f"Weights file not found: {weights_path}")
            st.info("Please run the `model.save_weights('hand_drawn_symbol_model_WEIGHTS.h5')` "
                    "command in your Colab notebook, download the file, "
                    "and place it in the same directory as this script.")
            return None

        #Create the model structure
        model = create_inference_model()

        #Load the weights from the .h5 file
        model.load_weights(weights_path)

        st.success(f"Model architecture created and weights loaded successfully from {weights_path}.")
        return model

    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None


@st.cache_resource
def load_basic_model(model_path):
    """
    Loads the *entire* Keras model (architecture + weights) from a .keras file.
    """
    try:
        if not os.path.exists(model_path):
            st.error(f"Basic model file not found: {model_path}")
            return None

        model = tf.keras.models.load_model(model_path)

        st.success(f"Basic model loaded successfully from {model_path}.")
        return model
    except Exception as e:
        st.error(f"Error loading basic model: {e}")
        return None

# --- Image Preprocessing ---
def preprocess_pil_image(img_pil):
    try:
        #Convert to RGB (3 channels)
        img_rgb = img_pil.convert("RGB")

        #Resize to the model's expected input size (160x160)
        img_resized = img_rgb.resize((160, 160), Image.LANCZOS)

        #Convert to numpy array
        img_array = np.array(img_resized)

        #Expand dimensions to match model's expected input shape (1, 160, 160, 3)
        img_final = np.expand_dims(img_array, axis=0)

        #Ensure dtype is float32 for the model
        img_final = img_final.astype("float32")

        return img_final

    except Exception as e:
        st.error(f"Error in preprocessing image: {e}")
        return None

CLASSES = ['anchor', 'balloon', 'bicycle', 'envelope', 'paper_boat', 'peace_symbol', 'smiley','speech_bubble', 'spiral', 'thumb']
# --- Load the Model ---
TRANSFER_MODEL_WEIGHTS_PATH = "final_model_transfer.weights.h5"
BASIC_MODEL_PATH = "final_model.keras"

transfer_model = load_tf_model(TRANSFER_MODEL_WEIGHTS_PATH)
basic_model = load_basic_model(BASIC_MODEL_PATH)
# --- Streamlit UI ---

left, centre, right = st.columns([1, 4, 1])

if "width" not in st.session_state:
    st.session_state.width = 10
if "color" not in st.session_state:
    st.session_state.color = "#000000"
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "transfer model"

with st.sidebar:
    st.session_state.selected_model = st.radio(label="Select model", options = ['transfer model', 'basic model'], captions = ["Model based on MobileNetV2", "Model made fully by me"])
    st.session_state.width = st.slider("Brush size:", min_value=5, max_value=50, value=10, step=1)
    st.session_state.color = st.color_picker("Brush color:", value="#000000")
    st.info(f"Selected model: {st.session_state.selected_model}")

with centre:
    st.title("Image Recognition")

    if (st.session_state.selected_model == "transfer model" and transfer_model is None) or \
            (st.session_state.selected_model == "basic model" and basic_model is None):
        st.error("The currently selected model could not be loaded. "
                 "Please check the sidebar and error messages above.")
    else:
        tab1, tab2 = st.tabs(["Draw", "Upload Image"])

        with tab1:
            st.header("Draw Image")
            drawing = st_canvas(
                stroke_width=st.session_state.width,
                stroke_color=st.session_state.color,
                background_color="#FFFFFF",
                height=800,  # Adjusted height for better layout
                width=800,  # Adjusted width
                drawing_mode="freedraw",
                key="canvas",
            )

            model_output_box = st.empty()

            if st.button("Send Image"):
                if drawing.image_data is not None:
                    # Convert canvas data to PIL Image
                    # The canvas returns RGBA, so we slice off the alpha channel
                    img_pil = Image.fromarray(drawing.image_data[:, :, :3].astype('uint8'), 'RGB')

                    get_prediction(img_pil)
                else:
                    model_output_box.warning("Please draw something first!")

        with tab2:
            st.header("Upload Image")
            uploaded_image = st.file_uploader(
                "Choose an image file", type=["png", "jpg", "jpeg"]
            )

            if uploaded_image is not None:
                # Display the uploaded image
                st.image(uploaded_image, use_container_width=True, caption="Uploaded Image")
                img_pil = Image.open(uploaded_image)

                get_prediction(img_pil)


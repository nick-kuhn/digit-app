import sys
import os

# This is to ensure that when streamlit_app.py is run directly (e.g., by Streamlit CLI),
# it can find its sibling modules (model.py, database.py) in the 'app' directory.
# This is primarily for local development. Docker execution context should be fine.
if __name__ == '__main__' and os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch
from model import Net # Changed from relative
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import os # Import os for environment variables
from database import get_db_connection, create_table, log_prediction_db # Changed from relative

# --- Initialize Database ---
# Ensure the table exists when the app starts
create_table()

#Load and instantiate the model
@st.cache_resource
def load_pytorch_model(model_filename='mnist_cnn.pth'): # Changed arg name for clarity
    # Construct path relative to this script file (streamlit_app.py)
    script_dir = os.path.dirname(__file__)
    model_path_abs = os.path.join(script_dir, model_filename)

    # print(f"Attempting to load model from: {model_path_abs}") # Debug line

    model = Net() # Instantiate the model structure
    model.load_state_dict(torch.load(model_path_abs)) # Use absolute path
    model.eval() # Set to evaluation mode
    return model

# --- Load the model (globally) ---
model = load_pytorch_model()

# --- Session State Initialization ---
if "true_label" not in st.session_state:
    st.session_state.true_label = None
if "predicted_digit" not in st.session_state:
    st.session_state.predicted_digit = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "show_prediction_results" not in st.session_state:
    st.session_state.show_prediction_results = False
if "canvas_key_suffix" not in st.session_state: # Add new key for canvas
    st.session_state.canvas_key_suffix = 0
if "prediction_history" not in st.session_state: # For storing prediction history
    st.session_state.prediction_history = []
if "last_true_label_for_display" not in st.session_state: # To show true label with current prediction
    st.session_state.last_true_label_for_display = None

# Placeholder for image preprocessing function
def preprocess_image(image_data):
    # Input: NumPy array (HxWx4 RGBA)
    if image_data is None:
        return None

    try:
        # 1. Convert NumPy array to PIL Image
        pil_image = Image.fromarray(image_data.astype('uint8'), 'RGBA')

        # 2. Convert RGBA to Grayscale
        gray_image = pil_image.convert('L')

        # --- Add Inversion Step ---
        # MNIST expects white digit on black background, canvas is black on white.
        inverted_image = ImageOps.invert(gray_image)
        # --- End Inversion Step ---

        # 3. Resize to 28x28 pixels (using the inverted image)
        resized_image = inverted_image.resize((28, 28), Image.Resampling.LANCZOS)

        # 4. Define transformations (ToTensor scales to [0, 1])
        # MNIST Mean and Std Deviation
        mnist_mean = (0.1307,)
        mnist_std = (0.3081,)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ])

        # 5. Convert to Tensor and Normalize
        tensor_image = transform(resized_image)

        # 6. Add Batch Dimension (channel dimension is added by ToTensor for grayscale)
        # tensor_image shape is [1, 28, 28], need [1, 1, 28, 28]
        processed_tensor = tensor_image.unsqueeze(0) # Add batch dimension

        return processed_tensor

    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

# Placeholder for prediction function
def predict_digit(processed_tensor): # model is accessed from global scope
    # Ensure the model is loaded (it's loaded globally and cached)
    if model is None:
        st.error("Model not loaded!")
        return None, None

    # Ensure model is in evaluation mode
    model.eval()

    # Disable gradient computation for inference
    with torch.no_grad():
        try:
            # Pass the preprocessed tensor to the model
            # Input tensor shape should be [1, 1, 28, 28]
            output = model(processed_tensor)

            # Process the output
            # Apply Softmax to get probabilities if output are raw logits
            probabilities = torch.softmax(output, dim=1)
            
            # Get the confidence (max probability)
            confidence_val = probabilities.max().item() # Renamed to avoid conflict
            
            # Get the prediction (index of the highest probability)
            predicted_digit_val = probabilities.argmax(dim=1).item() # Renamed

            return predicted_digit_val, confidence_val

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None, None

# --- Streamlit App ---

st.title("MNIST Digit Recognizer")
st.header("Draw a digit (0-9) below.")

# --- Canvas Configuration ---
# Define canvas parameters based on requirements
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10) # Example: Make stroke width adjustable
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#000000") # Black
background_color = "#EEEEEE" # Light gray background
drawing_mode = "freedraw"
canvas_height = 280 # Make canvas larger for easier drawing
canvas_width = 280

# --- Layout for Canvas and True Label ---
col1, col2 = st.columns([2, 1]) # Assign more space to the canvas

with col1:
    # --- Drawing Canvas ---
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=background_color,
        update_streamlit=True, # Update automatically on drawing
        height=canvas_height,
        width=canvas_width,
        drawing_mode=drawing_mode,
        key=f"canvas_{st.session_state.canvas_key_suffix}", # Use dynamic key
    )

with col2:
    # Input for the "True Label"
    # User input directly updates st.session_state.true_label due to the key
    st.number_input("Enter True Label (0-9):", min_value=0, max_value=9, step=1, key="true_label")

# --- Prediction Logic (Callback Function) ---
def handle_prediction_and_reset():
    # Access canvas_result from the outer scope (defined before this button)
    # Access model from the global scope
    
    if canvas_result.image_data is not None:
        # 1. Retrieve Image Data
        image_data_rgba = canvas_result.image_data # Shape (height, width, 4) - RGBA

        # 2. Preprocess the image
        processed_image = preprocess_image(image_data_rgba)

        if processed_image is not None:
            # 3. Perform Inference
            predicted_digit_val, confidence_val = predict_digit(processed_image)

            # 4. Store results in session state for display
            if predicted_digit_val is not None and confidence_val is not None:
                st.session_state.predicted_digit = predicted_digit_val
                st.session_state.confidence = confidence_val
                st.session_state.show_prediction_results = True
                
                current_true_label_for_logging_and_display = st.session_state.true_label
                st.session_state.last_true_label_for_display = current_true_label_for_logging_and_display

                # 5. Log the result ONLY if true_label was provided
                if current_true_label_for_logging_and_display is not None:
                    log_prediction_db(predicted_digit_val, confidence_val, current_true_label_for_logging_and_display, image_data_rgba)

                # 6. Add to prediction history
                history_entry = {
                    "Predicted Digit": predicted_digit_val,
                    "Confidence": f"{confidence_val:.2f}",
                    "True Label": current_true_label_for_logging_and_display if current_true_label_for_logging_and_display is not None else ""
                }
                st.session_state.prediction_history.insert(0, history_entry) # Insert at beginning for newest first

                # Reset true_label in session state after processing
                st.session_state.true_label = None 
            else:
                # Prediction error (message handled within predict_digit)
                st.session_state.show_prediction_results = False
                st.warning("Prediction failed.")
        else:
            st.warning("Could not preprocess the image.")
            st.session_state.show_prediction_results = False
    else:
        st.warning("Please draw a digit on the canvas first.")
        st.session_state.show_prediction_results = False
    
    # Increment canvas key suffix to force re-render (clear canvas)
    st.session_state.canvas_key_suffix += 1

# Add a button to trigger prediction, using the callback
predict_button = st.button("Predict Digit", on_click=handle_prediction_and_reset)

# --- Display Prediction Results (based on session state) ---
if st.session_state.show_prediction_results:
    if st.session_state.predicted_digit is not None and st.session_state.confidence is not None:
        st.subheader("Prediction Result")
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric(label="Predicted Digit", value=str(st.session_state.predicted_digit))
        with res_col2:
            st.metric(label="Confidence", value=f"{st.session_state.confidence:.2f}")
        with res_col3:
            true_label_to_display = st.session_state.last_true_label_for_display
            st.metric(label="True Label", value=str(true_label_to_display) if true_label_to_display is not None else "N/A")
    # Or, let it persist until the next button click naturally updates it.

# --- Display Prediction History ---
if st.session_state.prediction_history:
    st.subheader("Prediction History")
    st.dataframe(st.session_state.prediction_history)

#st.write("Note: Preprocessing, prediction, and logging are currently placeholders.")




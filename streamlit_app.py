import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import torch
from model import Net
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import os # Import os for environment variables
from database import get_db_connection, create_table, log_prediction_db # Import DB functions

# --- Initialize Database ---
# Ensure the table exists when the app starts
create_table()

#Load and instantiate the model
@st.cache_resource
def load_pytorch_model(model_path='mnist_cnn.pth'):
    model = Net() # Instantiate the model structure
    model.load_state_dict(torch.load(model_path))
    model.eval() # Set to evaluation mode
    return model

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
def predict_digit(processed_tensor):
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
            confidence = probabilities.max().item()
            
            # Get the prediction (index of the highest probability)
            predicted_digit = probabilities.argmax(dim=1).item()

            return predicted_digit, confidence

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None, None

# Placeholder for logging function - REMOVED (using log_prediction_db now)
# def log_prediction(image_data, predicted_digit, confidence, true_label):
#     # TODO: Implement logging logic (e.g., save to file/database)
#     st.write(f"Logging Step (Placeholder): Image Shape {image_data.shape if image_data is not None else 'None'}, Predicted: {predicted_digit}, Confidence: {confidence:.2f}, True Label: {true_label}")
#     pass

# --- Streamlit App ---

st.title("MNIST Digit Recognizer")
st.header("Draw a digit (0-9) below")

# --- Canvas Configuration ---
# Define canvas parameters based on requirements
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 10) # Example: Make stroke width adjustable
stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#000000") # Black
background_color = "#EEEEEE" # Light gray background
drawing_mode = "freedraw"
canvas_height = 280 # Make canvas larger for easier drawing
canvas_width = 280

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
    key="canvas",
)

# --- Prediction Logic ---
# Add a button to trigger prediction
predict_button = st.button("Predict Digit")

# Input for the "True Label"
true_label = st.number_input("Enter True Label (0-9):", min_value=0, max_value=9, step=1, key="true_label")

# --- Load the model ---
model = load_pytorch_model()

if predict_button:
    if canvas_result.image_data is not None:
        # 1. Retrieve Image Data
        image_data_rgba = canvas_result.image_data # Shape (height, width, 4) - RGBA

        # 2. Preprocess the image
        # Note: Preprocessing needs to convert RGBA to grayscale and resize to MNIST format (28x28)
        processed_image = preprocess_image(image_data_rgba)

        if processed_image is not None:
            # 3. Perform Inference
            predicted_digit, confidence = predict_digit(processed_image)

            # 4. Display Prediction (only if prediction was successful)
            if predicted_digit is not None and confidence is not None:
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Predicted Digit", value=str(predicted_digit))
                with col2:
                    st.metric(label="Confidence", value=f"{confidence:.2f}")

                # 5. Log the result (only if prediction was successful)
                # Using the original RGBA data for potential future use/visualization
                # Calls the new database logging function
                log_prediction_db(predicted_digit, confidence, true_label, image_data_rgba)
            # else: prediction error message is handled within predict_digit

        else:
            st.warning("Could not preprocess the image.")

    else:
        st.warning("Please draw a digit on the canvas first.")

#st.write("Note: Preprocessing, prediction, and logging are currently placeholders.")




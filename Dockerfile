# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .
# Ensure the model file (e.g., mnist_cnn.pth) is copied

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define healthcheck
# Note: The healthcheck provided in the prompt was incomplete.
# Streamlit's health endpoint is typically /_stcore/health
# A simple curl check:
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run streamlit when the container launches
# Use 0.0.0.0 to make it accessible from outside the container
# Assuming the main streamlit app file is named streamlit_app.py
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 
# syntax=docker/dockerfile:1
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if needed (e.g., for certain Python packages)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file first to leverage Docker cache
# This assumes requirements.txt is at the project root
COPY requirements.txt /app/

# Install Python dependencies using a BuildKit cache mount.
# This speeds up rebuilds by caching downloaded packages (like torch) 
# across builds, even if the requirements.txt file changes slightly, 
# preventing re-downloads if the package version itself is unchanged.
# The --no-cache-dir flag is removed here to allow pip to use the mounted cache.
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /app/requirements.txt

# Copy the application code from the local 'app' directory into the container's WORKDIR ('/app')
# This will include streamlit_app.py, model.py, database.py, mnist_cnn.pth, and the .streamlit subdirectory.
COPY ./app /app/

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
# streamlit_app.py is now directly in /app, so this path is correct.
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 
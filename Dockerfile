# Use an official Python runtime as a parent image (choose a specific version)
FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by OpenCV and potentially others
# Using non-interactive frontend to avoid prompts during build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (Gunicorn will use $PORT env var)
# Fly.io typically expects 8080 if not specified otherwise in fly.toml
EXPOSE 7860

# Command to run the application using Gunicorn
# Listens on 0.0.0.0 and uses the PORT environment variable provided by Fly.io
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT:-8080}", "--workers", "1", "--threads", "8", "--timeout", "120", "app:app"]
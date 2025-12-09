FROM python:3.11-slim

# Install system dependencies yang dibutuhkan dlib dan face_recognition
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libx11-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "face_recognition_api.py"]
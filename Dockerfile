FROM python:3.10-slim

# Install system dependencies (Tesseract OCR and FFmpeg for audio/image processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure storage directories exist
RUN mkdir -p uploads vector_store backend/database

# Grant execution permissions to start script
RUN chmod +x start.sh

# Expose Render default web port
EXPOSE 10000

CMD ["./start.sh"]

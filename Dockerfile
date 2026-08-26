# ==========================================
# Cats vs Dogs - Inference Service
# ==========================================

FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and ensure logs appear immediately
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies required by TensorFlow/Pillow
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src

# Copy trained model
COPY models ./models

# Expose FastAPI port
EXPOSE 8000

# Start the inference service
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
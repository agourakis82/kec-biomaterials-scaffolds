# Darwin API Dockerfile (prod-ready)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (if needed, minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Remove conflicting logging.py file that causes circular imports
RUN rm -f /app/logging.py || true

# Set PYTHONPATH to include the root directory
ENV PYTHONPATH=/app

# Expose port for Cloud Run
EXPOSE 8080

# Start the FastAPI app with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

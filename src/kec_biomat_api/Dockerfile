# KEC Biomaterials API v2.0 - Modular Backend (Cloud Run Optimized)
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (minimal for cloud optimization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first (for Docker layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code 
COPY . .

# Initialize git submodules (pcs-meta-repo)
RUN git submodule update --init --recursive || echo "No submodules to update"

# Create necessary directories
RUN mkdir -p /app/data/memory /app/logs /app/external

# Remove conflicting files that cause circular imports
RUN rm -f /app/logging.py || true

# Set environment variables for modular backend
ENV PYTHONPATH="/app/src:/app/external/pcs-meta-repo:/app"
ENV KEC_CONFIG_PATH="/app/src/kec_biomat/configs"
ENV DARWIN_MEMORY_PATH="/app/src/darwin_core/memory"
ENV GOOGLE_CLOUD_PROJECT="pcs-helio"
ENV PORT=8080

# Cloud Run optimizations
ENV WORKERS=1
ENV WORKER_CLASS="uvicorn.workers.UvicornWorker"
ENV WORKER_CONNECTIONS=1000
ENV MAX_REQUESTS=1000
ENV MAX_REQUESTS_JITTER=100
ENV TIMEOUT=300
ENV GRACEFUL_TIMEOUT=120

# Production optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONOPTIMIZE=1

# Security: Create non-root user
RUN groupadd -r kecuser && useradd -r -g kecuser kecuser
RUN chown -R kecuser:kecuser /app
USER kecuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/healthz || exit 1

# Expose port
EXPOSE 8080

# Optimized startup command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--access-log", "--log-level", "info"]

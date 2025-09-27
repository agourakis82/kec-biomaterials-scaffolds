# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies (com retries; sem alterar resolv.conf)
RUN printf 'Acquire::Retries "3";\n' > /etc/apt/apt.conf.d/80-retries \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential gcc g++ git curl \
       # Para neurociência/científico
       libblas-dev liblapack-dev gfortran \
       # Para visualização
       libcairo2-dev libpango1.0-dev \
       # Para document processing
       tesseract-ocr tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Use BuildKit cache for pip to speed up rebuilds
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -U pip setuptools wheel && \
    pip install -r requirements.txt
   pip install jax==0.4.20 --force-reinstall

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH="/app"
ENV PORT=8080
ENV ENVIRONMENT=production
ENV JAX_PLATFORM_NAME=cpu
ENV NVIDIA_VISIBLE_DEVICES=void

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

# Verify AutoGen installation
RUN pip show pyautogen || echo "pyautogen not found after install"

# Command to run the application
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT} --workers 4"]
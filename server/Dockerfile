FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY fastapi_app.py ./fastapi_app.py
COPY openapi.yaml ./openapi.yaml
COPY openapi.json ./openapi.json

EXPOSE 8080

CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8080"]


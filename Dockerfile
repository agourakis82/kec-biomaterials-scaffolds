# Dockerfile for Production - KEC Unified API

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install Python dependencies
COPY config/requirements-darwin.txt .
RUN pip install --no-cache-dir -r requirements-darwin.txt

# Copy application code
COPY src/kec_unified_api/ ./

# Expose the port the app runs on
EXPOSE 8080

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

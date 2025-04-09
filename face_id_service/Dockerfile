FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    cron \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt requests

# Copy application code
COPY . .

# Create upload directory and logs directory
RUN mkdir -p /app/uploads /app/logs

# Set up the wake-up script
COPY wake_service.py /app/wake_service.py
RUN chmod +x /app/wake_service.py

# Set up cron job to run every 10 minutes
RUN echo "*/10 * * * * /usr/local/bin/python /app/wake_service.py >> /app/logs/cron.log 2>&1" > /etc/cron.d/wake-service-cron
RUN chmod 0644 /etc/cron.d/wake-service-cron
RUN crontab /etc/cron.d/wake-service-cron

# Set environment variable for upload directory
ENV UPLOAD_DIR=/app/uploads

# Expose the port the app runs on
EXPOSE 8000

# Create a startup script to run both cron and the main application
RUN echo '#!/bin/bash\nservice cron start\nuvicorn main:app --host 0.0.0.0 --port 8000' > /app/start.sh
RUN chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"]

# Set the base CUDA-enabled image.
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Avoid interactive apt prompts.
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container.
WORKDIR /app
ENV PYTHONPATH=/app/src

# Install system dependencies in a stable layer.
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    python3 \
    python3-pip \
    openssl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying app code so small code edits do not
# reinstall the whole dependency stack.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Generate self-signed SSL certificate in a stable layer.
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/selfsigned.key -out /etc/ssl/certs/selfsigned.crt \
    -subj "/C=US/ST=California/L=LosAngeles/O=Dis/CN=www.example.com"

# Copy the large local model separately. This layer is reused as long as the
# model files do not change.
COPY model ./model

# Copy application code last. Editing Python modules should now
# rebuild only this small layer.
COPY src ./src

# Set the default worker count.
ENV WORKER_COUNT=1

# Expose HTTPS port.
EXPOSE 443

# Run the API when the container launches using HTTPS.
CMD gunicorn -b 0.0.0.0:443 --certfile=/etc/ssl/certs/selfsigned.crt --keyfile=/etc/ssl/private/selfsigned.key polyglot_api.app:app --workers $WORKER_COUNT -k uvicorn.workers.UvicornWorker --log-level info --access-logfile - --error-logfile -

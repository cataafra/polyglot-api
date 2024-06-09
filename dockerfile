# Set the base cuda enabled image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set the working directory in the container
WORKDIR /app

# Copy application code to the container
COPY . .

# Install basic packages for the container
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    python3 \
    python3-pip \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Generate self-signed SSL certificate
RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/selfsigned.key -out /etc/ssl/certs/selfsigned.crt \
    -subj "/C=US/ST=California/L=LosAngeles/O=Dis/CN=www.example.com"

# Set the default worker count
ENV WORKER_COUNT=1

# Expose HTTPS port
EXPOSE 443

# Run app.py when the container launches using HTTPS
CMD gunicorn -b 0.0.0.0:443 --certfile=/etc/ssl/certs/selfsigned.crt --keyfile=/etc/ssl/private/selfsigned.key app:app --workers $WORKER_COUNT -k uvicorn.workers.UvicornWorker --log-level info --access-logfile - --error-logfile -

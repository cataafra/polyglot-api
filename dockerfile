# Set the base cuda enabled image
FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# Set the working directory in the container
WORKDIR .

COPY . .

# Install basic packages for the container
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the default worker count
ENV WORKER_COUNT=1

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD gunicorn -b 0.0.0.0:80 app:app --workers $WORKER_COUNT -k uvicorn.workers.UvicornWorker --log-level info --access-logfile - --error-logfile -
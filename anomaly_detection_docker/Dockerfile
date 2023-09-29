# Use the official Python base image with Conda
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files into the container
COPY app.py /app
COPY detect_anomaly.py /app
COPY anomaly_api.yml /app
COPY trained_models /app/trained_models
COPY anomaly_detection /app/anomaly_detection

# Create and activate the Conda environment
RUN conda env create -f anomaly_api.yml
SHELL ["conda", "run", "-n", "anomaly_api", "/bin/bash", "-c"]

# Expose the desired port
EXPOSE 5000

# Set the command to run when the container starts
ENTRYPOINT [ "conda", "run", "-n", "anomaly_api", "python", "app.py" ]


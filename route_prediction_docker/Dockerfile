# Use the official Python base image with Conda
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files into the container
COPY app.py /app
COPY predict_trajectory.py /app
COPY route_prediction_api.yml /app
COPY trained_models /app/trained_models
COPY route_prediction /app/route_prediction

# Create and activate the Conda environment
RUN conda env create -f route_prediction_api.yml
SHELL ["conda", "run", "-n", "route_prediction_api", "/bin/bash", "-c"]

# Expose the desired port
EXPOSE 5000

# Set the command to run when the container starts
ENTRYPOINT [ "conda", "run", "-n", "route_prediction_api", "python", "app.py" ]


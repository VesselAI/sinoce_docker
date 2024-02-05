# Use the official Python base image with Conda
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app
# Copy the necessary files into the container
COPY pre_process.py /app
COPY pre_processing.yml /app
COPY location_data.csv /app
# Create and activate the Conda environment
RUN conda env create -f pre_processing.yml
SHELL ["conda", "run", "-n", "pre_processing", "/bin/bash", "-c"]





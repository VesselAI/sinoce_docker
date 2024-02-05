# Use the official Python base image with Conda
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the necessary files into the container
COPY train_data_dest_50.json /app
COPY app.py /app
COPY knn_cluster_prediction_api.yml /app
# Create and activate the Conda environment
RUN conda env create -f knn_cluster_prediction_api.yml
SHELL ["conda", "run", "-n", "knn_cluster_prediction_api", "/bin/bash", "-c"]

# Expose the desired port
EXPOSE 5000

# Set the command to run when the container starts
ENTRYPOINT [ "conda", "run", "-n", "knn_cluster_prediction_api", "python", "app.py" ]


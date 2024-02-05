# Overview
This repo contains different docker containers:
1. pre processing
2. anomaly detection
3. NN route prediction
4. kNN PCA route prediction


See readmes inside the different folders to run the containers. 

# Model files
The model files for the anomaly detection and route prediction are not included in this repo as they are too large.
To get the model files, contact 
1. Brian Murray: brian.murray@sintef.no
or 
2. Pauline Røstum Bellingmo: pauline.bellingmo@sintef.no


You will then get a ``model.pth`` file for the route prediction and/or anomaly detection (depending on your need). 
Place this file within
* ``route_prediction_docker/trained_models/model/data/`` for route prediction model, and 
* ``anomaly_detection_docker/trained_models/model/data/`` for anomaly detection model,

For knn_pca_route_prediction you will need a different file (must be requested):
`` train_data_dest_50.json`` 



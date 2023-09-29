# Overview
This repo contains two different docker container:  
1. anomaly detection
2. route prediction 

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



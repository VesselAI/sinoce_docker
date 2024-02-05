#Docker Route Prediction Image

The docker container provides a REST-API, accessible via an HTTP POST request. 

1. Run docker engine
2. In terminal run ``docker build -t ais_pre_processing .`` from pre-processing docker directory
3. Once built, run container using ``docker run -v /host/path/to/ais_data:/app/ais_data -v /host/path/to/output:/app/output ais_pre_processing python pre_process.py "/app/ais_data/" -o "/app/output/"``



docker run -v ./ais_data:/app/ais_data -v ./:/app/output ais_pre_processing python pre_process.py "/app/ais_data/" -o "/app/output/"



docker exec -it ais_pre_processing pre_process.py "/app/ais_data/" -o "/app/output/"

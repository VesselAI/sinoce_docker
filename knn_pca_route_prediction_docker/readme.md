#Docker Route Prediction Image

The docker container provides a REST-API, accessible via an HTTP POST request. 

1. Run docker engine
2. In terminal run ``docker build -t vesselai_route_prediction_app .`` from route_prediction_docker directory    
3. Update to desired port in docker-compose.yml 
4. Once built, run container using ``docker-compose up``

#REST-API

Request:

URL: http://localhost:5000/route_prediction

BODY: JSON with format (see json_examples/example_request.json)

{"trajectory_data": 10 minutes (11 points) [lon, lat],

"origin": LOCODE,

"destination":  LOCODE

"EPSG": 4326 (for lon ,lat)

"horizon": prediction horizon in minutes (up to 50)

}

Response:

JSON with format (see json_examples/example_response.json):

{prediction: list of predicted positions starting at 5 minutes up to and including horizon (max 50 minutes) [lon, lat]

}

#Model Validity

The model is valid within the following geographical region:

x_min:  10.205 #lon
y_min:  59.206 #lat
x_max:  10.817 #lon
y_max:  59.598 #lat


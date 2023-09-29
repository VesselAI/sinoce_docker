#Docker Route Prediction Image

The docker container provides a REST-API, accessible via an HTTP POST request. 

1. Run docker engine
2. In terminal run ``docker build -t vai_route_prediction_app .`` from route_prediction_docker directory       
3. Once built, run container using ``docker run -d -p 5000:5000 vai_route_prediction_app``

#REST-API

URL: http://localhost:5000/route_prediction

BODY: JSON with format (see json_examples/sample.json)

{"trajectory_data": [# routes [10 minutes [lon, lat]]],

"origin": [#routes LOCODE],

"destination": [#routes LOCODE],

"length": [#routes length of vessel]

"horizon" : [predicition horizon in minutes]

"EPSG": 4326 (for lon ,lat)

}



 
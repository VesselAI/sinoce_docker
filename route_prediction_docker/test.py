import json
import torch
import numpy as np
from predict_trajectory import predict_trajectory, load_models, convert_coordinates
import time
import pandas as pd
cat_scaler, cont_scaler, dynamic_scaler, model = load_models(device='cpu')

if __name__ == '__main__':


    x_output_file = "./json_examples/sample.json"
    """
    x_file = "./json_examples/x_sample.json"
    cat_file = "./json_examples/x_cat_sample.json"


    # Read the JSON data from the file
    with open(x_file, "r") as file:
        x = json.load(file)

    with open(cat_file, "r") as file:
        x_cat = json.load(file)


    



    x = np.array(x)
    x_cat = np.array(x_cat)

    with open(x_output_file, "w") as file:
        to_file = {
            'trajectory_data': x[:1000].tolist(),
            'origin': x_cat[:1000, 0].tolist(),
            'destination': x_cat[:1000, 1].tolist()
        }
        json.dump(to_file, file)
        
    """

    with open(x_output_file, "r") as file:
        test = json.load(file)

    """
    t = test["trajectory_data"][0]
    o = test["origin"][0]
    d = test["destination"][0]
    with open(x_output_file, "w") as file:
        to_file = {"trajectory_data": test["trajectory_data"][0],
                   "origin": test["origin"][0],
                   "destination": test["destination"][0],
                   "EPSG":3035}
        json.dump(to_file, "./json_examples/single_sample.json")
    """

    #destination = test["destination"]


    x = np.array(test["trajectory_data"])
    x_cat = pd.DataFrame({"origin": test["origin"], "destination": test["destination"]})
    x_cont = np.array(test["length"])

    predictions = predict_trajectory(x, x_cat, x_cont, cat_scaler=cat_scaler, cont_scaler=cont_scaler,
                                   dynamic_scaler=dynamic_scaler, model=model, horizon=test["horizon"],
                                   from_epsg=test["EPSG"])

    response = {
        'anomalies': predictions.tolist()
    }
    print('Done')
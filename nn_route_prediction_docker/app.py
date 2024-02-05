import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import torch
import numpy as np
from predict_trajectory import predict_trajectory, load_models


app = Flask(__name__)
 #May need to revisit if KNC wishes to run on GPU
device = 'cpu'
cat_scaler, cont_scaler, dynamic_scaler, model = load_models(device=device)

@app.route('/route_prediction', methods=['POST'])
def route_prediction():
    # Retrieve the input data from the request
    data = request.json
    x = np.array(data["trajectory_data"])
    x_cat = pd.DataFrame({"origin": data["origin"], "destination": data["destination"]})
    x_cont = np.array(data["length"])

    predictions = predict_trajectory(x, x_cat, x_cont, cat_scaler=cat_scaler, cont_scaler=cont_scaler,
                                   dynamic_scaler=dynamic_scaler, model=model, horizon=data["horizon"],
                                   from_epsg=data["EPSG"])

    # Return the predictions as a JSON response
    response = {
        'predictions': predictions.tolist()
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

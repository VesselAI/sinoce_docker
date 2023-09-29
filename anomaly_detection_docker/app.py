import pandas as pd
from flask import Flask, request, jsonify
import mlflow
import torch
import numpy as np
from detect_anomaly import detect_anomaly, load_models


app = Flask(__name__)
 #May need to revisit if KNC wishes to run on GPU
device = 'cpu'
cat_scaler, dynamic_scaler, model, threshold = load_models(device=device)


@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the input data from the request
    data = request.json
    device = torch.device('cpu')
    x = np.array(data["trajectory_data"])
    #x_cat = np.array([data["origin"], data["destination"]]).T
    x_cat = pd.DataFrame({"origin": data["origin"], "destination": data["destination"]})

    anomalies = detect_anomaly(x, x_cat, cat_scaler=cat_scaler, dynamic_scaler=dynamic_scaler, model=model,
                         threshold=threshold, from_epsg=data["EPSG"])

    # Return the predictions as a JSON response
    response = {
        'anomalies': anomalies.tolist()
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

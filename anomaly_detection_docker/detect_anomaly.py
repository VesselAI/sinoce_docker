import time
import torch
import mlflow
import pathlib
import numpy as np
import pyproj
from torch.nn import MSELoss




def load_models(device='cpu'):
    device = torch.device(device)
    cat_scaler_path = str(pathlib.Path(__file__).parent) + f'/trained_models/cat_scaler'
    dyn_scaler_path = str(pathlib.Path(__file__).parent) + f'/trained_models/dynamic_scaler'
    anomaly_model_path = str(pathlib.Path(__file__).parent) + f'/trained_models/model'
    threshold_path = str(pathlib.Path(__file__).parent) + f'/trained_models/threshold.npy'

    cat_scaler = mlflow.sklearn.load_model(cat_scaler_path)
    dynamic_scaler = mlflow.sklearn.load_model(dyn_scaler_path)
    model = mlflow.pytorch.load_model(anomaly_model_path, map_location=device)
    threshold = np.load(threshold_path)

    return cat_scaler, dynamic_scaler, model, threshold


def scale_data(x, x_cat, dynamic_scaler, cat_scaler):
    x = dynamic_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    x_cat = cat_scaler.transform(x_cat)

    return x, x_cat


def get_recon_loss(x, x_cat, model, device):

    loss = MSELoss(reduction='none')
    model.eval()

    x = torch.tensor(x, requires_grad=False).float()
    x = x.to(device)

    x_cat = torch.tensor(x_cat, requires_grad=False).int()
    x_cat = x_cat.to(device)

    y = model(x, x_cat)
    l = loss(x, y)

    return np.mean(np.mean(l.detach().numpy(), axis=2), axis=1)

def convert_coordinates(x, from_epsg=4326, to_epsg=3035):
    #x: [lon, lat] (x,y)

    transformer = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    x_y_tensor = np.apply_along_axis(lambda row: transformer.transform(row[0], row[1]), axis=1,
                                     arr=x.reshape(-1, x.shape[-1])).reshape(x.shape)
    #x_y_tensor [E, N] (x,y)
    return x_y_tensor

def convert_coordinates_3d(x, from_epsg=4326, to_epsg=3035):
    # x: [lon, lat] (x,y)
    transformer = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)

    # Reshape the input tensor to 2D
    x_2d = x.reshape(-1, x.shape[-1])

    # Transform each coordinate in the batch
    transformed_coords = np.apply_along_axis(lambda row: transformer.transform(row[0], row[1]), axis=1, arr=x_2d)

    # Reshape the transformed coordinates back to the original shape
    x_y_tensor = transformed_coords.reshape(x.shape)

    # x_y_tensor [E, N] (x,y)
    return x_y_tensor


def detect_anomaly(x, x_cat, dynamic_scaler, cat_scaler, model, threshold, device='cpu', from_epsg=4326, to_epsg=3035):
    # x: [batch_size, seq_len, n_features]
    # x_cat: [bat_size, n_cat]
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    start = time.time()
    if len(x.shape) != 3:
        x.unsqueeze(0)
    if from_epsg != to_epsg:
        x = convert_coordinates_3d(np.array(x), from_epsg=from_epsg, to_epsg=to_epsg)
    x, x_cat = scale_data(x, x_cat, dynamic_scaler, cat_scaler)
    error = get_recon_loss(x, x_cat, model, device=device)
    dur = time.time()-start

    return error > threshold

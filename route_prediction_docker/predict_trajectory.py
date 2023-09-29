import torch
import mlflow
import pathlib
import numpy as np
import pyproj


def load_models(device='cpu'):
    device = torch.device(device)
    cat_scaler_path = str(pathlib.Path(__file__).parent) + f'/trained_models/cat_scaler'
    cont_scaler_path = str(pathlib.Path(__file__).parent) + f'/trained_models/cont_scaler'
    dyn_scaler_path = str(pathlib.Path(__file__).parent) + f'/trained_models/dynamic_scaler'
    route_model_path = str(pathlib.Path(__file__).parent) + f'/trained_models/model'

    cat_scaler = mlflow.sklearn.load_model(cat_scaler_path)
    cont_scaler = mlflow.sklearn.load_model(cont_scaler_path)
    dynamic_scaler = mlflow.sklearn.load_model(dyn_scaler_path)
    model = mlflow.pytorch.load_model(route_model_path, map_location=device)
    model.eval()

    return cat_scaler, cont_scaler, dynamic_scaler, model


def scale_data(x, x_cat, x_cont, dynamic_scaler, cat_scaler, cont_scaler):
    x = dynamic_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
    x_cat = cat_scaler.transform(x_cat)
    x_cont = cont_scaler.transform(x_cont)

    return x, x_cat, x_cont



"""
def convert_coordinates(x, from_epsg=4326, to_epsg=3035):
    #x: [lon, lat] (x,y)

    transformer = pyproj.Transformer.from_crs(from_epsg, to_epsg, always_xy=True)
    x_y_tensor = np.apply_along_axis(lambda row: transformer.transform(row[0], row[1]), axis=1,
                                     arr=x.reshape(-1, x.shape[-1])).reshape(x.shape)
    #x_y_tensor [E, N] (x,y)
    return x_y_tensor

"""


def convert_coordinates(x, from_epsg=4326, to_epsg=3035):
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


def predict_trajectory(x, x_cat, x_cont, dynamic_scaler, cat_scaler, cont_scaler, model, horizon, device='cpu',
                      from_epsg=4326, to_epsg=3035):
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    if from_epsg != to_epsg:
        x = convert_coordinates(np.array(x), from_epsg=from_epsg, to_epsg=to_epsg)
    if len(x.shape) != 3:
        x = x.unsqueeze(0)
    if len(x_cat.shape) != 2:
        x_cat = x_cat.unsqueeze(0)
    if len(x_cont.shape) != 2:
        x_cont = x_cont.unsqueeze(0)

    x, x_cat, x_cont = scale_data(x, x_cat, x_cont, dynamic_scaler, cat_scaler, cont_scaler=cont_scaler)
    x = torch.tensor(x, requires_grad=False).float()
    x_cat = torch.tensor(x_cat, requires_grad=False).int()
    x_cont = torch.tensor(x_cont, requires_grad=False).float()

    x_in = x
    for i in range(horizon):
        y = model(x=x_in, x_cat=x_cat, x_cont=x_cont)
        if i == 0:
            pred = y
        else:
            pred = torch.concatenate((pred,y), axis=1)

        x_in = torch.concatenate((x_in[:,1:,],y), axis =1)
    pred = pred.detach().numpy()
    pred_ = dynamic_scaler.inverse_transform(pred.reshape(-1, pred.shape[-1])).reshape(pred.shape)
    if from_epsg != to_epsg:
        out = convert_coordinates(np.array(pred_), from_epsg=to_epsg, to_epsg=from_epsg)
    else:
        out = pred_
    return out

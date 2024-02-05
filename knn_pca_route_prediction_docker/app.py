import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import hdbscan
from scipy.spatial import distance
from flask import Flask, request, jsonify
import pyproj

n_knn = 50  # numer of nearest neighbors used

with open("./train_data_dest_50.json", "r") as json_file:
    data = json.load(json_file)


def unpack_data(data, key):

    try:
        data = data[key]
        src = np.array(data["src"])
        src_data = src.reshape(src.shape[0], -1)
        trg = np.array(data["trg"])
        trg_data = trg.reshape(trg.shape[0], -1)
    except:

        i = 0
        for key in data.keys():
            if i == 0:
                src = np.array(data[key]["src"])
                trg = np.array(data[key]["trg"])
            else:
                src = np.append(src, data[key]["src"], axis=0)
                trg = np.append(trg, data[key]["trg"], axis=0)
            i += 1

        src_data = src.reshape(src.shape[0], -1)
        trg_data = trg.reshape(trg.shape[0], -1)



    return src, src_data, trg, trg_data


def calc_similarity(input, src):
    distances = distance.cdist(input, src, 'euclidean')
    return distances


def get_similar_indexes(input, src, n):
    distances = calc_similarity(input, src)
    if distances.shape[0]:
        distances = distances[0]

    return np.argsort(distances)[:n], np.sort(distances)[:n]


def mc_sampling(x, w, n_samples):
    weighted_means = np.average(x, axis=0, weights=w)
    weighted_cov = np.cov(x, aweights=w, rowvar=False)
    samples = np.random.multivariate_normal(weighted_means, weighted_cov, n_samples)
    return samples


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


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    request_data = request.json
    epsg = request_data["EPSG"]
    input = np.array(request_data["trajectory_data"])
    horizon = request_data["horizon"]
    if epsg != 3035:
        input = convert_coordinates(input, from_epsg=epsg, to_epsg=3035)

    destination = request_data["destination"]

    key = destination
    """
    Get data
    """
    src, src_data, trg, trg_data = unpack_data(data, key)

    """
    Initialize classes for clusering, classification and PCA
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
    classifer = KNeighborsClassifier(n_neighbors=1, weights="distance")
    pca = PCA(n_components=3)

    """
    Evaluate similarity
    """

    src_idx, distances = get_similar_indexes(input.reshape(1, -1), src_data, n=n_knn)

    """
    Get weights
    """
    weights = 1 / (distances[:n_knn] ** 2)

    """
    Get PCA representation of targets for nearest neighbors
    """
    selected_trg = trg_data[src_idx[:n_knn]]
    trg_pca = pca.fit_transform(selected_trg)
    """
    Cluster pca values of target
    """
    labels = clusterer.fit(trg_pca).labels_
    """
    Classify input to one of clusters using kNN classifier
    """
    try:
        classifer.fit(X=src_data[src_idx[:n_knn]][labels != -1], y=labels[labels != -1])
    except:
        classifer.fit(X=src_data[src_idx[:n_knn]], y=labels)

    cluster_nr = classifer.predict(input.reshape(1, -1))  # Find cluster of data point


    """
    Get PCA representation of cluster data
    """
    latent = pca.fit_transform(selected_trg[labels == cluster_nr])

    """
    Update weights
    """
    weights = weights[labels == cluster_nr]
    weights /= np.sum(weights)
    """
    Get weighted average of latent representation
    """
    average = np.average(latent, axis=0, weights=weights)
    mean_pred = pca.inverse_transform(average).reshape(trg.shape[1:])
    mean_pred = convert_coordinates(mean_pred, from_epsg=3035, to_epsg=epsg)

    response = {
        'prediction': mean_pred[4:horizon, :].tolist(),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)

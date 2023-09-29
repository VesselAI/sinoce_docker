import numpy as np
from geopandas import GeoDataFrame
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
# from model_development.utils.db_utils import get_geo_data
#from utils.db_utils import get_geo_data



class DataPrep():
    def __init__(self):
        pass

    def fit(self, gdf, dynamic_features, categorical_features=['mmsi'], continuous_features=['length'],
            dynamic_range=(0,1), continuous_range=(0,1), test_size=0.2, val_size=0.25, test_train_random_state=None):

        self.scale_dynamic_data(gdf, features=dynamic_features, feature_range=dynamic_range)

        self.n_routes = len(self.scaled_dynamic_data)
        self.test_size = test_size
        self.val_size = val_size
        self.test_train_random_state = test_train_random_state


        self.train_idx, self.test_idx = train_test_split(np.arange(self.n_routes), test_size=self.test_size,
                                               shuffle=True, random_state=self.test_train_random_state)
        self.train_idx, self.val_idx = train_test_split(self.train_idx, test_size=self.val_size, shuffle=False,
                                              random_state=self.test_train_random_state)

        if categorical_features:
            self.scale_categorical_data(gdf, features=categorical_features)

        if continuous_features:
            self.scale_continuous_data(gdf, features=continuous_features, feature_range=continuous_range)

    """
    Sliding window splitting
    """

    def get_sliding_window(self, window_size=10, idx=None, to_torch=True):
        src = []
        for i in range(len(self.scaled_dynamic_data)):
            x = self.scaled_dynamic_data[i]
            unpacked_src = [torch.from_numpy(x[j:j + window_size, :]).unsqueeze(0) for j in
                            range(0, x.shape[0] - window_size + 1)]
            src.extend(unpacked_src)

        if to_torch:
            return torch.cat([torch.from_numpy(item).unsqueeze(0) for item in src], dim=0).float()
        else:
            return src

    """
    Data getters
    """

    def get_dynamic_data(self, gdf: GeoDataFrame, features):
        data_list = []
        for i in range(gdf.shape[0]):
            data = np.array(gdf.iloc[i]['line_geom'].xy).T

            if features is not None:
                for feature in features:
                    if feature == 'cog':
                        cog = gdf.iloc[i][feature]
                        cog_x = np.cos(cog)
                        cog_y = np.sin(cog)
                        data = np.concatenate((data, np.atleast_2d(cog_x).T), axis=1)
                        data = np.concatenate((data, np.atleast_2d(cog_y).T), axis=1)

                    else:
                        data = np.concatenate((data, np.atleast_2d(gdf.iloc[i][feature]).T), axis=1)

            data_list.append(data)
        self.dynamic_data = data_list #len nr routes

    def get_categorical_data(self, gdf: GeoDataFrame, features):

        self.categorical_data = gdf[features] #(n_routes, n_features)

    def get_continuous_data(self, gdf: GeoDataFrame, features):

        self.continuous_data = gdf[features] #(n_routes, n_features)

    """
    Scaler getters
    """
    def get_dynamic_scaler(self, feature_range):
        data = np.vstack(self.dynamic_data)
        dyn_scaler = MinMaxScaler(feature_range=feature_range)
        self.dynamic_scaler = dyn_scaler.fit(data)

    def get_categorical_scaler(self):

        categorical_scaler = OrdinalEncoder()
        self.categorical_scaler = categorical_scaler.fit(pd.DataFrame(self.categorical_data))

    def get_continuous_scaler(self, feature_range):
        continuous_scaler = MinMaxScaler(feature_range=feature_range)
        self.continuous_scaler = continuous_scaler.fit(self.continuous_data)

    """
    Scaling functions
    """

    def scale_dynamic_data(self, gdf, features, feature_range=(0, 1), dynamic_scaler=None):
        self.dynamic_crs = gdf.crs
        self.dynamic_features = ['lon', 'lat']
        if features:
            self.dynamic_features.extend(features)
        self.get_dynamic_data(gdf=gdf, features=features)
        if dynamic_scaler == None:
            self.get_dynamic_scaler(feature_range=feature_range)
        else:
            self.dynamic_scaler = dynamic_scaler
        self.scaled_dynamic_data = [self.dynamic_scaler.transform(x) for x in self.dynamic_data]

    def scale_categorical_data(self, gdf, features=['mmsi'], max_emb_size=3):

        self.categorical_features = features
        self.get_categorical_data(gdf=gdf, features=features)
        self.get_categorical_scaler()
        self.get_cat_emb_size(max_emb_size=max_emb_size)
        self.scaled_categorical_data = self.categorical_scaler.transform(self.categorical_data)

    def scale_continuous_data(self, gdf, features=['length'], feature_range=(0, 1)):
        self.get_continuous_data(gdf, features=features)
        self.get_continuous_scaler(feature_range=feature_range)
        self.scaled_continuous_data = self.continuous_scaler.transform(self.continuous_data)

    """
    For embeddings of categorical data
    """
    def get_cat_emb_size(self, max_emb_size):

        cat_szs = [len(item) for item in self.categorical_scaler.categories_]
        emb_szs = [(size, max_emb_size) for size in cat_szs]
        self.cat_emb_szs = emb_szs


def get_sliding_window(data: list, window_size:int =10, idx:list = None, to_torch=False, cat_data=None):
    """
    data[i] (sequence_len, n_features)
    """

    split_data = []
    split_cat_data = []
    if idx is None:
        data_idx = range(len(data))
    else:
        data_idx = idx

    for i in data_idx:
        x = data[i]
        unpacked_src = [x[j:j + window_size, :] for j in range(0, x.shape[0] - window_size + 1)]
        split_data.extend(unpacked_src)
        if cat_data is not None:
            x_cat = [cat_data[i, :] for _ in range(len(unpacked_src))]
            split_cat_data.extend(x_cat)

    """
    Add continuous data here (also in output
    """
    if to_torch:
        #batch first (dim=0)
            if cat_data is not None:
                return torch.cat([torch.from_numpy(item).unsqueeze(0) for item in split_data], dim=0).float(), \
                       torch.cat([torch.from_numpy(item).unsqueeze(0) for item in split_cat_data], dim=0).int()
            else:
                return torch.cat([torch.from_numpy(item).unsqueeze(0) for item in split_data], dim=0).float(), None

    else:
        return split_data, split_cat_data


def get_static_segments_dyn(dyn_data, stat_data, input_length, output_length, to_tensor=True, step=1):
    static = []
    for i in range(len(dyn_data)):
        x = stat_data[i]
        d = dyn_data[i]

        unpacked_stat = [torch.from_numpy(x).unsqueeze(0) for j in
                         range(0, d.shape[0] - input_length - output_length + 1)]
        static.extend(unpacked_stat)

    if to_tensor:
        static = torch.cat(static, dim=0).float()
    else:
        pass

    return static

if __name__ == "__main__":

    gdf = get_geo_data(schema='data', tablename='routes_all', cluster_nr=25, crs='EPSG:4326')
    data = DataPrep()
    data.fit(gdf)
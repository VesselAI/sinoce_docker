import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from anomaly_detection.pre_process import DataPrep, get_sliding_window
import torch
from geopandas.geodataframe import GeoDataFrame


class AISDataset(Dataset):
    # Custom Pytorch dataset
    def __init__(self, data, window_size, idx, device: torch.device):
        super().__init__()
        src, src_cat = get_sliding_window(data.scaled_dynamic_data, window_size=window_size, idx=idx, to_torch=True,
                                 cat_data=data.scaled_categorical_data)

        self.x, self.y, self.x_cat = src.to(device), src.to(device), src_cat.to(device)

        """
        self.x, self.y = get_traj_segments([data.dyn_data[idx] for idx in data_idx],
                                           input_length=input_len, output_length=output_len, to_tensor=True, step=step)
        
        self.x_cat = get_static_segments_dyn(dyn_data=[data.dyn_data[idx] for idx in data_idx],
                                         stat_data=[data.cat_data[idx, :] for idx in data_idx],
                                         input_length=input_len, output_length=output_len, to_tensor=True, step=step).long()

        self.x_cont = get_static_segments_dyn(dyn_data=[data.dyn_data[idx] for idx in data_idx],
                                          stat_data=[data.cont_data[idx, :] for idx in data_idx],
                                          input_length=input_len, output_length=output_len, to_tensor=True, step=step)
        """

    def __len__(self):
        # Number of samples in our dataset
        return self.x.shape[0]

    def __getitem__(self, idx):

        return self.x[idx, :, :], self.y[idx, :, :], self.x_cat[idx, :]

        #return self.x[idx, :, :], self.y[idx, :, :], self.x_cat[idx, :], self.x_cont[idx, :]

    def cuda(self):
        if torch.cuda.is_available():
            self.x = self.x.to('cuda')
            self.y = self.y.to('cuda')

class AEDataModule(pl.LightningDataModule):
    # Load data into our model:
    def __init__(self, gdf: GeoDataFrame, batch_size: int, device: torch.device, test_size=0.2, val_size=0.25,
                 window_size=10, test_train_random_state=22, dynamic_features=None,
                 categorical_features=None):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.window_size = window_size
        self.device = device
        self.test_train_random_state = test_train_random_state
        data = DataPrep()
        data.fit(gdf=gdf, test_size=self.test_size, val_size=self.val_size,
                 test_train_random_state=self.test_train_random_state, dynamic_features=dynamic_features,
                 categorical_features=categorical_features)
        self.data = data
        self.prepare_data_per_node = True
        self.input_dim = self.data.scaled_dynamic_data[0].shape[-1]
        self.emb_szs = data.cat_emb_szs

    def prepare_data(self):
        pass

    def setup(self, stage):

        self.train_dataset = AISDataset(data=self.data, window_size=self.window_size, idx=self.data.train_idx,
                                        device=self.device)
        self.val_dataset = AISDataset(data=self.data, window_size=self.window_size, idx=self.data.val_idx,
                                      device=self.device)
        self.test_dataset = AISDataset(data=self.data, window_size=self.window_size, idx=self.data.test_idx,
                                       device=self.device)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


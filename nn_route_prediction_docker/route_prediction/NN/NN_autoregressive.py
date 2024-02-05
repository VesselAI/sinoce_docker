import torch
import torch.nn as nn
import pytorch_lightning as pl
from .datasets import NNDataModule



class NN(pl.LightningModule):
    def __init__(self, feature_dim: int, input_len: int, output_len:int, emb_szs: list, hid_dim: int, lr: float):
        super().__init__()
        self.save_hyperparameters()  # Automtically saves all hyperparams passed in init.
        self.feature_dim = feature_dim
        self.hid_dim = hid_dim
        self.output_len = output_len
        self.lr = lr
        self.criterion = nn.MSELoss()#reduction='mean')
        self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
        self.static_dim = sum([item[1] for item in emb_szs])
        self.relu = nn.ReLU()
        """
        Remove batch norm for earlier model versions
        """
        #self.bn = nn.BatchNorm1d(hid_dim)
        """
        """
        self.fc_1= nn.Linear(input_len*feature_dim+self.static_dim+1, hid_dim)  # Remove +2 if not lookig at last state, +1 for length addition
        #self.fc_out1 = nn.Linear(hid_dim + stat_dim, int((hid_dim + stat_dim) / 2))
        self.fc_out = nn.Linear(hid_dim, output_len*feature_dim)

        nn.init.kaiming_uniform_(self.fc_1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc_1.bias)

    def forward(self, x, x_cat=None, x_cont=None):  # x dim [input_len, batch_size, feature_dim]
        # debugging: x dim [batch_size, input_len, feature_dim]
        embed_vals = []
        for i, e in enumerate(self.embeddings):
            embed_vals.append(e(x_cat[:, i]))

        x_cat = torch.cat(embed_vals, 1)
        #static = torch.cat((x_cat, x_cont), 1)

        #output, hidden = self.rnn(x)
        # pred_in = torch.cat((hidden.squeeze(0), static), 1).unsqueeze(1)

        x_in = torch.cat((x.view(x.shape[0], -1), x_cat, x_cont), axis=1)  # x[:,-1,:] allows model to look at last state

        """
        Experimentation
        For testing removing static data

        #pred_in = hidden
        """
        m = nn.ReLU()
        """
        With batch norm
        """
        #prediction = self.fc_out(m(self.bn(self.fc_1(x_in))))
        """
        Without batch norm
        """
        prediction = self.fc_out(m(self.fc_1(x_in)))
        """
        """

        prediction = prediction.view(x.shape[0], self.output_len, self.feature_dim)



        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch: NNDataModule, batch_idx: int):
        x, y, x_cat , x_cont = batch
        y_pred = self(x, x_cat, x_cont)
        loss = self.criterion(y_pred, y)
        # loss = self.criterion(torch.transpose(y_pred, 1, 0), y) # need to transpose since y is given as batch first

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: NNDataModule, batch_idx: int):
        x, y, x_cat, x_cont = batch
        y_pred = self(x, x_cat, x_cont)
        val_loss = self.criterion(y_pred, y)
        # val_loss = self.criterion(torch.transpose(y_pred, 1, 0), y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
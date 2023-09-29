import torch
import torch.nn as nn
import pytorch_lightning as pl
from anomaly_detection.datasets import AISDataset
from anomaly_detection.types_ import Tensor


class Encoder(pl.LightningModule):
    def __init__(self, input_dim, hid_dim, bidirectional=False, batch_first=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.bidirectional = bidirectional
        # input_dim – The number of expected features in the input x
        # hid_dim – The number of features in the hidden state h
        # batch_first input given as (batch_size, sequence length, feature_size)

        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hid_dim, bidirectional=bidirectional,
                          batch_first=batch_first)

    def forward(self, x: Tensor):
        # input of shape (batch, seq_len, input_size) if batch_first=True
        outputs, hidden = self.rnn(x) #(input_data.flip(0))  # flip input if padded to have padded values first
        return hidden


class Decoder(pl.LightningModule):
    def __init__(self, static_dim: int, output_dim: int, hid_dim: int, bidirectional: bool = False, batch_first: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.rnn = nn.GRU(output_dim+static_dim, hid_dim, bidirectional=bidirectional, batch_first=batch_first)
        if bidirectional:
            self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        else:
            self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, input, hidden):

        if torch.cuda.is_available():
            output, hidden = self.rnn(input.cuda(), hidden)
        else:
            output, hidden = self.rnn(input, hidden)

        prediction = self.fc_out(output)
        return prediction, hidden




class RAE(pl.LightningModule):
    def __init__(self, input_dim: int, hid_dim: int, bidirectional: bool, lr: float, emb_szs: list):
        super().__init__()
        self.criterion = nn.MSELoss()#reduction='mean')
        self.bidirectional = bidirectional

        if emb_szs is not None:
            self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])
            self.static_dim = sum([item[1] for item in emb_szs]) #+ data.cont_data.shape[1]

        else:
            self.static_dim = 0
        self.encoder = Encoder(input_dim, hid_dim, bidirectional=bidirectional, batch_first=True)
        self.decoder = Decoder(static_dim=self.static_dim, output_dim=input_dim, hid_dim=hid_dim,
                               bidirectional=bidirectional, batch_first=True)
        self.lr = lr


    def forward(self, x, x_cat):

        embed_vals = []
        for i, e in enumerate(self.embeddings):
            embed_vals.append(e(x_cat[:, i]))

        x_cat = torch.cat(embed_vals, 1)
        #static = torch.cat((x_cat, x_cont), 1)

        outputs = torch.zeros(x.shape)
        if torch.cuda.is_available():
            outputs.cuda()

        # last hidden state of the encoder is used as the initial hidden state of the decoder


        hidden = self.encoder(x)

        #hidden = hidden.view(1, hidden.shape[1], -1)

        output = torch.zeros(x.shape[0], 1, x.shape[2])


        for t in range(x.shape[1]):
            if torch.cuda.is_available():
                input = torch.cat((output.cuda(), x_cat.unsqueeze(1).cuda()), -1)  # TODO check cat dim
            else:
                input = torch.cat((output, x_cat.unsqueeze(1)), -1)

            output, hidden = self.decoder(input, hidden)

            outputs[:, t, :] = output.squeeze(1)

        return outputs
        #return torch.where(data.padded_f == 0, torch.zeros(outputs.shape), outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch: AISDataset, batch_idx: int):
        x, y, x_cat = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x_cat = x_cat.to(self.device)
        y_pred = self(x, x_cat)
        if torch.cuda.is_available():
            loss = self.criterion(y_pred.cuda(), y.cuda())
        else:

            loss = self.criterion(y_pred, y)
        #loss = self.criterion(torch.transpose(y_pred, 1, 0), y) # need to transpose since y is given as batch first

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: AISDataset, batch_idx: int):
        x, y, x_cat = batch
        x = x.to(self.device)
        y = y.to(self.device)
        x_cat = x_cat.to(self.device)
        y_pred = self(x, x_cat)
        if torch.cuda.is_available():
            val_loss = self.criterion(y_pred.cuda(), y.cuda())
        else:
            val_loss = self.criterion(y_pred, y)
        #val_loss = self.criterion(torch.transpose(y_pred, 1, 0), y)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

"""
def get_hidden(model, data):
    model.eval()
    x = data.train_pad
    data.pad()
    hidden = model.encoder(data.packed_data)
    hidden = hidden.view(1, hidden.shape[1], -1)
    hidden = model.hidden2dec(hidden).detach().numpy()

    return hidden
"""





#TODO append origin and destination LOCODEs to each route based on cluster_nr
#TODO then give orign and destination as categorical features







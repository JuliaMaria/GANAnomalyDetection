import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, signal_shape=100):
        super(Encoder, self).__init__()
        self.signal_shape = signal_shape
        ###
        self.lstm = nn.Transformer(num_encoder_layers=1, num_decoder_layers=1, d_model=100, nhead=10)
        ###
        self.dense = nn.Linear(in_features=100, out_features=20)

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x = self.lstm(x, x)
        x = self.dense(x)
        return (x)


class Decoder(nn.Module):
    def __init__(self, signal_shape=100):
        super(Decoder, self).__init__()
        self.signal_shape = signal_shape
        ###
        self.lstm = nn.Transformer(num_encoder_layers=2, num_decoder_layers=2, d_model=20, nhead=10)
        ###
        self.dense = nn.Linear(in_features=20, out_features=self.signal_shape)

    def forward(self, x):
        x = self.lstm(x, x)
        x = self.dense(x)
        return (x)


class CriticX(nn.Module):
    def __init__(self, signal_shape=100):
        super(CriticX, self).__init__()
        self.signal_shape = signal_shape
        self.dense1 = nn.Linear(in_features=self.signal_shape, out_features=20)
        self.dense2 = nn.Linear(in_features=20, out_features=1)

    def forward(self, x):
        x = x.view(1, 64, self.signal_shape).float()
        x = self.dense1(x)
        x = self.dense2(x)
        return (x)


class CriticZ(nn.Module):
    def __init__(self):
        super(CriticZ, self).__init__()
        self.dense1 = nn.Linear(in_features=20, out_features=1)

    def forward(self, x):
        x = self.dense1(x)
        return (x)

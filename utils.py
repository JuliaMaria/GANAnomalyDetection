from torch.utils.data import Dataset
import pandas as pd
import torch

class SignalDataset(Dataset):
    def __init__(self, path):
        self.signal_df = pd.read_csv(path)
        self.signal_columns = self.make_signal_list()
        self.make_rolling_signals()

    def make_signal_list(self):
        signal_list = list()
        for i in range(-50, 50):
            signal_list.append('signal' + str(i))
        return signal_list

    def make_rolling_signals(self):
        for i in range(-50, 50):
            self.signal_df['signal' + str(i)] = self.signal_df['signal'].shift(i)
        self.signal_df = self.signal_df.dropna()
        self.signal_df = self.signal_df.reset_index(drop=True)

    def __len__(self):
        return len(self.signal_df)

    def __getitem__(self, idx):
        row = self.signal_df.loc[idx]
        x = row[self.signal_columns].values.astype(float)
        x = torch.from_numpy(x)
        return {'signal': x, 'anomaly': row['anomaly']}
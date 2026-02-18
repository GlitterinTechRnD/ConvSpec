import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from AnalyticsFunctions import custom_train_test_split, get_regression_metrics
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for plotting

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu'])
    args = parser.parse_args()

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(args.device)
    return args

args = parse_args()
device = args.device
epochs = args.epochs
patience = args.patience
batch_size = args.batch_size

# Load data
data_path = 'Blood_glucose_spectral_datasets_single_participents.xlsx'

train_df = pd.read_excel(
    data_path, sheet_name='train_data', header=0, index_col=0
)
test_df = pd.read_excel(
    data_path, sheet_name='test_data', header=0, index_col=0
)

X_all = train_df.iloc[:, :-1].values.astype(np.float32)
y_all = train_df.iloc[:, -1].values.astype(np.float32)

X_test = test_df.iloc[:, :-1].values.astype(np.float32)
y_test = test_df.iloc[:, -1].values.astype(np.float32)

# for validation
X_train, X_val, y_train, y_val = custom_train_test_split(X_all, y_all, test_size=0.2, method='SPXY')

class SpeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_loader = DataLoader(
    SpeDataset(X_train, y_train),
    batch_size=batch_size, shuffle=True
)
val_loader = DataLoader(
    SpeDataset(X_val, y_val),
    batch_size=batch_size, shuffle=False
)
test_loader = DataLoader(
    SpeDataset(X_test, y_test),
    batch_size=batch_size, shuffle=False
)

class SpectralNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        return self.regressor(x)


model = SpectralNet(X_train.shape[1]).to(device)
state_dict = torch.load('best_model.pth', weights_only=True)
model.load_state_dict(state_dict)
model.eval()
with torch.no_grad():
    y_train_pred = model(
        torch.FloatTensor(X_train).to(device)
    ).cpu().numpy().ravel()

    y_test_pred = model(
        torch.FloatTensor(X_test).to(device)
    ).cpu().numpy().ravel()
results = get_regression_metrics(y_test, y_test_pred)
print(results)
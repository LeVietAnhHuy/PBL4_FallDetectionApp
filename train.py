from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from textwrap import dedent
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.mobile_optimizer import optimize_for_mobile

seed = 1
np.random.seed(seed)
trn_sz = 3401  # only the first `trn_sz` rows in each array include labelled data

# path = '/content/drive/MyDrive/PBL4_FallDetectionAndroidApp/data'
path  = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/gen_data'

sensor_data_array = np.load(os.path.join(path, 'sensor_data_array.npy')).transpose(0, 2, 1)
label_array = np.load(os.path.join(path, 'label_array.npy'), allow_pickle=True)
label_array = label_array.astype(np.float64)

def create_datasets(data, target, train_size, valid_pct=0.1, seed=None):
  """Converts NumPy arrays into PyTorch datsets.

    Three datasets are created in total:
        * training dataset
        * validation dataset
        * testing (un-labelled) dataset

  """
  raw = data
  sz = train_size
  idx = np.arange(sz)
  train_idx, val_idx = train_test_split(idx, test_size=valid_pct, random_state=seed)

  train_ds = TensorDataset(
      torch.tensor(raw[:sz][train_idx]).float(),
      torch.tensor(target[:sz][train_idx]).long()
  )
  val_ds = TensorDataset(
      torch.tensor(raw[:sz][val_idx]).float(),
      torch.tensor(target[:sz][val_idx]).long()
  )
  test_ds = TensorDataset(
      torch.tensor(raw[:sz]).float(),
      torch.tensor(target[:sz]).long()
  )

  return train_ds, val_ds, test_ds

def create_loaders(data, bs=128, jobs=0):
  """Wraps the datasets returned by create_datasets function with data loaders."""

  train_ds, val_ds, test_ds = data
  train_dl = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=jobs)
  val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=jobs)
  test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=jobs)

  return train_dl, val_dl, test_dl

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.

    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):

        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)

class Classifier(nn.Module):
    def __init__(self, raw_ni, no, drop=.5):
        super().__init__()

        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 5, 2, 3, drop=drop),
            SepConv1d(    32,  64, 5, 4, 2, drop=drop),
            SepConv1d(    64, 128, 5, 4, 2, drop=drop),
            SepConv1d(   128, 256, 5, 4, 2),
            Flatten(),
            nn.Dropout(drop), nn.Linear(1792, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(64, 64), nn.ReLU(inplace=True),
            nn.Linear(64, no))

        # self.fft = nn.Sequential(
        #     SepConv1d(fft_ni,  32, 8, 2, 4, drop=drop),
        #     SepConv1d(    32,  64, 8, 2, 4, drop=drop),
        #     SepConv1d(    64, 128, 8, 4, 4, drop=drop),
        #     SepConv1d(   128, 128, 8, 4, 4, drop=drop),
        #     SepConv1d(   128, 256, 8, 2, 3),
        #     Flatten(),
        #     nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
        #     nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))

        # self.out = nn.Sequential(
        #     nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, no))

    def forward(self, t_raw,):
        raw_out = self.raw(t_raw)
        # fft_out = self.fft(t_fft)
        # t_in = torch.cat([raw_out, fft_out], dim=1)
        # out = self.out(t_in)
        out = raw_out
        return out

datasets = create_datasets(sensor_data_array, label_array, trn_sz, seed=seed)
# make sure that we run on a proper device (not relevant for Kaggle kernels but helpful in Jupyter sessions)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO',
              'LYI', 'FOL', 'FKL', 'BSC', 'SDL']
# path = '/content/drive/MyDrive/PBL4_FallDetectionAndroidApp/trained_model'
path = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight'
raw_feat = sensor_data_array.shape[1]

trn_dl, val_dl, tst_dl = create_loaders(datasets, bs=256*40)

lr = 0.001
n_epochs = 3000
iterations_per_epoch = len(trn_dl)
num_classes = len(activities)
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
loss_history = []
acc_history = []

model = Classifier(raw_feat, num_classes).to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=lr)

print('Start model training')

for epoch in range(1, n_epochs + 1):

    model.train()
    epoch_loss = 0
    for i, batch in enumerate(trn_dl):
        x_raw, y_batch = [t.to(device) for t in batch]
        opt.zero_grad()
        out = model(x_raw)
        loss = criterion(out, y_batch)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()

    epoch_loss /= trn_sz
    loss_history.append(epoch_loss)

    model.eval()
    correct, total = 0, 0
    for batch in val_dl:
        x_raw, y_batch = [t.to(device) for t in batch]
        out = model(x_raw)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()

    acc = correct / total
    acc_history.append(acc)

    if epoch % base == 0:
        print(f'Epoch: {epoch:3d}. Loss: {epoch_loss:.4f}. Acc.: {acc:2.2%}')
        base *= step

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), os.path.join(path, 'best.pth'))
        print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
    else:
        trials += 1
        if trials >= patience:
            print(f'Early stopping on epoch {epoch}')
            break

path = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/best.pth'
model = Classifier(raw_feat, num_classes)
model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
model.eval()
model_script = torch.jit.script(model)
torchscript_model_optimized = optimize_for_mobile(model_script)
torchscript_model_optimized._save_for_lite_interpreter('/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/best_mobile.pt')
print('Done!')


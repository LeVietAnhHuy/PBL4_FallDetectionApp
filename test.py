import os
from tsai.all import *
import sklearn.metrics as skm
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from model.TCN_Separable_fft import TCN1 #TCN1 TCN_Separable
from torch import optim
from tqdm import tqdm
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from data_loader.sensor_data_loader import create_loaders, create_datasets
from model.TCN_fft import Model

path = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/gen_data'
path_pt = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/best.pth'
raw_data = np.load(os.path.join(path, 'raw_data.npy')).transpose(0, 2, 1)
fft_data = np.load(os.path.join(path, 'fft_data.npy')).transpose(0, 2, 1)
label = np.load(os.path.join(path, 'label.npy'), allow_pickle=True)
label = label.astype(np.float64)

activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO',
              'LYI', 'FOL', 'FKL', 'BSC', 'SDL']
seed = 1
np.random.seed(seed)
trn_sz = 6000
datasets = create_datasets((raw_data, fft_data), label, trn_sz, seed=seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
raw_feat = raw_data.shape[1]
fft_feat = fft_data.shape[1]

trn_dl, val_dl, tst_dl = create_loaders(datasets, bs=256)
model = Model().to(device)
model.load_state_dict(torch.load(path_pt))
criterion = nn.CrossEntropyLoss(reduction='sum')
print('Start model Testing')
model.eval()
correct, total = 0, 0
for batch in tqdm(tst_dl):
    x_raw, x_fft, y_batch = [t.to(device) for t in batch]
    out = model(x_raw, x_fft)
    preds = F.log_softmax(out, dim=1).argmax(dim=1)
    total += y_batch.size(0)
    correct += (preds == y_batch).sum().item()

acc = correct / total
print('Accuracy: ', acc)
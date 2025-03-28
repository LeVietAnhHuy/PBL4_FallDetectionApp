import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import pandas as pd
# from sklearn.externals import joblib
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
from textwrap import dedent
import os
from torch.nn.utils import weight_norm
import numpy as np
import matplotlib.pyplot as plt
from tsai.all import *
import sklearn.metrics as skm
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

path = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/gen_data'
sensor_data_array = np.load(os.path.join(path, 'sensor_data_array.npy')).transpose(0, 2, 1)
label_array = np.load(os.path.join(path, 'label_array.npy'), allow_pickle=True)
label_array = label_array.astype(np.float64)

activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO',
              'LYI', 'FOL', 'FKL', 'BSC', 'SDL']

X_train, X_test, y_train, y_test = train_test_split(sensor_data_array, label_array, random_state=0, train_size = .75)
X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)

class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = weight_norm(nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni))
        self.pointwise = weight_norm(nn.Conv1d(ni, no, kernel_size=1))
        self.downsample = nn.Conv1d(ni, no, 1) if ni != no else None
        self.init_weights()

    def init_weights(self):
        self.depthwise.weight.data.normal_(0, 0.01)
        self.pointwise.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
        self.flatten = Reshape()
    def forward(self, x):
        return self.flatten(self.gap(x))

class AdaptiveConcatPool1d(nn.Module):
    "Layer that concats `AdaptiveAvgPool1d` and `AdaptiveMaxPool1d`"
    def __init__(self, size=None):
        super(AdaptiveConcatPool1d, self).__init__()
        self.size = size or 1
        self.ap = nn.AdaptiveAvgPool1d(self.size)
        self.mp = nn.AdaptiveMaxPool1d(self.size)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Reshape(nn.Module):
    def __init__(self, *shape):
      super(Reshape, self).__init__()
      self.shape = shape
    def forward(self, x):
        return x.reshape(x.shape[0], -1) if not self.shape else x.reshape(-1) if self.shape == (-1,) else x.reshape(x.shape[0], *self.shape)
    def __repr__(self): return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"

class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()
        # self.conv1 = weight_norm(nn.Conv1d(ni,nf,ks,stride=stride,padding=padding,dilation=dilation))
        self.depthwise = weight_norm(nn.Conv1d(ni, ni, ks, stride, padding=padding, groups=ni))
        self.pointwise = weight_norm(nn.Conv1d(ni, nf, kernel_size=1))
        self.conv1 = self.pointwise(self.depthwise())
        # self.conv1 = SepConv1d(ni, nf, ks, stride=stride, pad=padding)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # self.conv2 = weight_norm(nn.Conv1d(nf,nf,ks,stride=stride,padding=padding,dilation=dilation))
        # self.conv2 = SepConv1d(nf,  nf, ks, stride=stride, pad=padding)
        self.depthwise = weight_norm(nn.Conv1d(nf, nf, ks, stride, padding=padding, groups=ni))
        self.pointwise = weight_norm(nn.Conv1d(nf, nf, kernel_size=1))
        self.conv2 = self.pointwise(self.depthwise())
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(ni,nf,1) if ni != nf else None
        self.relu = nn.ReLU()
        # self.init_weights()

    # def init_weights(self):
    #     self.conv1.weight.data.normal_(0, 0.01)
    #     self.conv2.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        res = 0
        return self.relu(out + res)

def TemporalConvNet(c_in, layers, ks=2, dropout=0.):
    temp_layers = []
    for i in range(len(layers)):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i-1]
        nf = layers[i]
        temp_layers += [TemporalBlock(ni, nf, ks, stride=1, dilation=dilation_size, padding=(ks-1) * dilation_size, dropout=dropout)]
    return nn.Sequential(*temp_layers)

class TCN1(nn.Module):
    def __init__(self, c_in, c_out, layers=8*[25], ks=7, conv_dropout=0., fc_dropout=0.):
        super().__init__()
        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1],c_out)
    #     self.init_weights()
    #
    # def init_weights(self):
    #     self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)



path = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/trained_model_learned.pth'
model  = TransformerRNNPlus(dls.vars, dls.c, 800)

state_dict = torch.load(path, map_location='cpu')
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_state_dict[k.replace("module.", "")] = v
model.load_state_dict(dict([(n, p) for n, p in state_dict['model'].items()]), strict=False)
model.eval()
model_script = torch.jit.script(model)
torchscript_model_optimized = optimize_for_mobile(model_script)
torchscript_model_optimized._save_for_lite_interpreter('/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/trained_model_learned_mobile.pt')
print('Done!')
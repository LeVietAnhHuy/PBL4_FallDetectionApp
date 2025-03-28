import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

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

class SepConv1d(nn.Module):
    """A simple separable convolution implementation.

    The separable convlution is a method to reduce number of the parameters
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, ks, stride, dilation, padding):
        super(SepConv1d, self).__init__()
        self.depthwise = nn.Conv1d(ni, ni, ks, stride, dilation=dilation, padding=padding)
        self.pointwise = weight_norm(nn.Conv1d(ni, no, kernel_size=1))
        self.act = nn.ReLU()
        # self.downsample = nn.Conv1d(ni, nf, 1) if ni != nf else None

        self.depthwise.weight.data.normal_(0, 0.01)
        self.pointwise.weight.data.normal_(0, 0.01)
        # if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)
    def init_weights(self):
        self.depthwise.weight.data.normal_(0, 0.01)
        self.pointwise.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.pointwise(self.depthwise(x))
        return x

########################################################################################################################

class TemporalBlock(nn.Module):
    def __init__(self, ni, nf, ks, stride, dilation, padding, dropout=0.):
        super(TemporalBlock, self).__init__()
        # self.conv1 = weight_norm(nn.Conv1d(ni,nf,ks,stride=stride,padding=padding,dilation=dilation))
        self.depthwise1 = weight_norm(nn.Conv1d(ni, ni, ks, stride=stride, dilation=dilation, padding=padding, groups=ni))
        self.pointwise1 = weight_norm(nn.Conv1d(ni, nf, kernel_size=1))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.Bn1 = nn.BatchNorm1d(nf)
        self.dropout1 = nn.Dropout(dropout)
        # self.conv2 = weight_norm(nn.Conv1d(nf,nf,ks,stride=stride,padding=padding,dilation=dilation))
        self.depthwise2 = weight_norm(nn.Conv1d(nf, nf, ks, stride=stride, dilation=dilation, padding=padding, groups=nf))
        self.pointwise2 = weight_norm(nn.Conv1d(nf, nf, kernel_size=1))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.Bn2 = nn.BatchNorm1d(nf)
        self.dropout2 = nn.Dropout(dropout)
        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
        #                          self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.depthwise1, self.pointwise1, self.chomp1, self.relu1, self.Bn1, self.dropout1,
                                 self.depthwise2, self.pointwise2, self.chomp2, self.relu2, self.Bn2, self.dropout2)
        self.downsample = nn.Conv1d(ni,nf,1) if ni != nf else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.depthwise1.weight.data.normal_(0, 0.01)
        self.depthwise2.weight.data.normal_(0, 0.01)
        self.pointwise1.weight.data.normal_(0, 0.01)
        self.pointwise2.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        # self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None: self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

def TemporalConvNet(c_in, layers, ks=2, dropout=0.):
    temp_layers = []
    for i in range(len(layers)):
        dilation_size = 2 ** i
        ni = c_in if i == 0 else layers[i-1]
        nf = layers[i]
        temp_layers += [TemporalBlock(ni, nf, ks, stride=1, dilation=dilation_size, padding=(ks-1) * dilation_size, dropout=dropout)]
        temp_layers
    return nn.Sequential(*temp_layers)

class Attention(nn.Module):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W = nn.Linear(units, units)
        self.V = nn.Linear(units, 1)

    def forward(self, inputs):
        # Compute attention scores
        score = torch.relu(self.W(inputs))
        # print('score: ', score.shape)
        attention_weights = F.softmax(self.V(score), dim=1)

        # Apply attention weights to input
        context_vector = attention_weights * inputs
        # print(context_vector.shape)
        # context_vector = torch.sum(context_vector, dim=0)

        return context_vector
class TCN1(nn.Module):
    def __init__(self, c_in, c_out, layers=8*[27], ks=7, conv_dropout=0., fc_dropout=0.):
        super(TCN1, self).__init__()
        self.tcn = TemporalConvNet(c_in, layers, ks=ks, dropout=conv_dropout)
        self.gap = GAP1d()
        self.dropout = nn.Dropout(fc_dropout) if fc_dropout else None
        self.linear = nn.Linear(layers[-1], c_out)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        x = self.tcn(x)
        x = self.gap(x)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.TCN1_standard = TCN1(9, 16)
        self.TCN1_fft = TCN1(6, 16)
        self.Attention = Attention(32)
        self.linear = nn.Linear(32, 16)


    def forward(self, x_raw, x_fft):
        b1 = self.TCN1_standard(x_raw)
        b2 = self.TCN1_fft(x_fft)
        b_in = torch.cat([b1, b2], dim=1)
        b_in = self.Attention(b_in)
        # print(b_in.shape)
        b_out = self.linear(b_in)
        # print(b_out.shape)
        return b_out



# Transformers Model, such as bert.
from calflops import calculate_flops
from tsai.all import *
from model.TCN_fft import Model, TCN1
import torch
activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO',
              'LYI', 'FOL', 'FKL', 'BSC', 'SDL']
device = 'cuda:0'
# model = Model().to(device)
model = GRU(9, len(activities)).to(device)
batch_size = 1
input_shape = (batch_size, 9, 800)

flops, macs, params = calculate_flops(model=model,
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4)
print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

path = 'E:/PBL4_FallDetection_App/data/MobiAct_Dataset_v2_0/RawData'
sensor_data_raw = pd.read_csv(os.path.join(path, 'sensor_data.csv'))
label = pd.read_csv(os.path.join(path, 'label.csv'))

sensor_data_raw = pd.merge(sensor_data_raw, label, on='Series_ID')

# encode categorical labels into numerical values
label_encoder = LabelEncoder()
sensor_data_raw['activity'] = label_encoder.fit_transform(sensor_data_raw['activity'])

# return unique values in column 'Series_ID'
sampling_rate = 800
num_series = sensor_data_raw['Series_ID'].nunique()

# sensor_data_raw to npy file
sensor_data_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll']
sensor_data_array = np.array(sensor_data_raw[sensor_data_cols])
sensor_data_array = np.reshape(sensor_data_array, [num_series, sampling_rate, len(sensor_data_cols)])
print(sensor_data_array.shape)
np.save(os.path.join(path, 'sensor_data_array.npy'), sensor_data_array)

label_array = np.array(sensor_data_raw['activity'])
label_array = np.reshape(label_array, [num_series, sampling_rate])
label_array = label_array[:,0]
print(label_array.shape)
np.save(os.path.join(path, 'label_array.npy'), label_array)







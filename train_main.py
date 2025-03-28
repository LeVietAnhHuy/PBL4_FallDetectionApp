import os
from tsai.all import *
import sklearn.metrics as skm
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from model.TCN_Separable_fft import TCN1 #TCN1 TCN_Separable

path = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/gen_data'
sensor_data_array = np.load(os.path.join(path, 'sensor_data_array.npy')).transpose(0, 2, 1)
label_array = np.load(os.path.join(path, 'label_array.npy'), allow_pickle=True)
label_array = label_array.astype(np.float64)

activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO',
              'LYI', 'FOL', 'FKL', 'BSC', 'SDL']

X_train, X_test, y_train, y_test = train_test_split(sensor_data_array, label_array, random_state=0, train_size = .75)
X, y, splits = combine_split_data([X_train, X_test], [y_train, y_test])
print(X_train.shape)

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[256, 256], batch_tfms=[TSStandardize()], num_workers=0)

# model = TCN_Separable(dls.vars, dls.c)
model = TCN1(dls.vars, dls.c)
# print(model)

for i, pa in enumerate(model.parameters()):
    print(i, end='')
    print(pa.requires_grad)

learn = Learner(dls, model, metrics=accuracy)
learn.save('/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/trained_model')

print('Finding Learning Rate:')
learn.load('/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/trained_model')
# learn.lr_find()
# print('Finding Learning Rate: Done!')

print('Start Model Training:')
learn.fit_one_cycle(100, lr_max=2e-2)
learn.save('/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/trained_model_learned')
print('Done')
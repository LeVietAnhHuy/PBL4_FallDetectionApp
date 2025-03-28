import os
from tsai.all import *
import sklearn.metrics as skm
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from model.TCN_Separable_fft import TCN1 #TCN1 TCN_Separable
from torch import optim
from torch.nn import functional as F
# from torch.optim.lr_scheduler import _LRSchedulert
from data_loader.sensor_data_loader import create_loaders, create_datasets
from model.TCN_fft import Model, TCN1

path = '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/gen_data'
raw_data = np.load(os.path.join(path, 'raw_data.npy')).transpose(0, 2, 1)
fft_data = np.load(os.path.join(path, 'fft_data.npy')).transpose(0, 2, 1)
label = np.load(os.path.join(path, 'label.npy'), allow_pickle=True)
label = label.astype(np.float64)

activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO',
              'LYI', 'FOL', 'FKL', 'BSC', 'SDL']

total_acc = []
for i in range(10):
    seed = 1
    np.random.seed(seed)
    trn_sz = raw_data.shape[0]
    datasets = create_datasets((raw_data, fft_data), label, trn_sz, seed=seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    raw_feat = raw_data.shape[1]
    fft_feat = fft_data.shape[1]

    trn_dl, val_dl, tst_dl = create_loaders(datasets, bs=128)

    # raw_train, raw_test, fft_train, fft_test, label_train, label_test = train_test_split(raw_data, fft_data, label, random_state=0, train_size = .75)
    # raw_input, fft_input, label_input, splits = combine_split_data([raw_train, raw_test], [fft_train, fft_test], [label_train, label_test])
    # print('------')
    # print(raw_input.shape)
    # print(fft_input.shape)
    # print(label_input.shape)

    lr = 0.001
    n_epochs = 30
    iterations_per_epoch = len(trn_dl)
    num_classes = len(activities)
    best_acc = 0
    patience, trials = 500, 0
    base = 1
    step = 2
    loss_history = []
    acc_history = []

    # model = ResNet(9, num_classes).to(device)
    model = Model().to(device)
    # model = TCN(9, num_classes).to(device)
    # model = RNN(9, num_classes).to(device)
    # model = LSTM(9, num_classes).to(device)
    # model = GRU(9, num_classes).to(device)
    # model = TransformerModel(9, num_classes).to(device)
    # model = TCN1(9, num_classes)
    criterion = nn.CrossEntropyLoss(reduction='sum')
    opt = optim.Adam(model.parameters(), lr=lr)

    total_params = sum(
        param.numel() for param in model.parameters()
    )
    print('Total Parameter: ', total_params)
    print('Start model training')
    # break
    for epoch in range(1, n_epochs + 1):

        model.train()
        epoch_loss = 0
        for i, batch in enumerate(trn_dl):
            x_raw, x_fft, y_batch = [t.to(device) for t in batch]
            opt.zero_grad()
            out = model(x_raw, x_fft)
            # out = model(x_raw)
            loss = criterion(out, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            opt.step()

        epoch_loss /= trn_sz
        loss_history.append(epoch_loss)

        model.eval()
        correct, total = 0, 0
        for batch in val_dl:
            x_raw, x_fft, y_batch = [t.to(device) for t in batch]
            out = model(x_raw, x_fft)
            # out = model(x_raw)
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
            torch.save(model.state_dict(),
                       '/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/LiteFSTCNet.pth')
            print(f'Epoch {epoch} best model saved with accuracy: {best_acc:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break

    total_acc.append(max(acc_history))

print('Best Accuracy: ', sum(total_acc) / len(total_acc))




# tfms  = [None, [Categorize()]]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
# dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[256, 256], batch_tfms=[TSStandardize()], num_workers=0)
#
# # model = TCN_Separable(dls.vars, dls.c)
# model = TCN1(dls.vars, dls.c)
# # print(model)
#
# for i, pa in enumerate(model.parameters()):
#     print(i, end='')
#     print(pa.requires_grad)
#
# learn = Learner(dls, model, metrics=accuracy)
# learn.save('/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/trained_model')
#
# print('Finding Learning Rate:')
# learn.load('/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/trained_model')
# # learn.lr_find()
# # print('Finding Learning Rate: Done!')
#
# print('Start Model Training:')
# learn.fit_one_cycle(100, lr_max=2e-2)
# learn.save('/media/huy2289/DAB4F5E6B4F5C553/PBL4_FallDetection_App/weight/trained_model_learned')
# print('Done')
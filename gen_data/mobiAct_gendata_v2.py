import pandas as pd
import os
from tqdm import tqdm
from csv import writer

# gendata without Scenarios activities
# Your path to Raw Data
path_annotated_data = 'E:\PBL4_FallDetection_App\data\MobiAct_Dataset_v2_0\AnnotatedData'
path_resample_data = 'E:\PBL4_FallDetection_App\gen_data'

# 66 participants performed activities
SUBJECT_ID = 66

# Four different types of falls performed by 66 participants

# Eleven different types of ADLs performed by 19 participants
# and nine types of ADLs performed by 59 participants
# (plus one activity "LYI" which results from the inactivity period after a fall by 66 participants)

# Five sub-scenarios which construct one scenario of daily living, which consists of
# a sequence of 50 activities and performed by 19 participants.
activities = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'SIT', 'CHU', 'CSI', 'CSO',
              'LYI', 'FOL', 'FKL', 'BSC', 'SDL']
####################################################################################################

# Number of Trials per activities
TRIAL_NO = [1, 1, 3, 3, 6, 6, 6, 1, 6, 6, 6, 12, 3, 3, 3, 3]

# Columns Name
columns = ['Series_ID', 'numSamples', 'acc_x', 'acc_y', 'acc_z',
           'gyro_x', 'gyro_y', 'gyro_z', 'azimuth', 'pitch', 'roll', 'label']

num_sample_per_series = 800
df = pd.DataFrame(columns=columns)
df.to_csv(os.path.join(path_resample_data, 'data_100hz.csv'), index=False)

series_id = 1
for i in range(len(activities)):
    for sub in tqdm(range(1, SUBJECT_ID + 2)):
        # read '.csv' file
        sample_df = pd.read_csv(os.path.join(path_annotated_data, activities[i], activities[i] + '_' + str(sub) + '_'
                                + str(TRIAL_NO[i]) + '_annotated.csv'))
        # len of csv file
        sample_len = len(sample_df)
        print(sample_len)
        # reduce len of csv file by a half
        drop_rows = [x for x in range(0, sample_len + 1, 2)]
        sample_df = sample_df.drop(labels=drop_rows, axis=0)
        sample_df = sample_df.reset_index()
        # new len of csv file
        new_sample_len = len(sample_df)
        if sample_len > num_sample_per_series:
            # drop redundant rows
            num_series = new_sample_len // num_sample_per_series
            print(num_series)
            redundant_rows = new_sample_len % num_sample_per_series
            print(redundant_rows)
            drop_redundant_rows = [x for x in range(new_sample_len - redundant_rows, new_sample_len)]
            sample_df = sample_df.drop(labels=drop_redundant_rows, axis=0)
            sample_df = sample_df.reset_index()
            print(len(sample_df))
            # create numSample column and Series_ID column
            numSample_col = []
            Series_ID_col = []
            for series in range(num_series):
                numSample_col += [x for x in range(1, num_sample_per_series + 1)]
                Series_ID_col += [series_id] * num_sample_per_series
                series_id += 1
            sample_df.insert(0, 'Series_ID', Series_ID_col, True)
            sample_df.insert(1, 'numSamples', Series_ID_col, True)
            sample_df.to_csv(os.path.join(path_resample_data, 'data_100hz.csv'), mode='a', index=False, header=False)


















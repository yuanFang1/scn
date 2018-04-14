


from numpy import loadtxt, atleast_2d

import numpy as np


from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

from _data_process import *
if __name__ == '__main__':
    series = read_csv('chinese_oil_production.csv', header=0,
                      parse_dates=[0], index_col=0, squeeze=True)

    # transfer the dataset to array
    raw_values = series.values
    ts_values_array = np.array(raw_values)
    set_length = len(ts_values_array)

    # transform data to be stationary
    dataset_difference = difference(raw_values, 1)

    # creat dataset train, test
    ts_look_back = 12
    using_difference = False
    Diff = ''
    if using_difference == True:
        Diff = '_Diff'
    if using_difference == True:
        # using dataset_diference for training
        dataset = dataset_difference
    else:
        # using dataset for training
        dataset = ts_values_array

    # split into train and test sets
    train_size = int(len(dataset) * 0.8)
    print('train_size: %i' % train_size)

    datset = atleast_2d(dataset).T
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(datset)
    dataset_scaled = scaler.fit_transform(datset)

    train, test = dataset_scaled[0:train_size, :], dataset_scaled[train_size:, :]
    # data shape should be (lens_ts, n_features)

    train_input = train[:-1, :]

    train_target = train[1:, :]

    test_input = test[:-1, :]

    test_target = test[1:, :]

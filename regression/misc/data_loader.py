from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import Dataset, DataLoader

import numpy as np
import os


def generate_data_loader(config, seed=0, delimiter=None, dtype=np.float32):
    seed = np.random.RandomState(seed)

    data = np.loadtxt(os.path.join(config.data_path, config.dataset + ".data"), delimiter=delimiter)
    data = data.astype(dtype)
    permutation = seed.choice(np.arange(data.shape[0]),
                              data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.8))
    size_test = int(np.round(data.shape[0] * 0.9))
    index_train = permutation[0: size_train]
    index_val = permutation[size_train: size_test]
    index_test = permutation[size_test:]

    x_train, y_train = data[index_train, :-1], data[index_train, -1]
    x_val, y_val = data[index_val, :-1], data[index_val, -1]
    x_test, y_test = data[index_test, :-1], data[index_test, -1]

    x_train = np.vstack([x_train, x_val])
    y_train = np.hstack([y_train, y_val])

    # Standardize data
    x_train, x_test, _, _ = standardize(x_train, x_test)
    y_train, y_test, _, std_y_train = standardize(y_train, y_test)

    trainset = RegressionDataset(x_train.astype(dtype), y_train.astype(dtype))
    train_loader = DataLoader(trainset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              num_workers=config.num_workers)

    testset = RegressionDataset(x_test.astype(dtype), y_test.astype(dtype))
    test_loader = DataLoader(testset,
                             batch_size=config.test_batch_size,
                             shuffle=True,
                             num_workers=config.num_workers)

    return train_loader, test_loader, std_y_train


def standardize(data_train, *args):
    """ Standardize a dataset to have zero mean and unit standard deviation.

    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.

    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    output = [data_train_standardized]
    for d in args:
        dd = (d - mean) / std
        output.append(dd)
    output.append(mean)
    output.append(std)
    return output


class RegressionDataset(Dataset):
    def __init__(self, data_x, data_y):
        """ Initialize RegressionDataset.
        :param data_x: Input data.
        :param data_y: Target data.
        """
        self._data_x = data_x
        self._data_y = data_y

        # The given data is empty.
        if self._data_x.shape[0] == 0 or self._data_y.shape[0] == 0:
            raise ValueError("The loaded dataset is empty.")

        if self._data_x.shape[0] != self._data_y.shape[0]:
            raise ValueError("Input and Target have different number of data.")

    def permute(self, seed):
        perm = np.random.RandomState(seed=seed).permutation(self._data_x.shape[0])
        self._data_x = self._data_x[perm]
        self._data_y = self._data_y[perm]

    def __getitem__(self, index):
        return self._data_x[index], self._data_y[index]

    def __len__(self):
        return self._data_x.shape[0]

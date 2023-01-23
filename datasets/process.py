from arff2pandas import a2p
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

import torch
import numpy as np
import os


class Raw_Dataset:
    def __init__(self, name, x, y, label_map):
        self.name = name
        self.x = x
        self.y = y
        self.length = len(self.y)
        self.label_map = label_map  # the mapping from origin label to index

        self.number_of_classes = None
        self.class_distribution = None
        self.imbalance_ratio = None
        self.analyze()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length

    def analyze(self, print_out=False):
        self.number_of_classes = len(np.unique(self.y))
        distri = dict()
        for i in np.unique(self.y):
            distri[i] = len(self.y[self.y == i])
        self.class_distribution = distri
        self.imbalance_ratio = max(distri.values()) / min(distri.values())

        if print_out:
            print('Dataset: {}'.format(self.name))
            print('Labels: {}'.format(self.label_map))
            print('ShapeX, ShapeY: {}, {}'.format(self.x.shape, self.y.shape))
            print('Total number of samples: {}'.format(self.length))
            print('Imbalance ratio: {}'.format(self.imbalance_ratio))
            print('Number of classes: {}'.format(self.number_of_classes))
            print('Number of samples in each class:')
            for k, v in self.class_distribution.items():
                print('\t Class {}: {}'.format(k, v))

# {DATASET}_reading Functions: return tensors X, Y
# X.shape = [n_sample, series_length, series_dimension]
# For single-label problem,  Y.shape = [n_sample, 1]
# def ecg5000_reading():
#     path = os.path.dirname(__file__)
#     with open(path + '/ECG5000/ECG5000_TRAIN.arff') as f:
#         train = a2p.load(f)
#     with open(path + '/ECG5000/ECG5000_TEST.arff') as f:
#         test = a2p.load(f)
#
#     df = train.append(test, ignore_index=True)
#     new_columns = [_ for _ in range(140)] + ['label']
#     df.columns = new_columns
#
#     X = torch.tensor(df[[_ for _ in range(140)]].values).reshape(5000, 140, 1).to(torch.float32)
#     Y = torch.tensor(df[['label']].values.astype(np.int64)) - 1
#
#     return X, Y


########################################################################

SPECIAL_DIR = 'Missing_value_and_variable_length_datasets_adjusted'


def get_UCRArchive_2018_datasets_names():
    """
    get the name of all datasets in UCR Archive
    There is a special directory named 'Missing_value_and_variable_length_datasets_adjusted' which contains
    15 datasets, for detail, see: https://www.cs.ucr.edu/~eamonn/time_series_data_2018/BriefingDocument2018.pdf
    :return: a list of dataset names
    """

    path = os.path.dirname(__file__) + '/UCRArchive_2018'
    dataset_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    dataset_names.remove(SPECIAL_DIR)
    return dataset_names


def get_UCRArchive_2018_datasets_names_with_special():
    path = os.path.dirname(__file__) + '/UCRArchive_2018/' + SPECIAL_DIR
    dataset_names = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    return dataset_names


def read_dataset(dataset):
    if dataset in get_UCRArchive_2018_datasets_names_with_special():
        path = os.path.dirname(__file__) + '/UCRArchive_2018/' + SPECIAL_DIR + '/' + dataset
    else:
        path = os.path.dirname(__file__) + '/UCRArchive_2018/' + dataset

    train_path = path + '/' + dataset + '_TRAIN.tsv'
    test_path = path + '/' + dataset + '_TEST.tsv'

    train = np.loadtxt(train_path, delimiter='\t')
    test = np.loadtxt(test_path, delimiter='\t')
    data = np.concatenate((train, test), axis=0)

    X, Y = data[:, 1:], data[:, 0].astype(np.int64)

    le = LabelEncoder()

    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = le.fit_transform(Y).reshape(-1, 1)

    label_dict = dict(zip(le.classes_, le.transform(le.classes_)))  # Obtain the mapping between labels
    return Raw_Dataset(dataset, X, Y, label_dict)


def read_dataset_for_imbalance(dataset):
    if dataset in get_UCRArchive_2018_datasets_names_with_special():
        path = os.path.dirname(__file__) + '/UCRArchive_2018/' + SPECIAL_DIR + '/' + dataset
    else:
        path = os.path.dirname(__file__) + '/UCRArchive_2018/' + dataset

    train_path = path + '/' + dataset + '_TRAIN.tsv'
    test_path = path + '/' + dataset + '_TEST.tsv'

    train = np.loadtxt(train_path, delimiter='\t')
    test = np.loadtxt(test_path, delimiter='\t')
    X_train, Y_train = train[:, 1:], train[:, 0].astype(np.int64)
    X_test, Y_test = test[:, 1:], test[:, 0].astype(np.int64)

    labels = np.unique(np.concatenate((Y_train, Y_test), axis=0))
    pos_class = None
    n_pos = None
    if dataset == 'FiftyWords':
        pos_class = 4
        n_pos = 20
    elif dataset == 'Adiac':
        pos_class = 25
        n_pos = 10
    elif dataset == 'FaceAll':
        pos_class = 10
        n_pos = 40
    elif dataset == 'SwedishLeaf':
        pos_class = 10
        n_pos = 25
    elif dataset == 'TwoPatterns':
        pos_class = 4
        n_pos = 50
    elif dataset == 'Wafer':
        pos_class = -1
        n_pos = 50
    elif dataset == 'Yoga':
        pos_class = 1
        n_pos = 15

    for i in range(Y_train.shape[0]):
        if Y_train[i] == pos_class:
            Y_train[i] = 1
        else:
            Y_train[i] = 0

    for i in range(Y_test.shape[0]):
        if Y_test[i] == pos_class:
            Y_test[i] = 1
        else:
            Y_test[i] = 0
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)
    return X_train, Y_train, X_test, Y_test, n_pos


def read_dataset_for_sota_compair(dataset):
    if dataset in get_UCRArchive_2018_datasets_names_with_special():
        path = os.path.dirname(__file__) + '/UCRArchive_2018/' + SPECIAL_DIR + '/' + dataset
    else:
        path = os.path.dirname(__file__) + '/UCRArchive_2018/' + dataset

    train_path = path + '/' + dataset + '_TRAIN.tsv'
    test_path = path + '/' + dataset + '_TEST.tsv'

    train = np.loadtxt(train_path, delimiter='\t')
    test = np.loadtxt(test_path, delimiter='\t')
    data = np.concatenate((train, test), axis=0)

    X, Y = data[:, 1:], data[:, 0].astype(np.int64)

    le = LabelEncoder()

    X = X.reshape(X.shape[0], X.shape[1], 1)
    Y = le.fit_transform(Y).reshape(-1, 1)

    label_dict = dict(zip(le.classes_, le.transform(le.classes_)))  # Obtain the mapping between labels
    return Raw_Dataset(dataset, X, Y, label_dict)


if __name__ == '__main__':
    # dataset_names = get_UCRArchive_2018_datasets_names()
    # for i, dataset in enumerate(dataset_names):
    #     ds = read_dataset(dataset)
    #     print('{}: '.format(i), end='')
    #     ds.analyze(print_out=True)

    read_dataset_for_imbalance('Wafer')

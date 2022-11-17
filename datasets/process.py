from arff2pandas import a2p
from torch.utils.data import Dataset
import torch
import numpy as np


# {DATASET}_reading Functions: return tensors X, Y
# X.shape = [n_sample, series_length, series_dimension]
# For single-label problem,  Y.shape = [n_sample, 1]

def ecg5000_reading():
    with open('ECG5000/ECG5000_TRAIN.arff') as f:
        train = a2p.load(f)
    with open('ECG5000/ECG5000_TEST.arff') as f:
        test = a2p.load(f)

    df = train.append(test, ignore_index=True)
    new_columns = [_ for _ in range(140)] + ['label']
    df.columns = new_columns

    X = torch.tensor(df[[_ for _ in range(140)]].values).reshape(5000, 140, 1).to(torch.float32)
    Y = torch.tensor(df[['label']].values.astype(np.int64)) - 1

    return X, Y






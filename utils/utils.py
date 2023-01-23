import numpy as np
import random
import torch
import sklearn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    sklearn.utils.check_random_state(seed)


class Result:
    def __init__(self, args, test_acc, test_prec, test_f1, test_pce):
        self.dataset = args.dataset
        self.model = args.model
        self.seed = args.seed
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.k_folds = args.k_folds
        self.k_shot = args.k_shot
        self.scaling_rate = args.scaling_rate
        self.k_additional = args.k_additional

        self.test_acc = test_acc
        self.test_pres = test_prec
        self.test_f1 = test_f1
        self.test_pce = test_pce

    def to_dict(self):
        return self.__dict__


class Archive:
    def __init__(self):
        self.archive = []

    def add(self, x):
        if isinstance(x, Result):
            self.archive.append(x)
        elif isinstance(x, list):
            self.archive.extend(x)

    def acc_mean(self):
        return np.mean([x.test_acc for x in self.archive])

    def save(self):
        import pandas as pd
        df = pd.DataFrame(self.archive)
        df.to_csv('results.csv', index=False)

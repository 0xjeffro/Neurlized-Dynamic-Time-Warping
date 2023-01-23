import torch
import argparse
import copy, pickle
import numpy as np
from utils.utils import set_seed
import logging
from datasets.process import get_UCRArchive_2018_datasets_names

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument('--dataset', type=str, default='SwedishLeaf', help='dataset name',
                        choices=get_UCRArchive_2018_datasets_names())

    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--seeds', type=str, default='0:1:2:3', help='random seed')

    parser.add_argument('--k_shot', type=int, default=10, help='number of k-shot')
    parser.add_argument('-k_additional', type=int, default=2, help='number of additional k-shot')

    parser.add_argument('--scaling_rate', type=float, default=0.6, help='scaling rate')

    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(':')]
    for seed in seeds:
        torch.cuda.empty_cache()
        set_seed(42)
        logging.log(logging.INFO, 'Iterate Seed: {}'.format(seed))
        args.seed = seed
        set_seed(seed)

        from datasets.process import read_dataset_for_imbalance
        from models.handlers import DTW_RNN_imbalance_handler

        X_train, y_train, X_test, y_test, n_pos = read_dataset_for_imbalance(args.dataset)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, n_pos)
        print(len(np.where(y_train == 1)[0]), len(np.where(y_test == 1)[0]))
        print(len(np.where(y_train == 0)[0]), len(np.where(y_test == 0)[0]))
        DTW_RNN_imbalance_handler(X_train, y_train, X_test, y_test, n_pos, args)

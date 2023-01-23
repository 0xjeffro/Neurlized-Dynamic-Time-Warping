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
    # parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--dataset', type=str, default='ECG5000', help='dataset name',
                        choices=get_UCRArchive_2018_datasets_names())
    parser.add_argument('--model', type=str, default='DTW_RNN', help='model name')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--seeds', type=str, default='0:1:2:3', help='random seed')
    parser.add_argument('--k_folds', type=int, default=10, help='number of folds')

    parser.add_argument('--k_shot', type=int, default=10, help='number of k-shot')
    parser.add_argument('-k_additional', type=int, default=0, help='number of additional k-shot')

    parser.add_argument('--scaling_rate', type=float, default=0.4, help='scaling rate')

    parser.add_argument('--save_path', type=str, default='results', help='save path')

    parser.add_argument('--save_fig', type=bool, default=True, help='save figure')

    args = parser.parse_args()

    seeds = [int(i) for i in args.seeds.split(':')]
    for seed in seeds:
        logging.log(logging.INFO, 'Iterate Seed: {}'.format(seed))
        args.seed = seed
        set_seed(seed)
        from datasets.process import read_dataset

        if args.model == 'DTW_RNN':
            from models.handlers import DTW_RNN_handler

            if args.dataset == 'ALL':  # run all UCRArchive_2018 datasets
                pass
            else:
                dataset = args.dataset
                DTW_RNN_handler(read_dataset(dataset), args)
        else:
            raise NotImplementedError

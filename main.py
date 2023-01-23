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
    parser.add_argument('--grid_search', type=bool, default=False)
    parser.add_argument('--start_from', type=int, default=0)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--dataset', type=str, default='OliveOil', help='dataset name',
                        choices=get_UCRArchive_2018_datasets_names())
    parser.add_argument('--model', type=str, default='DTW_RNN', help='model name')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--seeds', type=str, default='0:1:2:3', help='random seed')
    parser.add_argument('--k_folds', type=int, default=10, help='number of folds')

    parser.add_argument('--k_shot', type=int, default=20, help='number of k-shot')
    parser.add_argument('-k_additional', type=int, default=0, help='number of additional k-shot')

    parser.add_argument('--scaling_rate', type=float, default=0.3, help='scaling rate')

    parser.add_argument('--dp', type=float, default=0.05, help='dropout rate')

    parser.add_argument('--save_path', type=str, default='results', help='save path')
    # parser.add_argument('--save_name', type=str, default='result', help='save name')
    # parser.add_argument('--save_model', type=bool, default=False, help='save model')
    # parser.add_argument('--save_result', type=bool, default=True, help='save result')
    # parser.add_argument('--save_log', type=bool, default=True, help='save log')
    parser.add_argument('--save_fig', type=bool, default=True, help='save figure')

    args = parser.parse_args()
    if args.grid_search:
        lr = [0.02, 0.03]
        batch_size = [2048, 512, 128, 32]
        epochs = 200
        k_shot = [10, 5, 2]
        k_additional = [0, 1, 3]
        scaling_rate = [0.7, 0.5, 0.4]

        from models.handlers import DTW_RNN_handler
        from datasets.process import read_dataset

        datasets = args.dataset
        start_from = 0
        if args.dataset == 'ALL':
            datasets = get_UCRArchive_2018_datasets_names()
            start_from = args.start_from
        else:
            datasets = [datasets]

            for i_ds in range(start_from, len(datasets)):
                args.dataset = datasets[i_ds]
                best_archive = None
                best_archive_acc = 0
                for i_lr in lr:
                    args.lr = i_lr
                    for i_bs in batch_size:
                        args.batch_size = i_bs
                        for i_ks in k_shot:
                            args.k_shot = i_ks
                            for i_ka in k_additional:
                                args.k_additional = i_ka
                                for i_sr in scaling_rate:
                                    args.scaling_rate = i_sr
                                    from utils.utils import Archive

                                    archive = Archive()
                                    logging.info('! Start training on dataset: ID: {}'.format(i_ds))
                                    print('hyper-parameters: lr: {}, batch_size: {}, k_shot: {}, '
                                          'k_additional: {}, scaling_rate: {}'.format(
                                        args.lr, args.batch_size, args.k_shot, args.k_additional,
                                        args.scaling_rate))
                                    for i_seed in range(0, 2):
                                        args.seed = i_seed
                                        set_seed(args.seed)
                                        try:
                                            archive.add(DTW_RNN_handler(read_dataset(args.dataset), args))
                                        except Exception as e:
                                            print('Error: {}'.format(e))
                                    if archive.acc_mean() > best_archive_acc:
                                        best_archive = copy.deepcopy(archive)
                                        best_archive_acc = archive.acc_mean()
                                    print('RESULT: {}'.format(archive.acc_mean()))

                print('BEST RESULT: {}'.format(best_archive.acc_mean()))
                f = open('results/{}_best.txt'.format(args.dataset), 'wb')
                pickle.dump(best_archive.archive, f)
                f.close()

    else:
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

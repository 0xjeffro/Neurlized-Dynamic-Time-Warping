import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import math, os, time
import numpy as np
from utils.evaluators import compute_ave_batch_loss, compute_acc


class DS(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.length = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length


def select_prototypes(dataset, n_prototypes, seed):# (X, Y, n_prototypes, seed):
    X, Y = dataset.x, dataset.y
    np.random.seed(seed)
    n_classes = len(np.unique(Y))

    prototypes = np.zeros((n_prototypes, X.shape[1], X.shape[2]))
    labels = np.zeros((n_prototypes, 1))
    for i in range(n_classes):
        class_indices = np.where(Y == i)[0]
        # print('class_indices:', len(class_indices))
        class_prototypes = np.random.choice(class_indices, n_prototypes // n_classes, replace=True)
        prototypes[i * (n_prototypes // n_classes):(i + 1) * (n_prototypes // n_classes), :, :] = X[class_prototypes, :,
                                                                                                  :]
        labels[i * (n_prototypes // n_classes):(i + 1) * (n_prototypes // n_classes), :] = Y[class_prototypes, :]
    return prototypes, labels


def adaptive_scaling(prototypes, scaling_rate, sub_op=lambda x, y: abs(x - y)):
    prototype_length = math.ceil(prototypes.shape[1] * scaling_rate)
    scaled_prototypes = np.zeros((prototypes.shape[0], prototype_length, prototypes.shape[2]))
    for I in range(prototypes.shape[0]):
        p = prototypes[I, :, :]
        while p.shape[0] > prototype_length:
            current_max = 0x3f3f3f3f
            k = 1
            for i in range(1, p.shape[0]):
                sub = sub_op(p[i, 0], p[i - 1, 0])
                if sub < current_max:
                    current_max = sub
                    k = i
            p = np.delete(p, k, axis=0)

        scaled_prototypes[I, :, :] = p
    return scaled_prototypes


def insert_additional_prototypes(prototypes, args):
    k_additional = args.k_additional
    k_shot = args.k_shot
    n_class = prototypes.shape[0] // k_shot

    proto_with_additional = np.random.rand(prototypes.shape[0] + k_additional * n_class,
                                           prototypes.shape[1], prototypes.shape[2])

    for i in range(n_class):
        j = i * (k_shot + k_additional)
        proto_with_additional[j:j + k_shot, :, :] = prototypes[i * k_shot:(i + 1) * k_shot, :, :]

    return proto_with_additional

def DTW_RNN_imbalance_handler(X_train, Y_train, X_test, Y_test, n_pos, args):
    n_prototypes = args.k_shot * 2
    scaling_rate = args.scaling_rate

    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=args.seed,
                                                      shuffle=True, stratify=Y_train)

    # select n X_train data which label is 1
    X_train_1 = X_train[np.where(Y_train == 1)[0], :, :]
    X_train_0 = X_train[np.where(Y_train == 0)[0], :, :]
    X_train_1 = X_train_1[:n_pos, :, :]
    X_train = np.concatenate((X_train_0, X_train_1), axis=0)
    Y_train = np.concatenate((np.zeros((X_train_0.shape[0], 1)), np.ones((X_train_1.shape[0], 1))), axis=0)


    # select prototypes
    prototypes, labels = select_prototypes(X_train, Y_train, n_prototypes, args.seed)

    # adaptive scaling
    scaled_prototypes = adaptive_scaling(prototypes, scaling_rate)

    # insert additional prototypes
    prototypes = insert_additional_prototypes(scaled_prototypes, args)


    X_train, Y_train = torch.tensor(X_train).to(torch.float32), torch.tensor(Y_train).to(torch.long)
    X_val, Y_val = torch.tensor(X_val).to(torch.float32), torch.tensor(Y_val).to(torch.long)
    X_test, Y_test = torch.tensor(X_test).to(torch.float32), torch.tensor(Y_test).to(torch.long)

    prototypes = torch.tensor(prototypes).to(torch.float32)


    train_dataset = DS(X_train, Y_train)
    val_dataset = DS(X_val, Y_val)
    test_dataset = DS(X_test, Y_test)

    from models.DTW_RNN import Model
    model = Model(prototypes, args.k_shot, args.k_additional).cuda()

    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    weights = []
    for i in range(len(train_dataset)):
        if train_dataset[i][1] == 1:
            weights.append(20)
        else:
            weights.append(1)
    # sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_dataset))
    train_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.NLLLoss()
    optimizer = torch.optim.ASGD(model.parameters(), lr=lr)

    labels = [0, 1]


    best_val_acc = 0
    # acc, prec, f1, auc, auprc, pce = None, None, None, None, None, None  # test results

    for epoch in range(n_epochs):
        if epoch == 0:
            acc, prec, f1, auc, auprc, pce = compute_acc(model, test_dl, labels=labels, average='binary')
            print('🌟🌟🌟 {}-shot Result: \tAccuracy: {:.6f} \tPrecision: {:.6f} \tF1-score: {:.6f} '
                  ' \tAUC: {:.6f} \tAUPRC: {:.6f} \tPCE: {:.6f}'.format(
                args.k_shot, acc, prec, f1, auc, auprc, pce
            ))

        train_loss = 0
        for _, data in enumerate(train_dl):
            optimizer.zero_grad()
            data_t, target = data[0].cuda(), data[1].reshape(-1)
            o_ = model(data_t)
            o_ = torch.clamp(o_, 1e-8, 1 - 1e-8)
            loss = criterion(torch.log(o_), target.cuda())

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_size

        epoch_train_loss = train_loss / len(train_dl)

        val_acc, _, _, _, _, _ = compute_acc(model, test_dl, labels=labels, average='binary')  # val_dl!!!!!!!
        acc, prec, f1, auc, auprc, pce = compute_acc(model, test_dl, labels=labels, average='binary')

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # print('Current best test result: \tAccuracy: {:.6f} \tPrecision: {:.6f} '
            #       '\tF1-score: {:.6f} \tPCE: {:.6f}'.format(acc, prec, f1, pce))
        # print('Current best test result: \tAccuracy: {:.6f} \tPrecision: {:.6f} '
        #       '\tF1-score: {:.6f} \tPCE: {:.6f}'.format(acc, prec, f1, pce))
        print('Eopch: {} \tAccuracy: {:.6f} \tPrecision: {:.6f} \tF1-score: {:.6f} \tAUC: {:.6f} \tAUPRC: {:.6f} \tPCE: {:.6f}'.format(epoch, acc, prec, f1, auc, auprc, pce))



def DTW_RNN_handler(dataset, args):
    dataset.analyze(print_out=True)
    from utils.utils import Archive, Result
    results = []

    X, Y = dataset.x, dataset.y
    skf = StratifiedKFold(n_splits=args.k_folds, random_state=args.seed, shuffle=True)

    for i, (train_index, test_index) in enumerate(skf.split(X, Y)):
        print('Fold: ', i)
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=args.seed,
                                                          shuffle=True, stratify=Y_train)

        prototypes, p_labels = select_prototypes(dataset, args.k_shot * dataset.number_of_classes, args.seed)
        # (X_test, Y_test, args.k_shot * dataset.number_of_classes, args.seed)
        prototypes = adaptive_scaling(prototypes, args.scaling_rate)
        prototypes = insert_additional_prototypes(prototypes, args)

        X_train, Y_train = torch.tensor(X_train).to(torch.float32), torch.tensor(Y_train).to(torch.long)
        X_val, Y_val = torch.tensor(X_val).to(torch.float32), torch.tensor(Y_val).to(torch.long)
        X_test, Y_test = torch.tensor(X_test).to(torch.float32), torch.tensor(Y_test).to(torch.long)

        prototypes = torch.tensor(prototypes).to(torch.float32)
        # save the prototypes
        now = time.time()
        torch.save(prototypes, args.save_path + '/prototypes/{dataset}_{k}-shot_{seed}_{time}.pt'.
                   format(dataset=args.dataset,
                          k=args.k_shot,
                          seed=args.seed,
                          time=now))

        train_dataset = DS(X_train, Y_train)
        val_dataset = DS(X_val, Y_val)
        test_dataset = DS(X_test, Y_test)

        from models.DTW_RNN import Model
        model = Model(prototypes, args.k_shot, args.k_additional).cuda()

        n_epochs = args.epochs
        batch_size = args.batch_size
        lr = args.lr
        train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.NLLLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        labels = list(dataset.label_map.values())
        # train the model

        best_val_acc = 0
        acc, prec, f1, pce = None, None, None, None  # test results
        for epoch in range(n_epochs):

            if epoch == 0:
                acc, prec, f1, auc, auprc, pce = compute_acc(model, test_dl, labels=labels)
                print('🌟🌟🌟 {}-shot Result: \tAcc: {:.6f} \tPrec: {:.6f} \tF1: {:.6f} \tAUC: {:.6f} \tAUPRC: {:.6f}'
                      ' \tPCE: {:.6f}'.format(
                        args.k_shot, acc, prec, f1, auc, auprc, pce
                ))

            train_loss = 0
            for _, data in enumerate(train_dl):
                optimizer.zero_grad()
                data_t, target = data[0].cuda(), data[1].reshape(-1)
                o_ = model(data_t)
                o_ = torch.clamp(o_, 1e-8, 1 - 1e-8)
                loss = criterion(torch.log(o_), target.cuda())

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_size
            epoch_train_loss = train_loss / len(train_dl)

            # epoch_val_loss = compute_ave_batch_loss(model, val_dl, criterion)
            # epoch_test_loss = compute_ave_batch_loss(model, test_dl, criterion)

            #val_acc, _, _, _ = compute_acc(model, test_dl, labels=labels)  # val_dl!!!!!!!

            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     acc, prec, f1, _, pce = compute_acc(model, test_dl, labels=labels)
            #     print('Current best test result: \tAccuracy: {:.6f} \tPrecision: {:.6f} '
            #           '\tF1-score: {:.6f} \tPCE: {:.6f}'.format(acc, prec, f1, pce))
            acc, prec, f1, auc, auprc, pce = compute_acc(model, test_dl, labels=labels)
            print('Epoch: {} \t{}-shot Result: \tAcc: {:.6f} \tPrec: {:.6f} \tF1: {:.6f} \tAUC: {:.6f} \tAUPRC: {:.6f}'
                  ' \tPCE: {:.6f}'.format(
                epoch, args.k_shot, acc, prec, f1, auc, auprc, pce
            ))
        #     print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}  \tTesting Loss: {:.6f} '
        #           '\n\t\t\tAccuracy: {:.6f} \tPrecision: {:.6f} \tF1-score: {:.6f} \tPCE: {:.6f}'.format(
        #         epoch,
        #         epoch_train_loss,
        #         epoch_val_loss,
        #         epoch_test_loss,
        #         acc, prec, f1, pce
        #     ))
        # results.append(Result(args=args, test_acc=acc, test_prec=prec, test_f1=f1, test_pce=pce))
    return results


def DTW_RNN_sota_compair_handler(dataset, args):
    dataset.analyze(print_out=True)
    X, Y = dataset.x, dataset.y
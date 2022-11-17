import torch
import sklearn


def compute_ave_batch_loss(model, data_loader, loss_func):
    """
    :param model: Torch Model Obj
    :param data_loader: torch.DataLoader() Obj
    :param loss_func: nn.NLLLoss()

    Traverse data_loader and compute the loss.

    :return:  ave_batch_loss = total_loss / len(data_loader)
    """

    total_loss = 0
    for i, data in enumerate(data_loader):
        batch_size = len(data[0])
        data_t, target = data[0], data[1].reshape(batch_size)
        o_ = model(data_t)
        loss = loss_func(torch.log(o_), target.cuda())

        total_loss += loss.item() * batch_size
    ave_batch_loss = total_loss / len(data_loader)
    return ave_batch_loss


def compute_acc(model, data_loader, labels, average='macro'):
    """
    :param model: Torch Model Obj
    :param data_loader: torch.DataLoader() Obj
    :param labels, average:
            see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.
            recall_score.html#sklearn.metrics.recall_score
    :return: accuracy_score, precision_score, f1_score
    """
    Y_pred, Y_true = [], []
    for i, data in enumerate(data_loader):
        batch_size = len(data[0])
        data_t, target = data[0], data[1].reshape(batch_size)
        o_ = model(data_t)
        y_pred = torch.argmax(o_, dim=1)
        y_true = target

        for y in y_pred:
            Y_pred.append(y.item())
        for y in y_true:
            Y_true.append(y.item())

    accuracy_score = sklearn.metrics.accuracy_score(Y_true, Y_pred)
    precision_score = sklearn.metrics.precision_score(Y_true, Y_pred, labels=labels, average=average)
    f1_score = sklearn.metrics.f1_score(Y_true, Y_pred, labels=labels, average=average)

    return accuracy_score, precision_score, f1_score

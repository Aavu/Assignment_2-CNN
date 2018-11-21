import numpy as np
import torch
import torch.utils.data
from torch.autograd import Variable

torch.manual_seed(0)
np.random.seed(0)


def prepare_datasets(speech_path='./data/speech_train.npy'
                     , music_path='./data/music_train.npy'
                     , splits=None):
    """
    Prepare data from training and evaluation
    :param speech_path: Path to speech npy file
    :param music_path: Path to music npy file
    :param splits: list of split percentages for dataset
    :return: train, validation and test sets
    """

    if splits is None:
        splits = [0.7, 0.15, 0.15]
    assert np.sum(splits) == 1
    assert splits[0] != 0
    assert splits[1] != 0
    assert splits[2] != 0

    # load data into torch Tensors

    speech_train = torch.Tensor(np.load(speech_path))
    music_train = torch.Tensor(np.load(music_path))

    # generate labels: Speech = 0; Music= 1

    labels_speech = torch.LongTensor(np.zeros(speech_train.size(0)))
    labels_music = torch.LongTensor(np.ones(music_train.size(0)))

    X = torch.cat((speech_train, music_train))
    y = torch.cat((labels_speech, labels_music))

    # split dataset into training validation and test. 0.7, 0.15, 0.15 split

    n_points = y.size(0)

    train_split = (0, int(splits[0] * n_points))
    val_split = (train_split[1], train_split[1] + int(splits[1] * n_points))
    test_split = (val_split[1], val_split[1] + int(splits[2] * n_points))

    shuffle_indices = np.random.permutation(np.arange(n_points))

    train_indices = torch.LongTensor(shuffle_indices[train_split[0]:train_split[1]])
    val_indices = torch.LongTensor(shuffle_indices[val_split[0]:val_split[1]])
    test_indices = torch.LongTensor(shuffle_indices[test_split[0]:test_split[1]])

    train_set = (X[train_indices], y[train_indices])
    val_set = (X[val_indices], y[val_indices])
    test_set = (X[test_indices], y[test_indices])

    # create torch Datasets

    train_set = torch.utils.data.TensorDataset(train_set[0], train_set[1])
    val_set = torch.utils.data.TensorDataset(val_set[0], val_set[1])
    test_set = torch.utils.data.TensorDataset(test_set[0], test_set[1])

    return train_set, val_set, test_set


def evaluate(data_loader, model, criterion, cuda):
    """
    Evaluate the trained model
    :param data_loader: pytorch dataloader for eval data
    :param model: pytorch model to be evaluated
    :param criterion: loss function used to compute loss
    :param cuda: boolean for whether to use gpu
    :return: loss and accuracy
    """
    model.eval()
    loss = 0
    correct = 0
    n_examples = 0
    for batch_i, batch in enumerate(data_loader):
        data, target = batch
        if cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            data, target = Variable(data), Variable(target)
        output = model(data)
        loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        if cuda:
            correct += pred.eq(target.data.view_as(pred)).gpu().sum()
        else:
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        n_examples += pred.size(0)

    loss /= n_examples
    accuracy = 100.0 * float(correct) / float(n_examples)
    return loss, accuracy


def save(model, path):
    """
    Save model
    :param model: pytorch model to be saved
    :param path: path for model to be saved
    """
    torch.save(model.state_dict(), path)


def load(path):
    """
    load model
    :param path: path of model to be loaded
    :return: model state_dict
    """
    return torch.load(path)

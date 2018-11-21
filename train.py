import argparse
import os

import numpy as np
import torch.nn as nn
import torch.utils.data

from models import ConvNet, MyModel
from utils import prepare_datasets, evaluate, save, load

dir_path = os.path.dirname(os.path.realpath(__file__))

# Training settings
parser = argparse.ArgumentParser(description='HW 2: Music/Speech CNN')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M', default=0.9,
                    help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', default=50,
                    help='number of epochs to train')
parser.add_argument('--model', default='convnet',
                    choices=['convnet', 'mymodel'],
                    help='which model to train/evaluate')
parser.add_argument('--save-dir', default='models/')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(0)
np.random.seed(0)
if args.cuda:
    torch.cuda.manual_seed(0)

# fetch torch Datasets
train_set, val_set, test_set = prepare_datasets(splits=[0.7, 0.15, 0.15])

# create torch DataLoaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1000)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

# initialize the model
if args.model == 'convnet':
    model = ConvNet()
elif args.model == 'mymodel':
    model = MyModel()
else:
    raise Exception('Incorrect model name')

if args.cuda:
    model.cuda()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


def train(epoch):
    """
    Runs training for specified number of epochs
    :param epoch: denotes the epoch number for printing
    :type epoch: int
    """
    mean_training_loss = 0.0
    model.train()
    for i, batch in enumerate(train_loader):
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        mean_training_loss += loss.item()
        mean_training_loss /= len(train_loader)
        print('Training Epoch: [{}][{}/{}]\t' 'Training Loss: {:.6f}'.format(epoch, i, len(train_loader) - 1, mean_training_loss))


# Save model with best val accuracy
best_val_acc = 0.0
best_model = {}
for i in range(args.epochs):
    train(i)
    val_loss, val_acc = evaluate(val_loader, model, criterion, args.cuda)
    if val_acc > best_val_acc:
        best_val_acc = float(val_acc)
        best_model = model
    print('Validation Loss: {:.6f} \t'
          'Validation Acc.: {:.6f}'.format(val_loss, val_acc))
    save(best_model, args.save_dir + args.model + '.pt')
    print("best accuracy:", best_val_acc)


def test(model):
    """
    Evaluates the model
    :param model: The model to be evaluated
    """
    test_loss, test_acc = evaluate(test_loader, model, criterion, args.cuda)
    print('Test Loss: {:.6f} \t' 'Test Acc.: {:.6f}'.format(test_loss, test_acc))


# Load best model and test
test_model = ConvNet()
if args.model == 'mymodel':
    test_model = MyModel()

if args.cuda:
    test_model.cuda()

test_model.load_state_dict(load(args.save_dir + args.model + '.pt'))
test(test_model)

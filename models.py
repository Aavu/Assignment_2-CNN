import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        #############################################################################
        # TODO: Implement the ConvNet as per the instructions given in the assignment
        # 3 layer CNN followed by 1 fully connected layer. Use only ReLU activation
        #   CNN #1: k=8, f=3, p=1, s=1
        #   Max Pool #1: pooling size=2, s=2
        #   CNN #2: k=16, f=3, p=1, s=1
        #   Max Pool #2: pooling size=4, s=4
        #   CNN #3: k=32, f=3, p=1, s=1
        #   Max Pool #3: pooling size=4, s=4
        #   FC #1: 64 hidden units                                                             
        #############################################################################
        self.l1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.l2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(4))
        self.l3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(4))
        self.fc1 = nn.Sequential(nn.Linear(512, 64), nn.ReLU())
        self.fc2 = nn.Linear(64, 2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass for the model
        #############################################################################
        N, H, W = x.size()
        x = x.reshape(-1, 1, H, W)
        x = self.l3(self.l2(self.l1(x)))
        x = x.view(N, -1)
        x = self.fc1(x)
        return self.fc2(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Implement your own model based on the hyperparameters of your choice
        ############################################################################# 
        self.l1 = nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2))
        self.l2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(4))
        self.l3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(4))
        self.l4 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(4))
        self.fc1 = nn.Sequential(nn.Linear(64, 16), nn.ReLU())
        self.fc3 = nn.Linear(16, 2)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass for your model
        ############################################################################# 
        N, H, W = x.size()
        x = x.reshape(-1, 1, H, W)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = x.view(N, -1)
        x = self.fc1(x)
        return self.fc3(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

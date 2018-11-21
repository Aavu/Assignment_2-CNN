import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        """
        Inititialize the layers and parameters
        """
        super(ConvNet, self).__init__()
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

    def forward(self, x):
        """
        Define the forward pass
        :param x: training data
        :return: output of the last layer
        """
        N, H, W = x.size()
        x = x.reshape(-1, 1, H, W)
        x = self.l3(self.l2(self.l1(x)))
        x = x.view(N, -1)
        x = self.fc1(x)
        return self.fc2(x)


class MyModel(nn.Module):
    def __init__(self):
        """
        Inititialize the layers and parameters
        """
        super(MyModel, self).__init__()
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

    def forward(self, x):
        """
        Define the forward pass
        :param x: training data
        :return: output of the last layer
        """
        N, H, W = x.size()
        x = x.reshape(-1, 1, H, W)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = x.view(N, -1)
        x = self.fc1(x)
        return self.fc3(x)

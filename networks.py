import torch.nn as nn
import torch
import torch.nn.functional as F


"""
CartPole network
"""

class MLP(nn.Module):
  def __init__(self, state_dim, action_dim, hidden_dim=400):
    super(MLP, self).__init__()
    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, action_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)


#car racing env

class CNN(nn.Module):

  def __init__(self, history_length=0, n_classes=3):
    super(CNN, self).__init__()
    # TODO : define layers of a convolutional neural network
    self.conv1 = nn.Conv2d(in_channels=1 + history_length, out_channels=8, kernel_size=3)  # 96*96*8
    self.maxpool1 = nn.MaxPool2d(kernel_size=2)
    self.batchnorm1 = nn.BatchNorm2d(8)  # 48*48*8

    self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)  # 48*48*16
    self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # 24*24*16
    self.batchnorm2 = nn.BatchNorm2d(16)  # 24*24*16

    self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3)  # 24*24*16
    # self.maxpool3 = nn.MaxPool2d(kernel_size=2) #12*12*32
    self.batchnorm3 = nn.BatchNorm2d(16)  # 16*20*20

    self.fc1 = nn.Linear(20 * 20 * 16, 1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc3 = nn.Linear(512, 128)
    self.fc4 = nn.Linear(128, n_classes)

  def forward(self, x):
    # TODO: compute forward pass
    x = F.relu(self.conv1(x))
    x = self.batchnorm1(self.maxpool1(x))

    x = F.relu(self.conv2(x))
    x = self.batchnorm2(self.maxpool2(x))

    x = F.relu(self.conv3(x))
    x = self.batchnorm3(x)

    #print("x shape=", x.shape)
    # torch.flatten(x,1)
    x = torch.flatten(x, 1)
    #print("x shape=", x.shape)

    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5)
    x = F.relu(self.fc2(x))
    x = F.dropout(x, p=0.5)
    x = F.relu(self.fc3(x))
    x = F.dropout(x, p=0.5)
    x = self.fc4(x)
    x = F.log_softmax(x, dim=1)

    return x
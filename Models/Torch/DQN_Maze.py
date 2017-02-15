import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=2)
        self.conv2 = nn.Conv2d(8, 4, 3, stride=1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # print(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

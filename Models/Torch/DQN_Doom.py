import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(4, 42, 42), actions=1):
        super(DQN, self).__init__()

        img_size = input_size[1]
        self.conv1 = nn.Conv2d(4, 32, 8, padding=0, stride=4)
        img_size = int((img_size - 8) / 4 + 1)
        self.conv2 = nn.Conv2d(32, 64, 4, padding=0, stride=2)
        img_size = int((img_size - 4) / 2 + 1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=0, stride=1)
        img_size = int((img_size - 3) / 1 + 1)
        self.fc1 = nn.Linear(img_size * img_size * 64, 256)
        self.fc2 = nn.Linear(256, actions)

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size()[0], -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)

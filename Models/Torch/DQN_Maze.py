import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(1, 7, 7)):
        super(DQN, self).__init__()
        img_size = int(input_size[1])
        # print(img_size)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=2, stride=2)
        img_size = int((img_size - 3 + 4) / 2 + 1)
        # print(img_size)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=2, stride=2)
        img_size = int((img_size - 3 + 4) / 2 + 1)
        # print(img_size)
        # self.flatten_size = img_size * img_size * 8
        self.fc1 = nn.Linear(img_size * img_size * 16, img_size * img_size * 4)
        self.fc2 = nn.Linear(img_size * img_size * 4, 4)

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        x = F.relu(self.conv1(x))
        # print(x)
        x = F.relu(self.conv2(x))
        # print(x)
        x = x.view(x.size()[0], -1)
        # print(x)
        x = F.relu(self.fc1(x))
        # print(x)
        return self.fc2(x)

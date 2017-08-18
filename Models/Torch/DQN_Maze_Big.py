import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(1, 60, 60), actions=1):
        super(DQN, self).__init__()

        self.actions = actions

        img_size = input_size[1]
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.fc1 = nn.Linear(img_size * img_size * 32, 128)
        self.qvals = nn.Linear(128, actions)
        # self.value_branch = nn.Linear(128, 1)
        # self.adv_branch = nn.Linear(128, actions)

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # Flatten the final conv layer
        x = x.view(x.size()[0], -1)

        # FC layer
        x = F.relu(self.fc1(x))

        x = self.qvals(x)

        return x

        # Value branch
        # v = self.value_branch(x).expand(x.size(0), self.actions)

        # Advantages branch
        # advs = self.adv_branch(x)
        # advs_mean = advs.mean(1).expand(x.size(0), self.actions)

        # Dueling output
        # outputs = v + (advs - advs_mean)

        # return outputs

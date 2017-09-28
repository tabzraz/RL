import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(1, 60, 60), actions=1):
        super(DQN, self).__init__()

        self.actions = actions

        img_size = input_size[1]

        self.fc1 = nn.Linear(img_size * img_size, 64)
        self.fc2 = nn.Linear(64, 32)

        self.qvals = nn.Linear(32, actions)
        # self.value_branch = nn.Linear(128, 1)
        # self.adv_branch = nn.Linear(128, actions)

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        x = x.contiguous() 
        # Flatten the image
        x = x.view(x.size()[0], -1)

        # FC layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

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

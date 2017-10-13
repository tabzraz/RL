import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor
import torch


class DQN_Distrib(nn.Module):

    def __init__(self, input_size=(1, 60, 60), actions=1, atoms=11, V_Min=-1, V_Max=1):
        super(DQN_Distrib, self).__init__()

        self.actions = actions
        self.atoms = atoms

        img_size = input_size[1]
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)

        self.fc1 = nn.Linear(img_size * img_size * 32, actions * atoms * 2)
        self.qvals = nn.Linear(actions * atoms * 2, actions * atoms)
        self.softmax = nn.Softmax2d()
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

        x = x.view(x.size()[0], self.atoms, self.actions, 1)

        # print(x)

        x = self.softmax(x)

        # print("Softmaxed" ,x)

        # print(x)
        x = x.view(x.size()[0], self.atoms, self.actions)
        # print(x)
        x = x.transpose(1, 2)
        # print(x)
        # print(x)
        # print(x)
        # print(x)
        # print(self.softmax(x))
        # all_actions_q_val_distribs = []
        # for action in range(self.actions):
        #     q_val_distrib = x[:, action, :]
        #     print(q_val_distrib)
        #     # q_val_distrib = q_val_distrib.view(x.size()[0], self.atoms)
        #     q_val_distrib = nn.Softmax(q_val_distrib)
        #     print(q_val_distrib)
        #     all_actions_q_val_distribs.append(q_val_distrib)

        # x = torch.stack(all_actions_q_val_distribs, dim=1)

        return x

        # Value branch
        # v = self.value_branch(x).expand(x.size(0), self.actions)

        # Advantages branch
        # advs = self.adv_branch(x)
        # advs_mean = advs.mean(1).expand(x.size(0), self.actions)

        # Dueling output
        # outputs = v + (advs - advs_mean)

        # return outputs

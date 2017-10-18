import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(1, 60, 60), actions=1):
        super(DQN, self).__init__()

        self.actions = actions

        img_size = input_size[1]
        # print("{} x {} -> img_size)
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2)
        img_size = int((img_size + 2 * 0 - 4) / 2 + 1)
        # print(img_size)
        self.conv2 = nn.Conv2d(8, 8, 4, stride=2)
        img_size = int((img_size + 2 * 0 - 4) / 2 + 1)
        print(img_size)
        # self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        # img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        # self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        # img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.fc1 = nn.Linear(img_size * img_size * 8, 128)
        self.qvals = nn.Linear(128, actions)

        self.action_matrix = nn.Linear(actions, 128)
        self.img_size = img_size
        print(img_size)
        self.transition_fc1 = nn.Linear(128, img_size * img_size * 1)
        self.deconv1 = nn.ConvTranspose2d(1, 8, 4, padding=0, stride=2)
        self.deconv2 = nn.ConvTranspose2d(8, 1, 4, padding=0, stride=2)
        # self.deconv3 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        # self.deconv4 = nn.ConvTranspose2d(32, 1, 3, stride=2)
        # self.value_branch = nn.Linear(128, 1)
        # self.adv_branch = nn.Linear(128, actions)

    def forward(self, x, action=None):
        if next(self.parameters()).is_cuda:
            x = x.cuda()
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # Flatten the final conv layer
        # print(x)
        x = x.view(x.size()[0], -1)
        # FC layer
        x = F.relu(self.fc1(x))

        if action is not None:
            y = self.action_matrix(action)
            # print("Actioned",y)
            y = y + x
            # print("Added", y)
            y = self.transition_fc1(y)
            # print("Transed", y)
            # Unflatten y
            y = y.view(x.size()[0], 1, self.img_size, self.img_size)
            # print("View", y)
            y = F.relu(self.deconv1(y))
            # print("deconv1",y)
            y = self.deconv2(y)
            # print("deconv2",y)
            # y = self.deconv3(y)
            # print("deconv3", y)
            # y = self.deconv4(y)
            # print("deconv4",y)

        x = self.qvals(x)

        if action is not None:
            return x, y

        return x

        # Value branch
        # v = self.value_branch(x).expand(x.size(0), self.actions)

        # Advantages branch
        # advs = self.adv_branch(x)
        # advs_mean = advs.mean(1).expand(x.size(0), self.actions)

        # Dueling output
        # outputs = v + (advs - advs_mean)

        # return outputs

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(1, 60, 60), actions=1):
        super(DQN, self).__init__()

        self.actions = actions

        img_size = input_size[1]
        # print("{} x {} -> img_size)
        print("\n---DQN Architecture---")
        print("Input: {} x {} x 1".format(img_size, img_size))
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2)
        img_size = int((img_size + 2 * 0 - 3) / 2 + 1)
        print("Conv1, 3 x 3 filter, stride 2 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 16))
        # print(img_size)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2)
        img_size = int((img_size + 2 * 0 - 3) / 2 + 1)
        print("Conv2, 3 x 3 filter, stride 2 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 16))

        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        img_size = int((img_size + 2 * 0 - 3) / 1 + 1)
        print("Conv3, 3 x 3 filter, stride 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 16))

        # print(img_size)
        # self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        # img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        # self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        # img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.fc1 = nn.Linear(img_size * img_size * 16, 128)
        print("FC1 {} -> {}".format(img_size * img_size * 16, 128))
        self.qvals = nn.Linear(128, actions)
        print("---\nQVals {} -> {}".format(128, actions))

        print("---\n")

    def forward(self, x, action=None):
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # Flatten the final conv layer
        # print(x)
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

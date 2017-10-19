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
        self.conv1 = nn.Conv2d(1, 8, 4, stride=2)
        img_size = int((img_size + 2 * 0 - 4) / 2 + 1)
        print("Conv1, 4 x 4 filter, stride 2 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 8))
        # print(img_size)
        self.conv2 = nn.Conv2d(8, 8, 4, stride=2)
        img_size = int((img_size + 2 * 0 - 4) / 2 + 1)
        print("Conv2, 4 x 4 filter, stride 2 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 8))
        # print(img_size)
        # self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        # img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        # self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        # img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.fc1 = nn.Linear(img_size * img_size * 8, 128)
        print("FC1 {} -> {}".format(img_size * img_size * 8, 128))
        self.qvals = nn.Linear(128, actions)
        print("---\nQVals {} -> {}".format(128, actions))

        self.action_matrix = nn.Linear(actions, 128)
        print("---\nActions {} -> {}".format(actions, 128))

        self.img_size = img_size
        # print(img_size)
        self.transition_fc1 = nn.Linear(128, img_size * img_size * 1)
        print("Transition FC1 {} -> {} x {} x {}".format(128, img_size, img_size, 8))
        self.deconv1 = nn.ConvTranspose2d(1, 8, 4, padding=0, stride=2)
        img_size = (img_size - 1) * 2 + 4
        print("DeConv1, 4 x 4 filter, stride 2 -> {} x {} x {}".format(img_size, img_size, 8))

        self.deconv2 = nn.ConvTranspose2d(8, 8, 4, padding=0, stride=2)
        img_size = (img_size - 1) * 2 + 4
        print("DeConv2, 4 x 4 filter, stride 2 -> {} x {} x {}".format(img_size, img_size, 8))

        # self.deconv3 = nn.ConvTranspose2d(8, 8, 2, padding=0, stride=1)
        # img_size = (img_size - 1) * 1 - 2 * 1 + 2
        self.deconv3 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
        img_size = int((img_size + 2 * 1 - 3) / 1 + 1)
        print("DeConv3, 3 x 3 filter, stride 1 -> {} x {} x {}".format(img_size, img_size, 8))
        self.deconv4 = nn.Conv2d(8, 1, 3, stride=1, padding=1)
        img_size = int((img_size + 2 * 1 - 3) / 1 + 1)
        print("DeConv4, 3 x 3 filter, stride 1 -> {} x {} x {}".format(img_size, img_size, 8))

        # self.deconv4 = nn.ConvTranspose2d(8, 1, 1, padding=0, stride=1)
        # img_size = (img_size - 1) * 1 + 1
        # print("DeConv4, 1 x 1 filter, stride 1 -> {} x {} x {}".format(img_size, img_size, 1))
        # self.deconv3 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        # self.deconv4 = nn.ConvTranspose2d(32, 1, 3, stride=2)
        # self.value_branch = nn.Linear(128, 1)
        # self.adv_branch = nn.Linear(128, actions)
        print("---\n")

    def forward(self, x, action=None):
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        if action is not None:
            input_state = x + 0
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
            y = F.relu(self.deconv2(y))
            y = y + input_state.expand(x.size()[0], 8, y.size()[2], y.size()[3])
            y = F.relu(self.deconv3(y))
            # print("deconv1",y)
            y = self.deconv4(y)

            # Residual connection
            # y = y + input_state

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

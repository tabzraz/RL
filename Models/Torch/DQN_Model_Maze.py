import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):

    def __init__(self, input_size=(1, 60, 60), actions=1):
        super(DQN, self).__init__()

        self.actions = actions

        img_size = input_size[1]
        # print("{} x {} -> img_size)
        print("\n---DQN Architecture---")
        print("Input: {} x {} x 1".format(img_size, img_size))
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        print("Conv1, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 32))

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        print("Conv2, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 32))

        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        print("Conv3, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 32))
 
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        print("Conv4, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 32))

        # print(img_size)
        # self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        # img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        # self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        # img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        self.fc1 = nn.Linear(img_size * img_size * 32, 128)
        print("FC1 {} -> {}".format(img_size * img_size * 32, 128))
        # self.qvals1 = nn.Linear(128, 128)
        print("---")
        # print("QHead {} -> {}".format(128, 128))
        self.qvals = nn.Linear(128, actions)
        print("QVals {} -> {}".format(128, actions))

        self.action_matrix = nn.Linear(actions, 128)
        print("---\nActions {} -> {}".format(actions, 128))
        print("Actions x Embedding 128 -> 128")

        self.img_size = img_size
        # print(img_size)
        # gg = 128 // (img_size * img_size)
        self.transition_fc1 = nn.Linear(128, img_size * img_size * 32)
        print("Transition FC1 {} -> {}".format(128, img_size * img_size * 32))
        print("Reshape {} -> {} x {} x {}".format(img_size * img_size * 32, img_size, img_size, 32))

        filter_size = 3
        deconv_stride = 2

        self.deconv1 = nn.ConvTranspose2d(32, 32, filter_size, padding=0, stride=2)
        img_size = (img_size - 1) * 2 + filter_size - 2 * 0 + 0 * 1
        print("DeConv1, 3 x 3 filter, stride 2 -> {} x {} x {}".format(img_size, img_size, 32))

        self.deconv2 = nn.ConvTranspose2d(32, 32, filter_size, padding=1, stride=2, output_padding=0)
        img_size = (img_size - 1) * 2 + filter_size + 0 - 2 * 1
        print("DeConv2, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}".format(img_size, img_size, 32))

        self.deconv2_ = nn.ConvTranspose2d(32, 32, filter_size, padding=2, stride=2)
        img_size = (img_size - 1) * 2 + filter_size - 2 * 2
        print("DeConv3, 3 x 3 filter, stride 2, padding 2 -> {} x {} x {}".format(img_size, img_size, 32))

        self.deconv2__ = nn.ConvTranspose2d(32, 31, 2, padding=0, stride=2, output_padding=0)
        img_size = (img_size - 1) * 2 + 2 - 2 * 0 + 0
        print("DeConv4, 2 x 2 filter, stride 2 -> {} x {} x {}".format(img_size, img_size, 31))

        print("Concat State onto channels -> {} x {} x {}".format(img_size, img_size, 32))
        # self.deconv3 = nn.ConvTranspose2d(8, 8, 2, padding=0, stride=1)
        # img_size = (img_size - 1) * 1 - 2 * 1 + 2
        self.deconv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        img_size = int((img_size + 2 * 1 - 3) / 1 + 1)
        print("DeConv5, 3 x 3 filter, stride 1, padding 1 -> {} x {} x {}".format(img_size, img_size, 32))
        self.deconv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        img_size = int((img_size + 2 * 1 - 3) / 1 + 1)
        print("DeConv6, 3 x 3 filter, stride 1, padding 1 -> {} x {} x {}".format(img_size, img_size, 32))

        self.deconv5 = nn.Conv2d(32, 1, 1, stride=1, padding=0)
        img_size = int((img_size + 2 * 1 - 3) / 1 + 1)
        print("DeConv7, 1 x 1 filter, stride 1 -> {} x {} x {}".format(img_size, img_size, 1))

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
                action = action.cuda()

        if action is not None:
            input_state = x + 0
        # Conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # Flatten the final conv layer
        # print(x)
        x = x.view(x.size()[0], -1)
        # FC layer

        if action is not None:
            ay = self.action_matrix(action)
            # print("Actioned",y)
            y = ay * x
            # print("Added", y)
            y = self.transition_fc1(y)
            # print("Transed", y)
            # Unflatten y
            y = y.view(x.size()[0], -1, self.img_size, self.img_size)
            # print("View", y)
            y = F.relu(self.deconv1(y))
            y = F.relu(self.deconv2(y))
            y = F.relu(self.deconv2_(y))
            # y = F.relu(self.deconv2__(y))
            y = F.relu(self.deconv2__(y))
            # y = y + input_state

            # Makes all the difference...
            y = torch.cat((y, input_state), dim=1)
            y = F.relu(self.deconv3(y))
            y = F.relu(self.deconv4(y))

            y = self.deconv5(y)

            # Residual connection
            # y = y + input_state

        x = F.relu(self.fc1(x))
        # x = self.qvals1(x)
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

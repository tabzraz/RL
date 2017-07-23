import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(4, 42, 42), actions=1):
        super(DQN, self).__init__()
        img_size_x = int(input_size[1])
        img_size_y = int(input_size[2])
        # print(img_size)
        self.convs = []
        # conv_layers = 4

        # First layer
        self.conv_1 = nn.Conv2d(4, 32, 7, padding=2, stride=2)
        img_size_x = int((img_size_x - 7 + 4) / 2 + 1)
        img_size_y = int((img_size_y - 7 + 4) / 2 + 1)
        # print(img_size)

        for l in range(3):
            conv_layer = nn.Conv2d(32, 32, 3, padding=2, stride=2)
            self.convs.append(conv_layer)
            self.add_module("Conv Layer {}".format(l + 1), conv_layer)
            img_size_x = int((img_size_x - 3 + 4) / 2 + 1)
            img_size_y = int((img_size_y - 3 + 4) / 2 + 1)

        # for l in range(2):
        #     conv_layer = nn.Conv2d(32, 32, 3, padding=2, stride=1)
        #     self.convs.append(conv_layer)
        #     self.add_module("Conv Layer {}".format(l + 3), conv_layer)
        #     img_size_x = int((img_size_x - 3 + 4) / 1 + 1)
        #     img_size_y = int((img_size_y - 3 + 4) / 1 + 1)
            # img_size = int((img_size - 3 + 4) / 2 + 1)
            # print(img_size)
        # img_size = int((img_size - 3 + 4) / 2 + 1)
        # print(img_size * 16, img_size * 4, actions)
        # self.flatten_size = img_size * img_size * 8
        self.fc1 = nn.Linear(img_size_x * img_size_y * 32, img_size_x * img_size_y * 4)
        self.fc2 = nn.Linear(img_size_x * img_size_y * 4, actions)

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        x = F.relu(self.conv_1(x))
        # print(x)

        for conv in self.convs:
            x = F.relu(conv(x))
            # print(x)
        # print(x)
        # x = F.relu(self.conv2(x))
        # x = self.batch2(x)
        # print(x)
        x = x.view(x.size()[0], -1)
        # print(x)
        x = F.relu(self.fc1(x))
        # print(x)
        # print(x)
        return self.fc2(x)

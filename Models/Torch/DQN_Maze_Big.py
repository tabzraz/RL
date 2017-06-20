import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(1, 7, 7), actions=4, conv_layers=4, batchnorm=False):
        super(DQN, self).__init__()
        img_size = int(input_size[1])
        # print(img_size)
        self.convs = []
        self.batch_norms = []
        self.batchnorm = batchnorm

        # First layer
        self.conv_1 = nn.Conv2d(1, 32, 7, padding=2, stride=2)
        if self.batchnorm:
            self.batch_1 = nn.BatchNorm2d(16, affine=False)
        img_size = int((img_size - 7 + 4) / 2 + 1)

        for l in range(1, conv_layers):
            conv_layer = nn.Conv2d(32, 32, 5, padding=2, stride=1)
            self.convs.append(conv_layer)
            self.add_module("Conv Layer {}".format(l + 1), conv_layer)
            if self.batchnorm:
                batch_layer = nn.BatchNorm2d(16, affine=False)
                self.batch_norms.append(batch_layer)
                self.add_module("BatchNorm Layer {}".format(l + 1), batch_layer)
            else:
                # Hack for laziness
                self.batch_norms.append(None)
            img_size = int((img_size - 5 + 4) / 1 + 1)
        # print(img_size)
        # img_size = int((img_size - 3 + 4) / 2 + 1)
        # print(img_size)
        # self.flatten_size = img_size * img_size * 8
        self.fc1 = nn.Linear(img_size * img_size * 32, actions * 4)
        self.fc2 = nn.Linear(actions * 4, actions)

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        x = F.relu(self.conv_1(x))
        if self.batchnorm:
            x = self.batch_1(x)

        for conv, batch in zip(self.convs, self.batch_norms):
            x = F.relu(conv(x))
            if self.batchnorm:
                x = batch(x)
        # print(x)
        # x = F.relu(self.conv2(x))
        # x = self.batch2(x)
        # print(x)
        x = x.view(x.size()[0], -1)
        # print(x)
        x = F.relu(self.fc1(x))
        # print(x)
        return self.fc2(x)

import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, input_size=(4, 42, 42), actions=1):
        super(DQN, self).__init__()

        img_size = 42
        print("\n---DQN Architecture---")
        print("Input: {} x {} x 4".format(img_size, img_size))
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        print("Conv1, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 32))

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        print("Conv1, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 32))

        self.conv3 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        print("Conv1, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 32))

        self.conv4 = nn.Conv2d(32, 32, 3, padding=1, stride=2)
        img_size = int((img_size - 3 + 2 * 1) / 2 + 1)
        print("Conv1, 3 x 3 filter, stride 2, padding 1 -> {} x {} x {}  (w x h x c)".format(img_size, img_size, 32))

        self.fc1 = nn.Linear(img_size * img_size * 32, 512)
        print("FC1 {} -> {}".format(img_size * img_size * 32, 512))

        self.qvals = nn.Linear(512, actions)
        print("---\nQVals {} -> {}".format(512, actions))

    def forward(self, x):
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.view(x.size()[0], -1)

        x = F.relu(self.fc1(x))

        return self.qvals(x)

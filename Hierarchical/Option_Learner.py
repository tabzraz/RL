import numpy as np
import torch
from torch.autograd import Variable


class Option_Learner:

    def __init__(self, primitive_actions, extra_actions):
        self.primitive_actions = primitive_actions
        self.options = [None for _ in range(extra_actions)]

    def choose_option(self, action):
        self.higher_action = action
        self.current_action = self.higher_action
        self.option_stack = []

    def act(self, env):
        action = self.current_action
        while action >= self.primitive_actions:
            state = env.env.state
            option = action - self.primitive_actions
            net, goal_state, allowed_actions = self.options[option]
            beta = 0.0
            if state == goal_state:
                beta = 1.0
            net.eval()
            state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
            q_vals = net(Variable(state, volatile=True)).cpu().data[0]
            q_vals_numpy = q_vals.numpy()
            q_vals_numpy = q_vals_numpy[:allowed_actions]
            action = np.max(q_vals_numpy)

        if self.action < self.primitive_actions:
            return self.action, 1.0

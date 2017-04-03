import numpy as np
import torch
from torch.autograd import Variable


class Option_Subgoal:

    # A primitive action is just an Option that terminates after 1 step
    def __init__(self, net, allowed_actions, subgoal_state, options):
        self.net = net
        self.allowed_actions = allowed_actions
        self.subgoal_state = subgoal_state
        self.options = options
        self.executing = False
        self.executing_option = None
        self.net.eval()

    def action(self, state):
        if not self.executing:
            # Pick an action because we are not currently executing another option
            state = torch.from_numpy(state).float().transpose_(0, 2).unsqueeze(0)
            q_vals = self.net(Variable(state, volatile=True)).cpu().data[0]
            q_vals_numpy = q_vals.numpy()
            q_vals_numpy = q_vals_numpy[:self.allowed_actions]
            action = np.max(q_vals_numpy)
            self.executing_option = self.options[action]

        return self.executing_option.action(state)

    def terminate(self, state):
        if self.executing:
            termed = self.executing_option.terminate(state)
            if termed:
                self.executing = False
        if self.executing:
            return False
        else:
            if state == self.subgoal_state:
                return True
            else:
                return False

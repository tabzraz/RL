import numpy as np


class Primitive_Options:

    # A primitive action is just an Option that terminates after 1 step
    def __init__(self):
        self.action = 0

    def choose_option(self, action):
        self.action = action
        self.steps = 0

    def act(self, env=None):
        self.steps += 1
        beta = 1.0
        return self.action, beta

    def action(self, state):
        return self.action

    def terminate(self, state):
        return True

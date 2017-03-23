import numpy as np


class Random_Macro_Actions:

    # A primitive action is just an Option that terminates after 1 step
    def __init__(self, num_primitive_actions, lengths_of_macros, seed=1, with_primitives=False):
        self.macro = 0
        np.random.seed(seed)
        self.macros = []
        if with_primitives:
            for i in range(num_primitive_actions):
                self.macros.append([i])
        for length in lengths_of_macros:
            macro = np.random.randint(low=0, high=num_primitive_actions, size=length, dtype=np.int32)
            self.macros.append(macro)

    def choose_option(self, action):
        self.macro = action
        self.steps = 0

    def act(self, env=None):
        action = self.macros[self.macro][self.steps]
        self.steps += 1
        if self.steps == len(self.macros[self.macro]):
            beta = 1.0
        else:
            beta = 0.0
        return action, beta

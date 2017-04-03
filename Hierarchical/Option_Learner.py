from .Option_Subgoal import Option_Subgoal
from .Primitive_Option import Primitive_Option
import copy


class Option_Learner:

    def __init__(self, primitive_actions):
        self.options = [Primitive_Option(i) for i in range(primitive_actions)]

    def choose_option(self, option):
        self.option = self.options[option]
        # print("Chosen:", option)

    def add_option(self, net, subgoal_state):
        current_options = copy.deepcopy(self.options)
        allowed_actions = len(current_options)
        option = Option_Subgoal(net, allowed_actions, subgoal_state, current_options)
        self.options.append(option)

    def action(self, state):
        return self.option.action(state)

    def terminate(self, state):
        return self.option.terminate(state)

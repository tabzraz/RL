class Primitive_Option:

    def __init__(self, action):
        self.action_num = action

    def action(self, state):
        return self.action_num

    def terminate(self, state):
        return True

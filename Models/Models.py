from DQN_Maze import model as DQN_Maze
from VIME_Maze import model as VIME_Maze


# Define all the models here
models = {}

# DQN Maze
for size in range(10):
    def model_creator(name, actions=4):
        return DQN_Maze(name=name, size=size, actions=actions)
    models["Maze-{}".format(size)] = model_creator

# VIME Maze
for size in range(10):
    def model_creator(name, actions=4):
        return VIME_Maze(name=name, size=size, actions=actions)
    models["Vime-Maze-{}".format(size)] = model_creator


def get_models(name):
    return models[name]

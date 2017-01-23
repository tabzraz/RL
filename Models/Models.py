import DQN_Maze
import VIME_Maze


# Define all the models here
models = {}

# DQN Maze
for size in range(10):
    def model_creator(name, actions=4):
        return DQN_Maze.model(name=name, size=size, actions=actions)
    models["Maze-{}".format(size)] = model_creator

# VIME Maze
for size in range(10):
    def model_creator(name, actions=4):
        return VIME_Maze.model(name=name, size=size, actions=actions)
    models["Vime-Maze-{}".format(size)] = model_creator


def get_models(name):
    return models[name]

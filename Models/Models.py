from .Torch import DQN_Maze


# Define all the models here
models = {}

# DQN Maze
models["Maze-2"] = DQN_Maze.DQN


def get_models(name):
    return models[name]

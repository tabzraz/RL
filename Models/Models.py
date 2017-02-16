# Define all the models here
torch_models = {}
models = {}

# --- Torch models ---
from .Torch import DQN_Maze

# DQN Maze
torch_models["Maze-2"] = DQN_Maze.DQN

# --- Tensorflow models ---
from DQN_Maze import model as DQN_Maze_Model

# DQN Maze
models["Maze-2"] = DQN_Maze_Model


def get_torch_models(name):
    return torch_models[name]

def get_models(name):
    return models[name]

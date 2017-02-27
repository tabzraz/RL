# Define all the models here
torch_models = {}
models = {}

## --- Torch models ---
from .Torch import DQN_Maze

# DQN Maze
torch_models["Maze-1"] = DQN_Maze.DQN


def get_torch_models(name):
   return torch_models[name]


# --- Tensorflow models ---
from .DQN_Maze import model as DQN_Maze_Model
from .DRQN_Maze import DRQN

# DQN Maze
for s in range(10):
    def model_creator(name, size=s, actions=4):
        return DQN_Maze_Model(name=name, size=size, actions=actions)
    models["Maze-{}-v1".format(s)] = model_creator
# DRQN Maze
for s in range(10):
    def model_creator(name, size=s, actions=4):
        return DRQN(name=name, size=size, actions=actions)
    models["DRQN-Maze-{}-v1".format(s)] = model_creator

def get_models(name):
    return models[name]

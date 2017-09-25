# --- Torch models ---
from .Torch import DQN_Maze, DQN_Maze_Big
from .Torch import DQN_Atari
from .Torch import DQN_Doom
from .Torch import NEC_Maze

# Define all the models here
torch_models = {}

# DQN Atari
torch_models["VentureNoFrameskip-v4"] = DQN_Atari.DQN
torch_models["MontezumaRevengeNoFrameskip-v4"] = DQN_Atari.DQN

# DQN VizDoom
torch_models["ppaquette/DoomMyWayHome-v0"] = DQN_Doom.DQN
torch_models["DoomMaze-v0"] = DQN_Doom.DQN
torch_models["DoomMazeHard-v0"] = DQN_Doom.DQN

for i in range(5):
    torch_models["Doom_Maze_{}-v0".format(i + 1)] = DQN_Doom.DQN
torch_models["DoomMazeBig-v0"] = DQN_Doom.DQN
torch_models["DoomMazeBigHard-v0"] = DQN_Doom.DQN

# DQN Maze
# torch_models["Maze-1"] = DQN_Maze.DQN
for s in range(10):
    def model_creator(actions, size=s):
        return DQN_Maze.DQN(input_size=(1, size * 7, size * 7), actions=actions)
    torch_models["Maze-{}-v1".format(s)] = model_creator
    torch_models["Maze-{}-v0".format(s)] = model_creator

# DQN Room
for n in range(100):
    def model_creator(actions, size=n):
        return DQN_Maze.DQN(input_size=(1, size, size), actions=actions)
    torch_models["Room-{}-v0".format(n)] = model_creator

# DQN Wide Maze
for n in range(50):
    def model_creator(actions, size=n):
        return DQN_Maze.DQN(input_size=(1, size * 10, size * 10), actions=actions)
    torch_models["Wide-Maze-{}-v0".format(n)] = model_creator

# DQN Med Maze
for n in range(50):
    def model_creator(actions, size=n):
        return DQN_Maze.DQN(input_size=(1, size * 5, size * 5), actions=actions)
    torch_models["Med-Maze-{}-v0".format(n)] = model_creator

    def model_creator_big(actions, size=n):
        return DQN_Maze_Big.DQN(input_size=(1, size * 5, size * 5), actions=actions)
    torch_models["Med-Maze-{}-v0-Big".format(n)] = model_creator_big

# DQN Thin Maze
for n in range(50):
    # def model_creator(actions, size=n):
        # return DQN_Maze.DQN(input_size=(1, size * 3, size * 3), actions=actions)

    def model_creator_big(actions, size=n):
        return DQN_Maze_Big.DQN(input_size=(1, size * 3, size * 3), actions=actions)
    torch_models["Thin-Maze-{}-v0-Big".format(n)] = model_creator_big
    torch_models["Thin-Maze-{}-v0".format(n)] = model_creator_big

    def nec_model_creator(embedding, size=n):
        return NEC_Maze.NEC_Embedding(input_size=(1, size * 3, size * 3), embedding=embedding)
    torch_models["NEC_Thin-Maze-{}-v0".format(n)] = nec_model_creator

def get_torch_models(name):
    return torch_models[name]


'''
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

'''
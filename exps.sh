#screen -mdS DQN fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name DQN"
screen -mdS Count_1_Beta fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Count_1_Eps --count --beta 0.01"
screen -mdS Count_2_Beta fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Count_2_Eps --count --beta 0.001"
screen -mdS Count_3_Beta fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Count_3_Eps --count --beta 0.0001"

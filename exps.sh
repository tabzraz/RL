#screen -mdS DQN fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_DQN_2 --seed 2"
screen -mdS Count_1 fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_Count_Beta_2 --count --beta 0.01 --seed 2"

#screen -mdS DQN fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_DQN_3 --seed 3"
screen -mdS Count_1 fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_Count_Beta_3 --count --beta 0.01 --seed 3"

#screen -mdS DQN fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_DQN_4 --seed 4"
screen -mdS Count_1 fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_Count_Beta_4 --count --beta 0.01 --seed 4"

#screen -mdS DQN fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_DQN_5 --seed 5"
screen -mdS Count_1 fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_Count_Beta_5 --count --beta 0.01 --seed 5"

#screen -mdS DQN fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_DQN_6 --seed 6"
screen -mdS Count_1 fish -c "python3 RL_Trainer_PyTorch.py --gpu --debug-eval --env Maze-4-v1 --name Mz_4_Count_Beta_6 --count --beta 0.01 --seed 6"

python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_64_DQN_uid_0 --env Maze-5-v1 --lr 0.0001 --seed 13 --t-max 1000000 --eps-start 1 --batch-size 64 --gpu --debug-eval
python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_64_DQN_uid_1 --env Maze-5-v1 --lr 0.0001 --seed 66 --t-max 1000000 --eps-start 1 --batch-size 64 --gpu --debug-eval
python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_64_DQN_uid_2 --env Maze-5-v1 --lr 0.0001 --seed 75 --t-max 1000000 --eps-start 1 --batch-size 64 --gpu --debug-eval

python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_256_Count_Cts_7_Conv_False_Beta_0.01_Eps_0.1_uid_30 --env Maze-5-v1 --lr 0.0001 --seed 13 --t-max 1000000 --eps-start 0.1 --batch-size 256 --count --beta 0.01 --cts-size 7 --gpu --debug-eval
python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_256_Count_Cts_7_Conv_False_Beta_0.01_Eps_0.1_uid_31 --env Maze-5-v1 --lr 0.0001 --seed 66 --t-max 1000000 --eps-start 0.1 --batch-size 256 --count --beta 0.01 --cts-size 7 --gpu --debug-eval
python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_256_Count_Cts_7_Conv_False_Beta_0.01_Eps_0.1_uid_32 --env Maze-5-v1 --lr 0.0001 --seed 75 --t-max 1000000 --eps-start 0.1 --batch-size 256 --count --beta 0.01 --cts-size 7 --gpu --debug-eval

python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_256_Count_Cts_7_Conv_True_Beta_0.01_Eps_0.1_uid_33 --env Maze-5-v1 --lr 0.0001 --seed 13 --t-max 1000000 --eps-start 0.1 --batch-size 256 --count --beta 0.01 --cts-size 7 --cts-conv --gpu --debug-eval
python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_256_Count_Cts_7_Conv_True_Beta_0.01_Eps_0.1_uid_34 --env Maze-5-v1 --lr 0.0001 --seed 66 --t-max 1000000 --eps-start 0.1 --batch-size 256 --count --beta 0.01 --cts-size 7 --cts-conv --gpu --debug-eval
python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_256_Count_Cts_7_Conv_True_Beta_0.01_Eps_0.1_uid_35 --env Maze-5-v1 --lr 0.0001 --seed 75 --t-max 1000000 --eps-start 0.1 --batch-size 256 --count --beta 0.01 --cts-size 7 --cts-conv --gpu --debug-eval

python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_128_Count_Cts_7_Conv_True_Beta_0.01_Eps_1_uid_16 --env Maze-5-v1 --lr 0.0001 --seed 66 --t-max 1000000 --eps-start 1 --batch-size 128 --count --beta 0.01 --cts-size 7 --cts-conv --gpu --debug-eval
python3 RL_Trainer_PyTorch.py --name Maze_5_Batch_128_Count_Cts_7_Conv_True_Beta_0.01_Eps_1_uid_17 --env Maze-5-v1 --lr 0.0001 --seed 75 --t-max 1000000 --eps-start 1 --batch-size 128 --count --beta 0.01 --cts-size 7 --cts-conv --gpu --debug-eval






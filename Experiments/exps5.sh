python3 ../Main.py --name Thin_Maze_8_10_stp_LR_0.0001_Gamma_0.9999_Batch_32_itrs_1_Xp_50k_Big_UCB_Bandit_0.1_Scaler_Count_10_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_4 --env Thin-Maze-8-v0 --lr 0.0001 --seed 7 --t-max 600000 --eps-start 0.05 --batch-size 32 --xp 50000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 10 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-8-v0-Big --count --beta 0.0001 --cts-size 10 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --ucb --gpu
python3 ../Main.py --name Thin_Maze_10_10_stp_LR_0.0001_Gamma_0.9999_Batch_32_itrs_1_Xp_50k_Big_UCB_Bandit_0.1_Scaler_Count_10_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_20 --env Thin-Maze-10-v0 --lr 0.0001 --seed 7 --t-max 600000 --eps-start 0.05 --batch-size 32 --xp 50000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 10 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-10-v0-Big --count --beta 0.0001 --cts-size 10 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --ucb --gpu

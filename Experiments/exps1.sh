python3 ../Main.py --name Thin_Maze_10_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_100k_Prioritized_0.3_Alpha_1_NScaler_Bandit_0.1_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_0 --env Thin-Maze-10-Neg-v0 --lr 0.0001 --seed 7 --t-max 1000000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --priority --negative-td-scaler 1 --alpha 0.3 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --gpu
python3 ../Main.py --name Thin_Maze_10_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_100k_Prioritized_0.5_Alpha_1_NScaler_Bandit_0.01_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_16 --env Thin-Maze-10-Neg-v0 --lr 0.0001 --seed 7 --t-max 1000000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --priority --negative-td-scaler 1 --alpha 0.5 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --gpu

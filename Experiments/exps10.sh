python3 ../Main.py --name Thin_Maze_6_100_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_50k_Big_Prioritized_0.5_Alpha_1_NScaler_Count_6_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_9 --env Thin-Maze-6-v0 --lr 0.0001 --seed 14 --t-max 500000 --eps-start 0.05 --batch-size 32 --xp 50000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-6-v0-Big --priority --negative-td-scaler 1 --alpha 0.5 --count --beta 0.0001 --cts-size 6 --stale-limit 1000000 --gpu
python3 ../Main.py --name Thin_Maze_6_100_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_50k_Big_Prioritized_0.3_Alpha_1_NScaler_Bandit_0.01_Scaler_Count_6_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_25 --env Thin-Maze-6-v0 --lr 0.0001 --seed 14 --t-max 500000 --eps-start 0.05 --batch-size 32 --xp 50000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-6-v0-Big --priority --negative-td-scaler 1 --alpha 0.3 --count --beta 0.0001 --cts-size 6 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --gpu
python3 ../Main.py --name Thin_Maze_6_100_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_50k_Big_Prioritized_0.7_Alpha_1_NScaler_Bandit_0.01_Scaler_Count_6_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_41 --env Thin-Maze-6-v0 --lr 0.0001 --seed 14 --t-max 500000 --eps-start 0.05 --batch-size 32 --xp 50000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-6-v0-Big --priority --negative-td-scaler 1 --alpha 0.7 --count --beta 0.0001 --cts-size 6 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --gpu

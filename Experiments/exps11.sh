python3 ../Main.py --name Thin_Maze_14_100_stp_LR_0.0001_Gamma_0.9999_Batch_32_itrs_1_Xp_500k_Big_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_10 --env Thin-Maze-14-v0 --lr 0.0001 --seed 21 --t-max 800000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --model Thin-Maze-14-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --gpu

python3 ../Main.py --name Thin_Maze_10_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9999_Decay_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_2 --env Thin-Maze-10-Neg-v0 --lr 0.0001 --seed 21 --t-max 1000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu
python3 ../Main.py --name Thin_Maze_10_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Bandit_0.01_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_18 --env Thin-Maze-10-Neg-v0 --lr 0.0001 --seed 21 --t-max 1000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --gpu

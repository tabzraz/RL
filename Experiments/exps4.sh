python3 ../Main.py --name Thin_Maze_8_Deadly_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_100k_CEps_0.9999_Decay_Count_12_Stle_1000k_Beta_0.1_Eps_0.05_0.05_k_uid_3 --env Thin-Maze-8-Deadly-v0 --lr 0.0001 --seed 28 --t-max 1000000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.1 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu
python3 ../Main.py --name Thin_Maze_8_Deadly_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_100k_Bandit_0.1_Scaler_Count_12_Stle_1000k_Beta_0.01_Eps_0.05_0.05_k_uid_19 --env Thin-Maze-8-Deadly-v0 --lr 0.0001 --seed 28 --t-max 1000000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --gpu
python3 ../Main.py --name Thin_Maze_8_Deadly_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_100k_Bandit_0.01_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_35 --env Thin-Maze-8-Deadly-v0 --lr 0.0001 --seed 28 --t-max 1000000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --gpu

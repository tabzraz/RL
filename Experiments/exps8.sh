python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_1k_CEps_0.9999_Decay_100k_2Replay_0.001_Thr_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_7 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 1000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.001 --bonus-replay --bonus-replay-size 100000 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_1k_0.5_Bandit_0.1_Scaler_100k_2Replay_0.001_Thr_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_23 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 1000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.001 --bonus-replay --bonus-replay-size 100000 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_1k_0.5_Bandit_0.01_Scaler_100k_2Replay_0.001_Thr_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_39 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 1000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.001 --bonus-replay --bonus-replay-size 100000 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_1k_0.5_Bandit_0.001_Scaler_100k_2Replay_0.001_Thr_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_55 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 1000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.001 --bonus-replay --bonus-replay-size 100000 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --gpu

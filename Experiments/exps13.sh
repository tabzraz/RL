python3 ../Main.py --name Thin_Maze_8_Neg_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_100k_CEps_0.9999_Decay_1k_2Replay_0.9_Thr_Count_12_Stle_100k_Beta_0.01_Eps_0.05_0.05_k_uid_12 --env Thin-Maze-8-Neg-v0 --lr 0.0001 --seed 7 --t-max 600000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --bonus-replay-threshold 0.9 --bonus-replay --bonus-replay-size 1000 --count --beta 0.01 --cts-size 12 --stale-limit 100000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu
python3 ../Main.py --name Thin_Maze_8_Neg_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_100k_CEps_0.9999_Decay_10k_2Replay_0.9_Thr_Count_12_Stle_100k_Beta_0.01_Eps_0.05_0.05_k_uid_28 --env Thin-Maze-8-Neg-v0 --lr 0.0001 --seed 7 --t-max 600000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --bonus-replay-threshold 0.9 --bonus-replay --bonus-replay-size 10000 --count --beta 0.01 --cts-size 12 --stale-limit 100000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu
python3 ../Main.py --name Thin_Maze_8_Neg_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_100k_CEps_0.9999_Decay_25k_2Replay_0.9_Thr_Count_12_Stle_100k_Beta_0.01_Eps_0.05_0.05_k_uid_44 --env Thin-Maze-8-Neg-v0 --lr 0.0001 --seed 7 --t-max 600000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --bonus-replay-threshold 0.9 --bonus-replay --bonus-replay-size 25000 --count --beta 0.01 --cts-size 12 --stale-limit 100000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu

python3 ../Main.py --name Thin_Maze_8_Neg_100_stp_LR_0.0001_Gamma_0.9999_Batch_32_itrs_1_Xp_50k_FC_CEps_0.9999_Decay_Count_12_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_12 --env Thin-Maze-8-Neg-v0 --lr 0.0001 --seed 7 --t-max 400000 --eps-start 0.05 --batch-size 32 --xp 50000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 100 --iters 1 --model Thin-Maze-8-Neg-v0-FC --count --beta 0.0001 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu

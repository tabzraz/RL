python3 ../Main.py --name Thin_Maze_20_100_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_700k_Big_CEps_0.9999_Decay_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_15 --env Thin-Maze-20-v0 --lr 0.0001 --seed 28 --t-max 2000000 --eps-start 0.05 --batch-size 32 --xp 700000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-20-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --ucb --gpu
python3 ../Main.py --name Thin_Maze_22_100_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_700k_Big_CEps_0.9999_Decay_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_31 --env Thin-Maze-22-v0 --lr 0.0001 --seed 28 --t-max 2000000 --eps-start 0.05 --batch-size 32 --xp 700000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-22-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --ucb --gpu

python3 ../Main.py --name Thin_Maze_8_100_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_10k_Big_TABULAR_Count_10_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_3 --env Thin-Maze-8-v0 --lr 0.0001 --seed 28 --t-max 100000 --eps-start 0.05 --batch-size 32 --xp 10000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --tabular --model Thin-Maze-8-v0-Big --count --beta 0.0001 --cts-size 10 --stale-limit 1000000 --gpu

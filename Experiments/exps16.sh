python3 ../Main.py --name Thin_Maze_16_100_Step_1.0_Mix_LR_0.0001_Gamma_0.99_Batch_32_Iters_1_XpSize_300k_Big_BonusClip_-1_CountEps_0.9999_Decay_Count_Cts_21_Stale_1000k_Beta_0.01_Eps_0.05_0.05_uid_15 --env Thin-Maze-16-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 500 --logdir ../Logs --gamma 0.99 --eps-steps 50000 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-16-v0-Big --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu --tar

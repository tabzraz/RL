python3 ../Main.py --name Thin_Maze_18_100_Step_1.0_Mix_LR_0.0001_Gamma_0.9999_Batch_32_Iters_1_XpSize_500k_Big_Prioritized_0.5_Alpha_1_NScaler_BonusClip_-1_0.2_SoftBuffer_CountEps_0.9999_Decay_Count_Cts_21_Stale_10k_Beta_0.0001_Eps_0.05_0.05_uid_1 --env Thin-Maze-18-v0 --lr 0.0001 --seed 14 --t-max 2000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 50000 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-18-v0-Big --priority --negative-td-scaler 1 --alpha 0.5 --soft-buffer 0.2 --count --beta 0.0001 --cts-size 21 --stale-limit 10000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu --tar

python3 ../Main.py --name Thin_Maze_14_100_Step_1.0_Mix_LR_0.0001_Gamma_0.9999_Batch_32_Iters_1_XpSize_100k_Big_Prioritized_0.5_Alpha_1_NScaler_BonusClip_-1_CountEps_0.9999_Decay_OptimisticAction_0.1_Scaler_Count_Cts_21_Stale_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_2 --env Thin-Maze-14-v0 --lr 0.0001 --seed 21 --t-max 800000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-14-v0-Big --priority --negative-td-scaler 1 --alpha 0.5 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --optimistic-init --optimistic-scaler 0.1 --gpu --tar

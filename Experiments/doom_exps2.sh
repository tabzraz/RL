python3 ../Main.py --name DoomMaze_100_Step_1.0_Mix_LR_0.0001_Gamma_0.99_Batch_32_Iters_1_XpSize_500k_Prioritized_0.5_Alpha_0.5_NScaler_BonusClip_-1_CountEps_0.9_Decay_Count_Cts_21_Stale_500k_Beta_0.0001_Eps_0.05_0.05_uid_1 --env DoomMaze-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 50000 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --priority --negative-td-scaler 0.5 --alpha 0.5 --count --beta 0.0001 --cts-size 21 --stale-limit 500000 --count-epsilon --epsilon-decay --decay-rate 0.9 --exp-bonus-save 0.999999 --gpu --tar

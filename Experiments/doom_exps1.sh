python3 ../Main.py --name DoomMaze_100_Step_1.0_Mix_LR_0.0001_Gamma_0.99_Batch_32_Iters_1_XpSize_500k_BonusClip_-1_OptimisticAction_1e-06_Scaler_Count_Cts_21_Stale_500k_Beta_0.0001_Eps_0.05_0.05_uid_0 --env DoomMaze-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 50000 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --count --beta 0.0001 --cts-size 21 --stale-limit 500000 --optimistic-init --optimistic-scaler 1e-06 --exp-bonus-save 1.0 --gpu --tar

python3 ../Main.py --name Thin_Maze_14_100_Step_1.0_Mix_LR_0.0001_Gamma_0.9999_Batch_32_Iters_1_XpSize_100k_Big_BonusClip_-1_ForceAction_1_FCount_Count_Cts_21_Stale_1000k_Beta_0.0001_Eps_0.05_0.05_50k_uid_6 --env Thin-Maze-14-v0 --lr 0.0001 --seed 21 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 50000 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-14-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --force-low-count-action --min-action-count 1 --gpu --tar
python3 ../Main.py --name Thin_Maze_14_100_Step_1.0_Mix_LR_0.0001_Gamma_0.9999_Batch_32_Iters_1_XpSize_100k_Big_BonusClip_-1_OptimisticAction_-0.001_Scaler_Count_Cts_21_Stale_1000k_Beta_0.0001_Eps_0.05_0.05_50k_uid_22 --env Thin-Maze-14-v0 --lr 0.0001 --seed 21 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 50000 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-14-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler -0.001 --gpu --tar

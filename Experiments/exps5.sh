python3 ../Main.py --name Med_Maze_10_100_Step_1.0_Mix_LR_0.0001_Gamma_0.9999_Batch_32_Iters_1_XpSize_300k_Prioritized_MinusPseudo_BonusClip_0_CountEps_0.9999_Decay_Count_Cts_20_Stale_300k_Beta_0.0001_Eps_0.1_1e-05_uid_4 --env Med-Maze-10-v0 --lr 0.0001 --seed 11 --t-max 800000 --eps-start 0.1 --batch-size 32 --xp 300000 --epsilon-finish 1e-05 --logdir ../Logs --gamma 0.9999 --eps-steps 50000 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip 0 --priority --count-td-priority --count --beta 0.0001 --cts-size 20 --stale-limit 300000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu --tar

python3 ../Main.py --name Med_Maze_12_100_Step_LR_0.0001_Gamma_0.9999_Batch_32_Iters_1_XpSize_300k_OptimisticAction_10_Scaler_Count_Cts_20_Stale_300k_Beta_0.0001_Eps_0.1_uid_10 --env Med-Maze-12-v0 --lr 0.0001 --seed 11 --t-max 600000 --eps-start 0.1 --batch-size 32 --xp 300000 --logdir ../Logs --gamma 0.9999 --eps-steps 200000 --n-step 100 --iters 1 --count --beta 0.0001 --cts-size 20 --stale-limit 300000 --optimistic-init --optimistic-scaler 10 --gpu --tar
python3 ../Main.py --name Med_Maze_12_100_Step_LR_0.0001_Gamma_0.9999_Batch_32_Iters_1_XpSize_300k_CountEps_0.9999_Decay_Count_Cts_20_Stale_300k_Beta_0.0001_Eps_0.1_uid_2 --env Med-Maze-12-v0 --lr 0.0001 --seed 33 --t-max 600000 --eps-start 0.1 --batch-size 32 --xp 300000 --logdir ../Logs --gamma 0.9999 --eps-steps 200000 --n-step 100 --iters 1 --count --beta 0.0001 --cts-size 20 --stale-limit 300000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu --tar

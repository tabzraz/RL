python3 Main.py --name Med_Maze_10_100_Step_LR_0.0001_Gamma_0.9999_Batch_32_XpSize_300k_CountEps_0.9999_Decay_Count_Cts_20_Stale_k_Beta_0.0001_Eps_0.1_uid_19 --env Med-Maze-10-v0 --lr 0.0001 --seed 55 --t-max 600000 --eps-start 0.1 --batch-size 32 --xp 300000 --gamma 0.9999 --eps-steps 200000 --n-step 100 --count --beta 0.0001 --cts-size 20 --stale-limit 0 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu --tar
python3 Main.py --name Med_Maze_10_100_Step_LR_0.0001_Gamma_0.9999_Batch_32_XpSize_300k_Prioritized_CountEps_0.9999_Decay_Count_Cts_20_Stale_k_Beta_0.0001_Eps_0.1_uid_19 --env Med-Maze-10-v0 --lr 0.0001 --seed 55 --t-max 600000 --eps-start 0.1 --batch-size 32 --xp 300000 --gamma 0.9999 --eps-steps 200000 --n-step 100 --priority --count --beta 0.0001 --cts-size 20 --stale-limit 0 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu --tar

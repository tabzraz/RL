python3 ../Main.py --name Thin_Maze_19_100_stp_LR_0.0001_Gamma_0.9999_Batch_32_itrs_1_Xp_100k_Big_CEps_0.9999_Decay_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_5 --env Thin-Maze-19-v0 --lr 0.0001 --seed 42 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-19-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu
python3 ../Main.py --name Thin_Maze_19_100_stp_LR_0.0001_Gamma_0.9999_Batch_32_itrs_1_Xp_500k_Big_CEps_0.9999_Decay_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_21 --env Thin-Maze-19-v0 --lr 0.0001 --seed 42 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 100 --n-step-mixing 1.0 --iters 1 --bonus-clip -1 --model Thin-Maze-19-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu

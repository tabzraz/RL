python3 ../Main.py --name Thin_Maze_8_Neg_100_Step_LR_0.0001_Gamma_0.9999_Batch_32_Iters_1_Xp_100k_CEps_0.9999_Decay_Count_12_Stle_1000k_Beta_0.001_NEC_20k_DND_16_Embed_0.1_DLR_10_Neigh_10_Update_Eps_0.05_0.05_k_uid_2 --env Thin-Maze-8-Neg-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 100 --iters 1 --nec --dnd-size 20000 --nec-embedding 16 --nec-alpha 0.1 --nec-neighbours 10 --nec-update 10 --tb-interval 10 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu
python3 ../Main.py --name Thin_Maze_8_Neg_100_Step_LR_0.0001_Gamma_0.99_Batch_32_Iters_1_Xp_100k_CEps_0.9999_Decay_Count_12_Stle_1000k_Beta_0.001_NEC_20k_DND_16_Embed_0.1_DLR_10_Neigh_10_Update_Eps_0.05_0.05_k_uid_10 --env Thin-Maze-8-Neg-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --iters 1 --nec --dnd-size 20000 --nec-embedding 16 --nec-alpha 0.1 --nec-neighbours 10 --nec-update 10 --tb-interval 10 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --gpu

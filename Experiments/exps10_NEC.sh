python3 ../Main.py --name Thin_Maze_8_10_Step_LR_0.0001_Gamma_0.99_Batch_32_Iters_1_Xp_100k_CEps_0.999_Decay_Count_12_Stle_1000k_Beta_0.001_NEC_20k_DND_4_Embed_0.1_DLR_10_Neigh_10_Update_Eps_0.05_0.05_k_uid_9 --env Thin-Maze-8-v0 --lr 0.0001 --seed 14 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --nec --dnd-size 20000 --nec-embedding 4 --nec-alpha 0.1 --nec-neighbours 10 --nec-update 10 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.999 --gpu

python3 ../Main.py --name Thin_Maze_6_100_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_50k_Big_GoalDQN_5000_I_0.9_Thr_1000_itrs_1000_mx_CEps_0.999_Decay_Count_6_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_8 --env Thin-Maze-6-v0 --lr 0.0001 --seed 7 --t-max 500000 --eps-start 0.05 --batch-size 32 --xp 50000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --iters 1 --model Thin-Maze-6-v0-Big --count --beta 0.0001 --cts-size 6 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.999 --goal-dqn --goal-state-interval 5000 --goal-state-threshold 0.9 --goal-iters 1000 --max-option-steps 1000 --gpu
python3 ../Main.py --name Thin_Maze_6_100_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_50k_Big_GoalDQN_10000_I_0.9_Thr_1000_itrs_1000_mx_CEps_0.999_Decay_Count_6_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_24 --env Thin-Maze-6-v0 --lr 0.0001 --seed 7 --t-max 500000 --eps-start 0.05 --batch-size 32 --xp 50000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 100 --iters 1 --model Thin-Maze-6-v0-Big --count --beta 0.0001 --cts-size 6 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.999 --goal-dqn --goal-state-interval 10000 --goal-state-threshold 0.9 --goal-iters 1000 --max-option-steps 1000 --gpu

python3 ../Main.py --name Thin_Maze_14_100_stp_LR_0.0001_Gamma_0.9999_Batch_32_itrs_1_Xp_100k_Big_GoalDQN_50000_I_0.75_Thr_10000_itrs_1000_mx_CEps_0.9999_Decay_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_4 --env Thin-Maze-14-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 100 --iters 1 --model Thin-Maze-14-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --goal-dqn --goal-state-interval 50000 --goal-state-threshold 0.75 --goal-iters 10000 --max-option-steps 1000 --gpu
python3 ../Main.py --name Thin_Maze_14_100_stp_LR_0.0001_Gamma_0.9999_Batch_32_itrs_1_Xp_100k_Big_GoalDQN_100000_I_0.75_Thr_10000_itrs_1000_mx_CEps_0.9999_Decay_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_20 --env Thin-Maze-14-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 100000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.9999 --eps-steps 1 --n-step 100 --iters 1 --model Thin-Maze-14-v0-Big --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --goal-dqn --goal-state-interval 100000 --goal-state-threshold 0.75 --goal-iters 10000 --max-option-steps 1000 --gpu

python3 ../Main.py --name Empty_Room_12_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_0 --env Empty-Room-12-v0 --lr 0.0001 --seed 7 --t-max 600000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --gpu
python3 ../Main.py --name Empty_Room_12_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_8 --env Empty-Room-12-v0 --lr 0.0001 --seed 7 --t-max 600000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --gpu
python3 ../Main.py --name Empty_Room_12_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_400k_uid_16 --env Empty-Room-12-v0 --lr 0.0001 --seed 7 --t-max 600000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs --gamma 0.99 --eps-steps 400000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --gpu

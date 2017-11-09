
ssh -t brown "cd /data/brown/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_10k_2Replay_0.1_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_1_0.05_k_uid_0 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.1 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_10k_2Replay_0.2_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_1_0.05_k_uid_3 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.2 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_1__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_10k_2Replay_0.1_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_1_0.05_k_uid_1 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.1 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_10k_2Replay_0.1_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_4 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.1 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_2__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_10k_2Replay_0.2_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_1_0.05_k_uid_2 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.2 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_3__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'sleep 1; screen -mdS 1_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash server_exps_1__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 2; screen -mdS 2_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash server_exps_2__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 3; screen -mdS 3_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash server_exps_3__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 20; screen -mdS DUDSCREEN_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"sleep 120\"
' > run_server_experiments__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh"

echo 'Written experiments to brown'
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'Started Running Experiments on brown'\""

echo 'Started running 5 exps on brown with 3 screens'


ssh -t savitar "cd /data/savitar/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_10k_2Replay_0.1_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_5 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.1 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_10k_2Replay_0.1_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_13 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.1 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_1__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_10k_2Replay_0.2_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_6 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.2 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_10k_2Replay_0.2_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_14 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.2 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_2__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_10k_2Replay_0.2_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_7 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.2 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_10k_2Replay_0.2_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_15 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.2 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_3__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_10k_2Replay_0.1_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_8 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.1 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_4__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_10k_2Replay_0.1_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_9 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.1 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_5__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_10k_2Replay_0.2_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_10 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.2 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_6__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_10k_2Replay_0.2_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_11 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.2 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_7__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'python3 ../Main.py --name DoomMazeHard_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_10k_2Replay_0.1_Thr_Count_21_Stle_1000k_Beta_0.0001_Eps_0.05_0.05_k_uid_12 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --bonus-replay-threshold 0.1 --bonus-replay --bonus-replay-size 10000 --count --beta 0.0001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_8__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'sleep 1; screen -mdS 1_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_1__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 2; screen -mdS 2_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_2__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 3; screen -mdS 3_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash server_exps_3__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 4; screen -mdS 4_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_4__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 5; screen -mdS 5_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_5__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 6; screen -mdS 6_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_6__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 7; screen -mdS 7_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash server_exps_7__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 8; screen -mdS 8_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_8__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh\"
sleep 20; screen -mdS DUDSCREEN_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 bash -c \"sleep 120\"
' > run_server_experiments__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh"

echo 'Written experiments to savitar'
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments__Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09.sh;echo 'Started Running Experiments on savitar'\""

echo 'Started running 12 exps on savitar with 8 screens'



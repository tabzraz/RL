
ssh -t savitar "cd /data/savitar/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name DoomMazeHard_25_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_Count_21_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_0 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_25_Stp__2017_11_02/ --gamma 0.99 --eps-steps 1 --n-step 25 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_1.sh;echo 'python3 ../Main.py --name DoomMazeHard_25_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_Count_21_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_1 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_25_Stp__2017_11_02/ --gamma 0.99 --eps-steps 1 --n-step 25 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_2.sh;echo 'python3 ../Main.py --name DoomMazeHard_25_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_2 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_25_Stp__2017_11_02/ --gamma 0.99 --eps-steps 1 --n-step 25 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_3.sh;echo 'sleep 1; screen -mdS 1_Exps_DoomMazeHard_25_Stp__2017_11_02 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash server_exps_1.sh\"
sleep 2; screen -mdS 2_Exps_DoomMazeHard_25_Stp__2017_11_02 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_2.sh\"
sleep 3; screen -mdS 3_Exps_DoomMazeHard_25_Stp__2017_11_02 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_3.sh\"
sleep 20; screen -mdS DUDSCREEN_DoomMazeHard_25_Stp__2017_11_02 bash -c \"sleep 120\"
' > run_server_experiments.sh"

echo 'Written experiments to savitar'
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments.sh;echo 'Started Running Experiments on savitar'\""

echo 'Started running 3 exps on savitar with 3 screens'


ssh -t brown "cd /data/brown/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name DoomMazeHard_25_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_3 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_25_Stp__2017_11_02/ --gamma 0.99 --eps-steps 1 --n-step 25 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_1.sh;echo 'python3 ../Main.py --name DoomMazeHard_25_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_4 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_25_Stp__2017_11_02/ --gamma 0.99 --eps-steps 1 --n-step 25 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_2.sh;echo 'python3 ../Main.py --name DoomMazeHard_25_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_5 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_25_Stp__2017_11_02/ --gamma 0.99 --eps-steps 1 --n-step 25 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_3.sh;echo 'sleep 1; screen -mdS 1_Exps_DoomMazeHard_25_Stp__2017_11_02 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash server_exps_1.sh\"
sleep 2; screen -mdS 2_Exps_DoomMazeHard_25_Stp__2017_11_02 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_2.sh\"
sleep 3; screen -mdS 3_Exps_DoomMazeHard_25_Stp__2017_11_02 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash server_exps_3.sh\"
sleep 20; screen -mdS DUDSCREEN_DoomMazeHard_25_Stp__2017_11_02 bash -c \"sleep 120\"
' > run_server_experiments.sh"

echo 'Written experiments to brown'
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments.sh;echo 'Started Running Experiments on brown'\""

echo 'Started running 3 exps on brown with 3 screens'



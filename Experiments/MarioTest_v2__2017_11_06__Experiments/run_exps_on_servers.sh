
ssh -t savitar "cd /data/savitar/tabhid;cd Server_Exps;echo 'xvfb-run-safe -s \"-screen 0 1400x900x24\" python3 ../Main.py --name Mario_1_1_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9999_Decay_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_0 --env Mario-1-1-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/MarioTest_v2__2017_11_06/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_1__MarioTest_v2__2017_11_06.sh;echo 'xvfb-run-safe -s \"-screen 0 1400x900x24\" python3 ../Main.py --name Mario_1_1_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9999_Decay_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_1 --env Mario-1-1-v0 --lr 0.0001 --seed 14 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/MarioTest_v2__2017_11_06/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9999 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_2__MarioTest_v2__2017_11_06.sh;echo 'xvfb-run-safe -s \"-screen 0 1400x900x24\" python3 ../Main.py --name Mario_1_1_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_2 --env Mario-1-1-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/MarioTest_v2__2017_11_06/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_3__MarioTest_v2__2017_11_06.sh;echo 'xvfb-run-safe -s \"-screen 0 1400x900x24\" python3 ../Main.py --name Mario_1_1_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_3 --env Mario-1-1-v0 --lr 0.0001 --seed 14 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/MarioTest_v2__2017_11_06/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_4__MarioTest_v2__2017_11_06.sh;echo 'xvfb-run-safe -s \"-screen 0 1400x900x24\" python3 ../Main.py --name Mario_1_1_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_4 --env Mario-1-1-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/MarioTest_v2__2017_11_06/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_5__MarioTest_v2__2017_11_06.sh;echo 'xvfb-run-safe -s \"-screen 0 1400x900x24\" python3 ../Main.py --name Mario_1_1_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_12_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_5 --env Mario-1-1-v0 --lr 0.0001 --seed 14 --t-max 1200000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/MarioTest_v2__2017_11_06/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_6__MarioTest_v2__2017_11_06.sh;echo 'xvfb-run-safe -s \"-screen 0 1400x900x24\" ' > server_exps_7__MarioTest_v2__2017_11_06.sh;echo 'sleep 1; screen -mdS 1_Exps_MarioTest_v2__2017_11_06 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash server_exps_1__MarioTest_v2__2017_11_06.sh\"
sleep 2; screen -mdS 2_Exps_MarioTest_v2__2017_11_06 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_2__MarioTest_v2__2017_11_06.sh\"
sleep 3; screen -mdS 3_Exps_MarioTest_v2__2017_11_06 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_3__MarioTest_v2__2017_11_06.sh\"
sleep 4; screen -mdS 4_Exps_MarioTest_v2__2017_11_06 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash server_exps_4__MarioTest_v2__2017_11_06.sh\"
sleep 5; screen -mdS 5_Exps_MarioTest_v2__2017_11_06 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash server_exps_5__MarioTest_v2__2017_11_06.sh\"
sleep 6; screen -mdS 6_Exps_MarioTest_v2__2017_11_06 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash server_exps_6__MarioTest_v2__2017_11_06.sh\"
sleep 7; screen -mdS 7_Exps_MarioTest_v2__2017_11_06 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_7__MarioTest_v2__2017_11_06.sh\"
sleep 20; screen -mdS DUDSCREEN_MarioTest_v2__2017_11_06 bash -c \"sleep 120\"
' > run_server_experiments__MarioTest_v2__2017_11_06.sh"

echo 'Written experiments to savitar'
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments__MarioTest_v2__2017_11_06.sh;echo 'Started Running Experiments on savitar'\""

echo 'Started running 6 exps on savitar with 7 screens'


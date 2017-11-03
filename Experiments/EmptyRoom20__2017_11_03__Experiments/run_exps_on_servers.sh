
ssh -t dgx1 "cd /data/dgx1/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Count_20_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_0 --env Empty-Room-20-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_1.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Count_20_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_1 --env Empty-Room-20-v0 --lr 0.0001 --seed 14 --t-max 300000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_2.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_1_Scaler_Count_20_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_2 --env Empty-Room-20-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --optimistic-init --optimistic-scaler 1 --bandit-p 0.5 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_3.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_1_Scaler_Count_20_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_3 --env Empty-Room-20-v0 --lr 0.0001 --seed 14 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --optimistic-init --optimistic-scaler 1 --bandit-p 0.5 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_4.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_Count_20_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_4 --env Empty-Room-20-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_5.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_Count_20_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_5 --env Empty-Room-20-v0 --lr 0.0001 --seed 14 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_6.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_20_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_6 --env Empty-Room-20-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_7.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_20_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_7 --env Empty-Room-20-v0 --lr 0.0001 --seed 14 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_8.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_20_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_8 --env Empty-Room-20-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_9.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_20_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_9 --env Empty-Room-20-v0 --lr 0.0001 --seed 14 --t-max 300000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_10.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Count_20_Stle_1000k_Beta_0.001_Eps_1_0.05_100k_uid_10 --env Empty-Room-20-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 100000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_11.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Count_20_Stle_1000k_Beta_0.001_Eps_1_0.05_100k_uid_11 --env Empty-Room-20-v0 --lr 0.0001 --seed 14 --t-max 300000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 100000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_12.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Count_20_Stle_1000k_Beta_0.001_Eps_1_0.05_200k_uid_12 --env Empty-Room-20-v0 --lr 0.0001 --seed 7 --t-max 300000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 200000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_13.sh;echo 'python3 ../Main.py --name Empty_Room_20_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_Count_20_Stle_1000k_Beta_0.001_Eps_1_0.05_200k_uid_13 --env Empty-Room-20-v0 --lr 0.0001 --seed 14 --t-max 300000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/EmptyRoom20__2017_11_03/ --gamma 0.99 --eps-steps 200000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 20 --stale-limit 1000000 --eval-interval 2 --interval-size 1000 --frontier-interval 1000 --exploration-steps 0 --gpu
' > server_exps_14.sh;echo '' > server_exps_15.sh;echo '' > server_exps_16.sh;echo 'sleep 1; screen -mdS 1_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash server_exps_1.sh\"
sleep 2; screen -mdS 2_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_2.sh\"
sleep 3; screen -mdS 3_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_3.sh\"
sleep 4; screen -mdS 4_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash server_exps_4.sh\"
sleep 5; screen -mdS 5_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash server_exps_5.sh\"
sleep 6; screen -mdS 6_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash server_exps_6.sh\"
sleep 7; screen -mdS 7_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash server_exps_7.sh\"
sleep 8; screen -mdS 8_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_8.sh\"
sleep 9; screen -mdS 9_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash server_exps_9.sh\"
sleep 10; screen -mdS 10_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_10.sh\"
sleep 11; screen -mdS 11_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_11.sh\"
sleep 12; screen -mdS 12_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash server_exps_12.sh\"
sleep 13; screen -mdS 13_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash server_exps_13.sh\"
sleep 14; screen -mdS 14_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash server_exps_14.sh\"
sleep 15; screen -mdS 15_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash server_exps_15.sh\"
sleep 16; screen -mdS 16_Exps_EmptyRoom20__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_16.sh\"
sleep 20; screen -mdS DUDSCREEN_EmptyRoom20__2017_11_03 bash -c \"sleep 120\"
' > run_server_experiments.sh"

echo 'Written experiments to dgx1'
ssh -t dgx1 "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments.sh;echo 'Started Running Experiments on dgx1'\""

echo 'Started running 14 exps on dgx1 with 16 screens'



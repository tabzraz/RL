
ssh -t dgx1 "cd /data/dgx1/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name DoomMazeHard_5_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_Count_21_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_0 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 5 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_Count_21_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_8 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_1.sh;echo 'python3 ../Main.py --name DoomMazeHard_5_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_Count_21_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_1 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 5 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_CEps_0.9_Decay_Count_21_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_9 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.9 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_2.sh;echo 'python3 ../Main.py --name DoomMazeHard_5_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_2 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 5 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_10 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_3.sh;echo 'python3 ../Main.py --name DoomMazeHard_5_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_3 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 5 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_11 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_4.sh;echo 'python3 ../Main.py --name DoomMazeHard_5_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_4 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 5 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_12 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_5.sh;echo 'python3 ../Main.py --name DoomMazeHard_5_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_5 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 5 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_13 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_6.sh;echo 'python3 ../Main.py --name DoomMazeHard_5_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_6 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 5 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_14 --env DoomMazeHard-v0 --lr 0.0001 --seed 7 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_7.sh;echo 'python3 ../Main.py --name DoomMazeHard_5_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_7 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 5 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name DoomMazeHard_10_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_0.5_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_15 --env DoomMazeHard-v0 --lr 0.0001 --seed 14 --t-max 3000000 --eps-start 0.05 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/DoomMazeHard_5_10_Step__2017_10_31/ --gamma 0.99 --eps-steps 1 --n-step 10 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_8.sh;echo 'screen -mdS 1_Exps_2017-10-31_00-54 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash server_exps_1.sh\"
screen -mdS 2_Exps_2017-10-31_00-54 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_2.sh\"
screen -mdS 3_Exps_2017-10-31_00-54 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_3.sh\"
screen -mdS 4_Exps_2017-10-31_00-54 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash server_exps_4.sh\"
screen -mdS 5_Exps_2017-10-31_00-54 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash server_exps_5.sh\"
screen -mdS 6_Exps_2017-10-31_00-54 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash server_exps_6.sh\"
screen -mdS 7_Exps_2017-10-31_00-54 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash server_exps_7.sh\"
screen -mdS 8_Exps_2017-10-31_00-54 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_8.sh\"
' > run_server_experiments.sh"

echo 'Written experiments to dgx1'
ssh -t dgx1 "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments.sh;echo 'Started Running Experiments on dgx1'\""

echo 'Started running 16 exps on dgx1'



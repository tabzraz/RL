
ssh -t brown "cd /data/brown/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_CEps_0.999_Decay_Count_21_Stle_1000k_Beta_0.01_Eps_1_0.05_k_uid_0 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 7 --t-max 20000000 --eps-start 1 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.999 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_CEps_0.999_Decay_Count_21_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_2 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 7 --t-max 20000000 --eps-start 1 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.999 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_1__Montezuma_20mil__2017_11_10.sh;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_CEps_0.999_Decay_Count_21_Stle_1000k_Beta_0.01_Eps_1_0.05_k_uid_1 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 14 --t-max 20000000 --eps-start 1 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.999 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_CEps_0.999_Decay_Count_21_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_3 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 14 --t-max 20000000 --eps-start 1 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --count-epsilon --epsilon-decay --decay-rate 0.999 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_2__Montezuma_20mil__2017_11_10.sh;echo 'sleep 1; screen -mdS 1_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash server_exps_1__Montezuma_20mil__2017_11_10.sh\"
sleep 2; screen -mdS 2_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash server_exps_2__Montezuma_20mil__2017_11_10.sh\"
sleep 20; screen -mdS DUDSCREEN_Montezuma_20mil__2017_11_10 bash -c \"sleep 120\"
' > run_server_experiments__Montezuma_20mil__2017_11_10.sh"

echo 'Written experiments to brown'
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments__Montezuma_20mil__2017_11_10.sh;echo 'Started Running Experiments on brown'\""

echo 'Started running 4 exps on brown with 2 screens'


ssh -t dgx1 "cd /data/dgx1/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.01_Eps_0.05_0.05_k_uid_4 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 7 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.01_Eps_0.05_0.05_k_uid_8 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 7 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_1__Montezuma_20mil__2017_11_10.sh;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.01_Eps_0.05_0.05_k_uid_5 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 14 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.01_Eps_0.05_0.05_k_uid_9 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 14 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_2__Montezuma_20mil__2017_11_10.sh;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_6 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 7 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_10 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 7 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_3__Montezuma_20mil__2017_11_10.sh;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.1_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_7 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 14 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.1 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_4__Montezuma_20mil__2017_11_10.sh;echo 'sleep 1; screen -mdS 1_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash server_exps_1__Montezuma_20mil__2017_11_10.sh\"
sleep 2; screen -mdS 2_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash server_exps_2__Montezuma_20mil__2017_11_10.sh\"
sleep 3; screen -mdS 3_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash server_exps_3__Montezuma_20mil__2017_11_10.sh\"
sleep 4; screen -mdS 4_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_4__Montezuma_20mil__2017_11_10.sh\"
sleep 20; screen -mdS DUDSCREEN_Montezuma_20mil__2017_11_10 bash -c \"sleep 120\"
' > run_server_experiments__Montezuma_20mil__2017_11_10.sh"

echo 'Written experiments to dgx1'
ssh -t dgx1 "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments__Montezuma_20mil__2017_11_10.sh;echo 'Started Running Experiments on dgx1'\""

echo 'Started running 7 exps on dgx1 with 4 screens'


ssh -t savitar "cd /data/savitar/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.01_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_11 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 14 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.01 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_15 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 14 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_1__Montezuma_20mil__2017_11_10.sh;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.01_Eps_0.05_0.05_k_uid_12 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 7 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_2__Montezuma_20mil__2017_11_10.sh;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.01_Eps_0.05_0.05_k_uid_13 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 14 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.01 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_3__Montezuma_20mil__2017_11_10.sh;echo 'python3 ../Main.py --name Wrapped_MontezumaRevenge_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_500k_0.5_Bandit_0.001_Scaler_Count_21_Stle_1000k_Beta_0.001_Eps_0.05_0.05_k_uid_14 --env Wrapped_MontezumaRevenge-v0 --lr 0.0001 --seed 7 --t-max 20000000 --eps-start 0.05 --batch-size 32 --xp 500000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Montezuma_20mil__2017_11_10/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 21 --stale-limit 1000000 --optimistic-init --optimistic-scaler 0.001 --bandit-p 0.5 --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --tb-interval 100 --gpu
' > server_exps_4__Montezuma_20mil__2017_11_10.sh;echo 'sleep 1; screen -mdS 1_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_1__Montezuma_20mil__2017_11_10.sh\"
sleep 2; screen -mdS 2_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_2__Montezuma_20mil__2017_11_10.sh\"
sleep 3; screen -mdS 3_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash server_exps_3__Montezuma_20mil__2017_11_10.sh\"
sleep 4; screen -mdS 4_Exps_Montezuma_20mil__2017_11_10 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_4__Montezuma_20mil__2017_11_10.sh\"
sleep 20; screen -mdS DUDSCREEN_Montezuma_20mil__2017_11_10 bash -c \"sleep 120\"
' > run_server_experiments__Montezuma_20mil__2017_11_10.sh"

echo 'Written experiments to savitar'
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments__Montezuma_20mil__2017_11_10.sh;echo 'Started Running Experiments on savitar'\""

echo 'Started running 7 exps on savitar with 4 screens'


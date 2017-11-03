
ssh -t brown "cd /data/brown/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_0 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_300k_uid_5 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 14 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 300000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_1.sh;echo 'python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_1 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 14 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_300k_uid_6 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 21 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 300000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_2.sh;echo 'python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_2 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 21 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_300k_uid_7 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 300000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_3.sh;echo 'python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_k_uid_3 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 1 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_600k_uid_8 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 600000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_4.sh;echo 'python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_300k_uid_4 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 300000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_600k_uid_9 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 14 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 600000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_5.sh;echo 'sleep 1; screen -mdS 1_Exps_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash server_exps_1.sh\"
sleep 2; screen -mdS 2_Exps_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash server_exps_2.sh\"
sleep 3; screen -mdS 3_Exps_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash server_exps_3.sh\"
sleep 4; screen -mdS 4_Exps_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash server_exps_4.sh\"
sleep 5; screen -mdS 5_Exps_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash server_exps_5.sh\"
sleep 20; screen -mdS DUDSCREEN_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"sleep 120\"
' > run_server_experiments.sh"

echo 'Written experiments to brown'
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments.sh;echo 'Started Running Experiments on brown'\""

echo 'Started running 10 exps on brown with 5 screens'


ssh -t savitar "cd /data/savitar/tabhid;cd Server_Exps;echo 'python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_600k_uid_10 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 21 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 600000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_900k_uid_13 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 14 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 900000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_1.sh;echo 'python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_600k_uid_11 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 600000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_900k_uid_14 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 21 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 900000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_2.sh;echo 'python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_900k_uid_12 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 7 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 900000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
python3 ../Main.py --name Thin_Maze_12_Neg_1_stp_LR_0.0001_Gamma_0.99_Batch_32_itrs_1_Xp_300k_StateAction_Count_12_Stle_1000k_Beta_0.001_Eps_1_0.05_900k_uid_15 --env Thin-Maze-12-Neg-v0 --lr 0.0001 --seed 28 --t-max 1200000 --eps-start 1 --batch-size 32 --xp 300000 --epsilon-finish 0.05 --target 1000 --logdir ../Logs/Maze12_Epsilon_Schedule__2017_11_03/ --gamma 0.99 --eps-steps 900000 --n-step 1 --iters 1 --count --beta 0.001 --cts-size 12 --stale-limit 1000000 --count-state-action --eval-interval 100 --interval-size 100 --frontier-interval 100 --exploration-steps 500 --gpu
' > server_exps_3.sh;echo 'sleep 1; screen -mdS 1_Exps_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash server_exps_1.sh\"
sleep 2; screen -mdS 2_Exps_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash server_exps_2.sh\"
sleep 3; screen -mdS 3_Exps_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='7' bash server_exps_3.sh\"
sleep 20; screen -mdS DUDSCREEN_Maze12_Epsilon_Schedule__2017_11_03 bash -c \"sleep 120\"
' > run_server_experiments.sh"

echo 'Written experiments to savitar'
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"cd RL;git pull;cd Server_Exps;bash run_server_experiments.sh;echo 'Started Running Experiments on savitar'\""

echo 'Started running 6 exps on savitar with 3 screens'



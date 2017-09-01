screen -mdS 1_Exps bash -c "sleep 10; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash doom_exps1.sh"
screen -mdS 2_Exps bash -c "sleep 20; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash doom_exps2.sh"
screen -mdS 3_Exps bash -c "sleep 30; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash doom_exps3.sh"
screen -mdS 4_Exps bash -c "sleep 40; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash doom_exps4.sh"
screen -mdS 5_Exps bash -c "sleep 50; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash doom_exps5.sh"
screen -mdS 6_Exps bash -c "sleep 60; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash doom_exps6.sh"
screen -mdS 7_Exps bash -c "sleep 70; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash doom_exps7.sh"
CUDA_VISIBLE_DEVICES='7' bash doom_exps8.sh
# 16 Experiments total

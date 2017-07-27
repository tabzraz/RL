screen -mdS 1_Exps bash -c "sleep 10; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash exps1.sh"
screen -mdS 2_Exps bash -c "sleep 20; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash exps2.sh"
screen -mdS 3_Exps bash -c "sleep 30; export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash exps3.sh"
CUDA_VISIBLE_DEVICES='3' bash exps4.sh
# 8 Experiments total

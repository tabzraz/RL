screen -mdS 1_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash exps1.sh"
screen -mdS 2_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash exps2.sh"
screen -mdS 3_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash exps3.sh"
screen -mdS 4_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash exps4.sh"
screen -mdS 5_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash exps5.sh"
screen -mdS 6_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash exps6.sh"
CUDA_VISIBLE_DEVICES='7' bash exps7.sh
# 32 Experiments total

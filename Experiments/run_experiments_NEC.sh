screen -mdS 1_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash exps1_NEC.sh"
screen -mdS 2_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash exps2_NEC.sh"
screen -mdS 3_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash exps3_NEC.sh"
screen -mdS 4_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash exps4_NEC.sh"
screen -mdS 5_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash exps5_NEC.sh"
screen -mdS 6_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash exps6_NEC.sh"
screen -mdS 7_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash exps7_NEC.sh"
CUDA_VISIBLE_DEVICES='7' bash exps8_NEC.sh
# 16 Experiments total

screen -mdS 1_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash exps1.sh"
CUDA_VISIBLE_DEVICES='1' bash exps2.sh
# 2 Experiments total

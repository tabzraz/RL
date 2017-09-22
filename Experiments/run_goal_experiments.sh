screen -mdS 1_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='0' bash exps_goal1.sh"
screen -mdS 2_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='1' bash exps_goal2.sh"
screen -mdS 3_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='2' bash exps_goal3.sh"
screen -mdS 4_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='3' bash exps_goal4.sh"
screen -mdS 5_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='4' bash exps_goal5.sh"
screen -mdS 6_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='5' bash exps_goal6.sh"
screen -mdS 7_Exps bash -c "export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='6' bash exps_goal7.sh"
CUDA_VISIBLE_DEVICES='7' bash exps_goal8.sh
# 16 Experiments total

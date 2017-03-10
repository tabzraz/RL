envs = ["Maze-4-v1"]
lrs = [0.0001]
counts = [True]
betas = [0.01]
t_maxs = [1000000]
cts_sizes = [7]
seeds = [13, 66, 99]
epsilon_starts = [0.1]

gpu = True
debug_eval = True
screen = False

uid = 0
for env in envs:
    for t_max in t_maxs:
        for lr in lrs:
            for eps in epsilon_starts:
                for count in counts:
                    for beta in betas:
                        for cts_size in cts_sizes:
                            for seed in seeds:
                                name = env.replace("-", "_")[:-3]
                                if count:
                                    name += "_Count_Cts_{}_Beta_{}_Eps_{}_uid_{}".format(cts_size, beta, eps, uid)
                                else:
                                    name += "_DQN_uid_{}".format(uid)
                                python_command = "python3 RL_Trainer_PyTorch.py --name {} --env {} --lr {} --seed {} --t-max {} --eps-start {}".format(name, env, lr, seed, t_max, eps)
                                if count:
                                    python_command += " --count --beta {} --cts-size {}".format(beta, cts_size)
                                if gpu:
                                    python_command += " --gpu"
                                if debug_eval:
                                    python_command += " --debug-eval"
                                if screen:
                                    screen_name = "DQN"
                                    if count:
                                        screen_name = "Count"
                                    screen_name += "_{}".format(seed)
                                    python_command = "screen -mdS {}__{} bash -c \"{}\"".format(uid, screen_name, python_command)
                                print(python_command)
                                uid += 1

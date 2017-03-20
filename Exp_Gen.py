envs = ["Maze-{}-v1".format(size) for size in [5, 6, 7, 8, 9, 10]]
lrs = [0.0001]
counts = [False, True]
cts_convs = [True]
betas = [0.01]
t_maxs = [200000]
cts_sizes = [7]
seeds = [13, 66, 75]
epsilon_starts = [1]
batch_sizes = [32]
options = [True]

gpu = True
debug_eval = True
screen = False

uid = 0
for env in envs:
    for t_max in t_maxs:
        for option in options:
            for batch_size in batch_sizes:
                for lr in lrs:
                    for eps in epsilon_starts:
                        for count in counts:
                            for beta in betas:
                                for cts_size in cts_sizes:
                                    for cts_conv in cts_convs:
                                        for seed in seeds:
                                            name = env.replace("-", "_")[:-3]
                                            name += "_Batch_{}".format(batch_size)
                                            if option:
                                                name += "_Options"
                                            if count:
                                                name += "_Count_Cts_{}_Conv_{}_Beta_{}_Eps_{}_uid_{}".format(cts_size, cts_conv, beta, eps, uid)
                                            else:
                                                name += "_DQN_uid_{}".format(uid)
                                            python_command = "python3 RL_Trainer_PyTorch.py --name {} --env {} --lr {} --seed {} --t-max {} --eps-start {} --batch-size {}".format(name, env, lr, seed, t_max, eps, batch_size)
                                            if count:
                                                python_command += " --count --beta {} --cts-size {}".format(beta, cts_size)
                                                if cts_conv:
                                                    python_command += " --cts-conv"
                                            if option:
                                                python_command += " --options"
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
                                        print()

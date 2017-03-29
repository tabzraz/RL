envs = ["Maze-{}-v1".format(size) for size in [5]]
lrs = [0.0001]
counts = [True]
cts_convs = [True, False]
betas = [0.01]
t_maxs = [1000000]
cts_sizes = [7]
seeds = [13, 66, 75]
epsilon_starts = [1, 0.1]
batch_sizes = [32]
xp_replay_sizes = [50000]

options = [False]

num_macros = [1]#[10, 50, 200]
max_macro_lengths = [1]#[8, 16, 32]
macro_seeds = [1]#[5, 6]

gpu = True
debug_eval = True
screen = False
random_macros = False
with_primitives = False
files = 2
write_to_files = True

uid = 0
commands = []
for env in envs:
    for t_max in t_maxs:
        for option in options:
            for batch_size, xp_replay_size in [(b, x) for b in batch_sizes for x in xp_replay_sizes]:
                for lr in lrs:
                    for eps in epsilon_starts:
                        for count in counts:
                            for beta in betas:
                                for cts_size in cts_sizes:
                                    for cts_conv in cts_convs:
                                        for num_macro in num_macros:
                                            for max_macro_length in max_macro_lengths:
                                                for macro_seed in macro_seeds:
                                                    for seed in seeds:
                                                        name = env.replace("-", "_")[:-3]
                                                        name += "_Batch_{}_XpSize_{}k".format(batch_size, str(xp_replay_size)[:-3])
                                                        if option:
                                                            if random_macros:
                                                                name += "_Rnd_Macros_{}_Length_{}_Mseed_{}_Primitives_{}".format(num_macro, max_macro_length, macro_seed, with_primitives)
                                                            else:
                                                                name += "_Options"
                                                        if count:
                                                            name += "_Count_Cts_{}_Conv_{}_Beta_{}_Eps_{}_uid_{}".format(cts_size, cts_conv, beta, eps, uid)
                                                        else:
                                                            name += "_DQN_uid_{}".format(uid)
                                                        python_command = "python3 RL_Trainer_PyTorch.py --name {} --env {} --lr {} --seed {} --t-max {} --eps-start {} --batch-size {} --xp {}".format(name, env, lr, seed, t_max, eps, batch_size, xp_replay_size)
                                                        if count:
                                                            python_command += " --count --beta {} --cts-size {}".format(beta, cts_size)
                                                            if cts_conv:
                                                                python_command += " --cts-conv"
                                                        if option:
                                                            if random_macros:
                                                                python_command += " --options Random_Macros --num-macros {} --max-macro-length {} --macro-seed {}".format(num_macro, max_macro_length, macro_seed)
                                                                if with_primitives:
                                                                    python_command += " --train-primitives"
                                                            else:
                                                                python_command += " --options Maze_Good"
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
                                                        commands.append(python_command)
                                                        uid += 1
                                                    print()

if write_to_files:
    for i in range(files):
        with open("exps{}.sh".format(i + 1), "w") as f:
            for cc in commands[i::files]:
                f.write("{}\n".format(cc))

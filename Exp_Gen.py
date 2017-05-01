envs = ["Wide-Maze-{}-v0".format(size) for size in [7]]
lrs = [0.001]
counts = [True, False]
cts_convs = [False]
betas = [0.00005]
t_maxs = [200000]
cts_sizes = [20]
seeds = [23, 53, 99]  # Randomly generated by me =)
epsilon_starts = [1]
epsilon_steps = [120000]
batch_sizes = [32]
xp_replay_sizes = [50000]
stale_limits = [0, 1]
epsilon_scaling = [False, True]

options = [False]

num_macros = [1]#[10, 50, 200]
max_macro_lengths = [1]#[8, 16, 32]
macro_seeds = [1]#[5, 6]

gpu = True
# debug_eval = True
screen = False
random_macros = False
with_primitives = False
files = 4
write_to_files = True

tar = True

uid = 0
commands = []
for env in envs:
    for t_max in t_maxs:
        for option in options:
            for batch_size, xp_replay_size in [(b, x) for b in batch_sizes for x in xp_replay_sizes]:
                for lr in lrs:
                    for eps, eps_steps in zip(epsilon_starts, epsilon_steps):
                        for count, eps_scaling in [(c, e) for c in counts for e in epsilon_scaling if not (c is False and e is True)]:
                            for stale_val in stale_limits:
                                for beta in betas:
                                    for cts_size in cts_sizes:
                                        for cts_conv in cts_convs:
                                            for seed in seeds:
                                                name = env.replace("-", "_")[:-3]
                                                name += "_LR_{}".format(lr)
                                                name += "_Batch_{}_XpSize_{}k".format(batch_size, str(xp_replay_size)[:-3])
                                                # if option:
                                                #     if random_macros:
                                                #         name += "_Rnd_Macros_{}_Length_{}_Mseed_{}_Primitives_{}".format(num_macro, max_macro_length, macro_seed, with_primitives)
                                                #     else:
                                                #         name += "_Options"
                                                if count:
                                                    stale = int(xp_replay_size * stale_val)
                                                    # print(stale)
                                                    if eps_scaling:
                                                        name += "_CountEps"
                                                    name += "_Count_Cts_{}_Stale_{}k_Conv_{}_Beta_{}_Eps_{}_uid_{}".format(cts_size, str(stale)[:-3], cts_conv, beta, eps, uid)
                                                else:
                                                    name += "_DQN_Eps_{}_uid_{}".format(eps, uid)
                                                python_command = "python3 Main.py --name {} --env {} --lr {} --seed {} --t-max {} --eps-start {} --batch-size {} --xp {}".format(name, env, lr, seed, t_max, eps, batch_size, xp_replay_size)
                                                python_command += " --eps-steps {}".format(eps_steps)
                                                if count:
                                                    python_command += " --count --beta {} --cts-size {}".format(beta, cts_size)
                                                    python_command += " --stale-limit {}".format(stale)
                                                    if eps_scaling:
                                                        python_command += " --count-epsilon"
                                                    if cts_conv:
                                                        python_command += " --cts-conv"
                                                # if option:
                                                #     if random_macros:
                                                #         python_command += " --options Random_Macros --num-macros {} --max-macro-length {} --macro-seed {}".format(num_macro, max_macro_length, macro_seed)
                                                #         if with_primitives:
                                                #             python_command += " --train-primitives"
                                                #     else:
                                                #         python_command += " --options Maze_Good"
                                                if gpu:
                                                    python_command += " --gpu"
                                                # if debug_eval:
                                                    # python_command += " --debug-eval"
                                                if tar:
                                                    python_command += " --tar"
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

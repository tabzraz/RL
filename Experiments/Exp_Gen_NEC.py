import sys
from math import ceil

envs = ["Thin-Maze-{}-v0".format(size) for size in [8]]
target_network = 1000
tb_interval = 10
lrs = [0.0001]
counts = [True]

betas = [0.001, 0.0001]
t_maxs = [x * 1000 for x in [300]]
cts_sizes = [12]
num_seeds = 2
epsilon_starts = [0.05]
epsilon_finishs = [0.05]
epsilon_steps = [1]
batch_sizes = [(32, 1)]
xp_replay_sizes = [x * 1000 for x in [100]]

dnd_sizes = [x * 1000 for x in [5, 20]]
nec_embeddings = [16]
nec_alphas = [0.1]
nec_neighbours = [10]
nec_updates = [10]

stale_limits = [x * 1000 for x in [1000]]
epsilon_scaling = [True]
epsilon_decay = [0.9999]

n_steps = [100]

gpu = True
# debug_eval = True
screen = False
random_macros = False
with_primitives = False
files = 16
# (Prioritised, I.S. correction, Negative td error scaler, Subtract pseudo rewards, alpha)
# prioritizeds = [(True, False, 8, False, 0.5)]  # [(True, False, True), (True, True, True)]
gammas = [0.9999, 0.99]

set_replays = [(False, 1)]
doubles = [False]

seeds = [7 * (i + 1) for i in range(num_seeds)]  # Randomly generated by me =)

big_model = True
tabular = False

write_to_files = False
append = False

if "--write" in sys.argv:
    write_to_files = True
if "--append" in sys.argv:
    append = True

start_at = 0

gpus = 8
exps_per_gpu = 2
files = gpus * exps_per_gpu

gpu_start = 0

tar = False

uid = 0
commands = []
for env in envs:
    for t_max, lr in [(t, l) for t in t_maxs for l in lrs]:
        for n_step, gamma in [(n, g) for n in n_steps for g in gammas]:
            for batch_size, iters, xp_replay_size in [(b, i, x) for (b, i) in batch_sizes for x in xp_replay_sizes]:
                for eps, eps_steps, eps_finish in zip(epsilon_starts, epsilon_steps, epsilon_finishs):
                    for count, eps_scaling in [(c, e) for c in counts for e in epsilon_scaling if not (c is False and e is True)]:
                        for eps_decay in epsilon_decay:
                            for stale_val in stale_limits:
                                for beta in betas:
                                    for cts_size in cts_sizes:
                                        for dnd_size in dnd_sizes:
                                            for nec_embedding in nec_embeddings:
                                                for nec_alpha, nec_neighbour, nec_update in zip(nec_alphas, nec_neighbours, nec_updates):
                                                    for seed in seeds:

                                                        name = env.replace("-", "_")[:-3]
                                                        name += "_{}_Step".format(n_step)
                                                        name += "_LR_{}".format(lr)
                                                        name += "_Gamma_{}".format(gamma)
                                                        name += "_Batch_{}_Iters_{}_Xp_{}k".format(batch_size, iters, str(xp_replay_size)[:-3])
                                                        if count:
                                                            if eps_scaling:
                                                                name += "_CEps_{}_Decay".format(eps_decay)
                                                        if count:
                                                            name += "_Count_{}_Stle_{}k_Beta_{}".format(cts_size, str(stale_val)[:-3], beta)

                                                        name += "_NEC_{}k_DND_{}_Embed_{}_DLR_{}_Neigh_{}_Update".format(str(dnd_size)[:-3], nec_embedding, nec_alpha, nec_neighbour, nec_update)

                                                        name += "_Eps_{}_{}_{}k_uid_{}".format(eps, eps_finish, str(eps_steps)[:-3], uid)

                                                        python_command = "python3 ../Main.py --name {} --env {} --lr {} --seed {} --t-max {} --eps-start {} --batch-size {} --xp {}".format(name, env, lr, seed, t_max, eps, batch_size, xp_replay_size)
                                                        python_command += " --epsilon-finish {}".format(eps_finish)
                                                        python_command += " --logdir ../Logs"
                                                        python_command += " --gamma {}".format(gamma)
                                                        python_command += " --eps-steps {}".format(eps_steps)
                                                        python_command += " --n-step {}".format(n_step)
                                                        python_command += " --iters {}".format(iters)

                                                        python_command += " --nec"
                                                        python_command += " --dnd-size {} --nec-embedding {}".format(dnd_size, nec_embedding)
                                                        python_command += " --nec-alpha {} --nec-neighbours {} --nec-update {}".format(nec_alpha, nec_neighbour, nec_update)

                                                        python_command += " --tb-interval {}".format(tb_interval)

                                                        if count:
                                                            python_command += " --count --beta {} --cts-size {}".format(beta, cts_size)
                                                            python_command += " --stale-limit {}".format(stale_val)
                                                            if eps_scaling:
                                                                python_command += " --count-epsilon"
                                                                python_command += " --epsilon-decay --decay-rate {}".format(eps_decay)
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
                                                        if seed == seeds[0]:
                                                            print(python_command)
                                                        commands.append(python_command)
                                                        uid += 1
                                                    print()

if write_to_files:
    print("\n--WRITING TO FILES--\n")
if write_to_files and append:
    print("--APPENDING--")

print("\n--- {} Runs ---\n--- {} Files => Upto {} Runs per file ---\n".format(uid, files, ceil(uid / files)))
print("--- {} GPUs, {} Concurrent runs per GPU ---\n".format(gpus, exps_per_gpu))

if write_to_files:

    if not append:
        for i in range(files):
            with open("exps{}_NEC.sh".format(i + 1), "w") as f:
                f.write("")
                print("Cleared exps{}_NEC.sh".format(i + 1))

    for i in range(uid):
        i += start_at
        i = i % files
        with open("exps{}_NEC.sh".format(i + 1), "a") as f:
            # for cc in commands[i::files]:
            if len(commands) > 0:
                print("Writing to exps{}_NEC.sh".format(i + 1))
                cc = commands[0]
                f.write("{}\n".format(cc))
                commands = commands[1:]

    # Write to the file running the experiments
    if not append:
        exp_num = 1
        with open("run_experiments_NEC.sh", "w") as f:
            for _ in range(exps_per_gpu):
                for g in range(gpus):
                    g += gpu_start
                    if exp_num == exps_per_gpu * gpus:
                        f.write("CUDA_VISIBLE_DEVICES='{}' bash exps{}_NEC.sh\n".format(g, exp_num))
                    else:
                        f.write("screen -mdS {}_Exps bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='{}' bash exps{}_NEC.sh\"\n".format(exp_num, g, exp_num))
                    exp_num += 1
            f.write("# {} Experiments total\n".format(uid))
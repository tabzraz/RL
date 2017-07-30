import sys
from math import ceil

envs = ["Med-Maze-{}-v0".format(size) for size in [12]]
target_network = 500
lrs = [0.0001]
counts = [True]
# cts_convs = [False]
betas = [0.0001]
t_maxs = [x * 1000 for x in [1200]]
cts_sizes = [20]
num_seeds = 8
epsilon_starts = [0.05]
epsilon_finishs = [0.05]
epsilon_steps = [50000]
batch_sizes = [(32, 1)]
xp_replay_sizes = [x * 1000 for x in [400, 500]]
stale_limits = [x * 1000 for x in [1000]]
epsilon_scaling = [True]
epsilon_decay = [0.9999]

n_steps = [100]
variable_n_step = False

negative_rewards = [(False, 0)]
# negative_reward_scaler = [0.9]
reward_clips = [-1]

# state_action_modes = ["Plain", "Force", "Optimistic"]
state_action_modes = [None] #["Optimistic"]
optimism_scalers = [0.001]
bandit_no_epsilon_scaling = True #HACK

n_step_mixings = [1.0]

options = [False]

gpu = True
# debug_eval = True
screen = False
random_macros = False
with_primitives = False
files = 16
# (Prioritised, I.S. correction, Negative td error scaler, Subtract pseudo rewards, alpha)
# prioritizeds = [(True, False, 8, False, 0.5)]  # [(True, False, True), (True, True, True)]

alphas = [0.5]
prioritiseds = [False]
is_corrections = [False]
minus_pseudos = [False]
negative_td_scalers = [1]

prioritizeds = [(p, p_is, n_td, m_pseudos, alpha) for p in prioritiseds for p_is in is_corrections for n_td in negative_td_scalers for alpha in alphas for m_pseudos in minus_pseudos]

count_td_scalers = [1]
density_priority = False
eligibility_trace = False
gammas = [0.9999]

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
exps_per_gpu = 1
files = gpus * exps_per_gpu

gpu_start = 0

tar = True

uid = 0
commands = []
for env in envs:
    for t_max, lr in [(t, l) for t in t_maxs for l in lrs]:
        for n_step, gamma in [(n, g) for n in n_steps for g in gammas]:
            for count_td_scaler in count_td_scalers:
                for batch_size, iters, xp_replay_size in [(b, i, x) for (b, i) in batch_sizes for x in xp_replay_sizes]:
                    for eps, eps_steps, eps_finish in zip(epsilon_starts, epsilon_steps, epsilon_finishs):
                        for count, eps_scaling in [(c, e) for c in counts for e in epsilon_scaling if not (c is False and e is True)]:
                            for eps_decay in epsilon_decay:
                                for state_action_mode in state_action_modes:
                                    for o_scaler in optimism_scalers:
                                        for stale_val in stale_limits:
                                            for beta in betas:
                                                for cts_size in cts_sizes:
                                                    for neg_reward, neg_reward_scaler in negative_rewards:
                                                        for prioritized, is_weight, neg_scaler, sub_pseudo_reward, alpha in prioritizeds:
                                                            for n_mixing in n_step_mixings:
                                                                for set_replay, set_replay_num in set_replays:
                                                                    for double in doubles:
                                                                        for reward_clip in reward_clips:
                                                                            for seed in seeds:

                                                                                if state_action_mode != None and count is False:
                                                                                    continue
                                                                                if bandit_no_epsilon_scaling and state_action_mode == "Optimistic":
                                                                                    eps_scaling = False
                                                                                if set_replay:
                                                                                    xp_replay_size_ = t_max
                                                                                else:
                                                                                    xp_replay_size_ = xp_replay_size

                                                                                name = env.replace("-", "_")[:-3]
                                                                                if variable_n_step:
                                                                                    name += "_Variable"
                                                                                name += "_{}_Step_{}_Mix".format(n_step, n_mixing)
                                                                                name += "_LR_{}".format(lr)
                                                                                name += "_Gamma_{}".format(gamma)
                                                                                name += "_Batch_{}_Iters_{}_XpSize_{}k".format(batch_size, iters, str(xp_replay_size_)[:-3])
                                                                                if big_model:
                                                                                    name += "_Big"
                                                                                if prioritized:
                                                                                    name += "_Prioritized_{}_Alpha_{}_NScaler".format(alpha, neg_scaler)
                                                                                    if density_priority:
                                                                                        name += "_DensityP"
                                                                                    if is_weight:
                                                                                        name += "_IS"
                                                                                    if sub_pseudo_reward:
                                                                                        name += "_{}_MinusPseudo".format(count_td_scaler)
                                                                                if set_replay:
                                                                                    name += "_SetReplay_{}".format(set_replay_num)
                                                                                name += "_BonusClip_{}".format(reward_clip)
                                                                                # if option:
                                                                                #     if random_macros:
                                                                                #         name += "_Rnd_Macros_{}_Length_{}_Mseed_{}_Primitives_{}".format(num_macro, max_macro_length, macro_seed, with_primitives)
                                                                                #     else:
                                                                                #         name += "_Options"
                                                                                if double:
                                                                                    name += "_Double"
                                                                                if tabular:
                                                                                    name += "_TABULAR"
                                                                                if count:
                                                                                    stale = stale_val #int(xp_replay_size * stale_val)
                                                                                    if neg_reward:
                                                                                        name += "_NegativeReward_{}".format(neg_reward_scaler)
                                                                                    # print(stale)
                                                                                    if eps_scaling:
                                                                                        name += "_CountEps_{}_Decay".format(eps_decay)
                                                                                    if state_action_mode == "Plain":
                                                                                        name += "_StateAction"
                                                                                    elif state_action_mode == "Optimistic":
                                                                                        name += "_OptimisticAction_{}_Scaler".format(o_scaler)
                                                                                    name += "_Count_Cts_{}_Stale_{}k_Beta_{}_Eps_{}_{}_uid_{}".format(cts_size, str(stale)[:-3], beta, eps, eps_finish, uid)
                                                                                else:
                                                                                    name += "_DQN_Eps_{}_{}_uid_{}".format(eps, eps_finish, uid)
                                                                                python_command = "python3 ../Main.py --name {} --env {} --lr {} --seed {} --t-max {} --eps-start {} --batch-size {} --xp {}".format(name, env, lr, seed, t_max, eps, batch_size, xp_replay_size_)
                                                                                python_command += " --epsilon-finish {}".format(eps_finish)
                                                                                python_command += " --target {}".format(target_network)
                                                                                python_command += " --logdir ../Logs"
                                                                                python_command += " --gamma {}".format(gamma)
                                                                                python_command += " --eps-steps {}".format(eps_steps)
                                                                                python_command += " --n-step {} --n-step-mixing {}".format(n_step, n_mixing)
                                                                                python_command += " --iters {}".format(iters)
                                                                                python_command += " --bonus-clip {}".format(reward_clip)
                                                                                if variable_n_step:
                                                                                    python_command += " --variable-n-step"
                                                                                if tabular:
                                                                                    python_command += " --tabular"
                                                                                if big_model:
                                                                                    python_command += " --model {}-Big".format(env)
                                                                                if set_replay:
                                                                                    python_command += " --set-replay"
                                                                                    python_command += " --set-replay-num {}".format(set_replay_num)
                                                                                if prioritized:
                                                                                    python_command += " --priority --negative-td-scaler {}".format(neg_scaler)
                                                                                    python_command += " --alpha {}".format(alpha)
                                                                                    if density_priority:
                                                                                        python_command += " --density-priority"
                                                                                    if is_weight:
                                                                                        python_command += " --prioritized-is"
                                                                                    if sub_pseudo_reward:
                                                                                        python_command += " --count-td-priority"
                                                                                        python_command += " --count-td-scaler {}".format(count_td_scaler)
                                                                                if double:
                                                                                    python_command += " --double"
                                                                                if count:
                                                                                    python_command += " --count --beta {} --cts-size {}".format(beta, cts_size)
                                                                                    python_command += " --stale-limit {}".format(stale)
                                                                                    if neg_reward:
                                                                                        python_command += " --negative-rewards --negative-reward-threshold {}".format(neg_reward_scaler)
                                                                                    if eps_scaling:
                                                                                        python_command += " --count-epsilon"
                                                                                        python_command += " --epsilon-decay --decay-rate {}".format(eps_decay)
                                                                                    if state_action_mode == "Plain":
                                                                                        python_command += " --count-state-action"
                                                                                    elif state_action_mode == "Optimistic":
                                                                                        python_command += " --optimistic-init --optimistic-scaler {}".format(o_scaler)
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
            with open("exps{}.sh".format(i + 1), "w") as f:
                f.write("")
                print("Cleared exps{}.sh".format(i + 1))

    for i in range(uid):
        i += start_at
        i = i % files
        with open("exps{}.sh".format(i + 1), "a") as f:
            # for cc in commands[i::files]:
            if len(commands) > 0:
                print("Writing to exps{}.sh".format(i + 1))
                cc = commands[0]
                f.write("{}\n".format(cc))
                commands = commands[1:]

    # Write to the file running the experiments
    if not append:
        exp_num = 1
        with open("run_experiments.sh", "w") as f:
            for _ in range(exps_per_gpu):
                for g in range(gpus):
                    g += gpu_start
                    if exp_num == exps_per_gpu * gpus:
                        f.write("CUDA_VISIBLE_DEVICES='{}' bash exps{}.sh\n".format(g, exp_num))
                    else:
                        f.write("screen -mdS {}_Exps bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='{}' bash exps{}.sh\"\n".format(exp_num, g, exp_num))
                    exp_num += 1
            f.write("# {} Experiments total\n".format(uid))

import sys
from math import ceil

envs = ["Thin-Maze-{}-Neg-v0".format(size) for size in [8]]
target_network = 1000
lrs = [0.0001]
counts = [True]
# cts_convs = [False]
betas = [0.001]
t_maxs = [x * 1000 for x in [500]]
cts_sizes = [12]
num_seeds = 4
epsilon_starts = [0.05]
epsilon_finishs = [0.05]
epsilon_steps = [1]
batch_sizes = [(32, 1)]
xp_replay_sizes = [x * 1000 for x in [10, 50]]
stale_limits = [x * 1000 for x in [1000]]
epsilon_scaling = [True]
epsilon_decay = [0.9999]

n_steps = [1]
variable_n_step = False

negative_rewards = [(False, 0)]
# negative_reward_scaler = [0.9]
reward_clips = [-1]

# state_action_modes = ["Plain", "Force", "Optimistic"]
# state_action_modes = [None]
optimism_scalers = [0]
state_action_modes = [None]
force_scalers = [0] 
bandit_no_epsilon_scaling = True #HACK
ucb_bandit = False

goal_intervals = [x * 1000 for x in [5, 10]]
goal_thresholds = [0.75, 0.9]
goal_iters = [x * 1000 for x in [10]]
goal_max_steps = [x * 100 for x in [1]]

options = [False]

gpu = True
# debug_eval = True
screen = False
random_macros = False
with_primitives = False
files = 16
# (Prioritised, I.S. correction, Negative td error scaler, Subtract pseudo rewards, alpha)
# prioritizeds = [(True, False, 8, False, 0.5)]  # [(True, False, True), (True, True, True)]

alphas = [0.3, 0.4, 0.5, 0.6, 0.7]
prioritiseds = [False]
is_corrections = [False]
minus_pseudos = [False]
negative_td_scalers = [1]

prioritizeds = [(p, p_is, n_td, m_pseudos, alpha) for p in prioritiseds if p for p_is in is_corrections for n_td in negative_td_scalers for alpha in alphas for m_pseudos in minus_pseudos ]
if False in prioritiseds:
    prioritizeds += [(False, False, 1, 0, 0.5)]

count_td_scalers = [1]
density_priority = False
eligibility_trace = False
gammas = [0.99]

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
            for count_td_scaler in count_td_scalers:
                for batch_size, iters, xp_replay_size in [(b, i, x) for (b, i) in batch_sizes for x in xp_replay_sizes]:
                    for eps, eps_steps, eps_finish in zip(epsilon_starts, epsilon_steps, epsilon_finishs):
                        for count, eps_scaling in [(c, e) for c in counts for e in epsilon_scaling if not (c is False and e is True)]:
                            for eps_decay in epsilon_decay:
                                for state_action_mode, o_scaler, f_scaler in zip(state_action_modes, optimism_scalers, force_scalers):
                                    for stale_val in stale_limits:
                                        for beta in betas:
                                            for cts_size in cts_sizes:
                                                for neg_reward, neg_reward_scaler in negative_rewards:
                                                    for prioritized, is_weight, neg_scaler, sub_pseudo_reward, alpha in prioritizeds:
                                                        for g_interval in goal_intervals:
                                                            for g_threshold in goal_thresholds:
                                                                for g_iters in goal_iters:
                                                                    for g_max_step in goal_max_steps:
                                                                        for seed in seeds:

                                                                            if state_action_mode != None and count is False:
                                                                                continue
                                                                            if bandit_no_epsilon_scaling and state_action_mode != None:
                                                                                eps_scaling = False

                                                                            xp_replay_size_ = xp_replay_size

                                                                            name = env.replace("-", "_")[:-3]
                                                                            if variable_n_step:
                                                                                name += "_Variable"
                                                                            name += "_{}_stp".format(n_step)
                                                                            name += "_LR_{}".format(lr)
                                                                            name += "_Gamma_{}".format(gamma)
                                                                            name += "_Batch_{}_itrs_{}_Xp_{}k".format(batch_size, iters, str(xp_replay_size_)[:-3])
                                                                            if prioritized:
                                                                                name += "_Prioritized_{}_Alpha_{}_NScaler".format(alpha, neg_scaler)
                                                                                if density_priority:
                                                                                    name += "_DensityP"
                                                                                if is_weight:
                                                                                    name += "_IS"
                                                                                if sub_pseudo_reward:
                                                                                    name += "_{}_MinusPseudo".format(count_td_scaler)

                                                                            name += "_GoalDQN_{}_I_{}_Thr_{}_itrs_{}_mx".format(g_interval, g_threshold, g_iters, g_max_step)
                                                                            # name += "_BonusClip_{}".format(reward_clip)
                                                                            # if option:
                                                                            #     if random_macros:
                                                                            #         name += "_Rnd_Macros_{}_Length_{}_Mseed_{}_Primitives_{}".format(num_macro, max_macro_length, macro_seed, with_primitives)
                                                                            #     else:
                                                                            #         name += "_Options"
                                                                            if tabular:
                                                                                name += "_TABULAR"
                                                                            if count:
                                                                                stale = stale_val #int(xp_replay_size * stale_val)
                                                                                if neg_reward:
                                                                                    name += "_NegativeReward_{}".format(neg_reward_scaler)
                                                                                # print(stale)
                                                                                if eps_scaling:
                                                                                    name += "_CEps_{}_Decay".format(eps_decay)
                                                                                if state_action_mode == "Plain":
                                                                                    name += "_StateAction"
                                                                                elif state_action_mode == "Optimistic":
                                                                                    if ucb_bandit:
                                                                                        name += "_UCB"
                                                                                    name += "_Bandit_{}_Scaler".format(o_scaler)
                                                                                elif state_action_mode == "Force":
                                                                                    name += "_ForceAction_{}_FCount".format(f_scaler)

                                                                                name += "_Count_{}_Stle_{}k_Beta_{}_Eps_{}_{}_{}k_uid_{}".format(cts_size, str(stale)[:-3], beta, eps, eps_finish, str(eps_steps)[:-3], uid)
                                                                            else:
                                                                                name += "_DQN_Eps_{}_{}_uid_{}".format(eps, eps_finish, uid)
                                                                            python_command = "python3 ../Main.py --name {} --env {} --lr {} --seed {} --t-max {} --eps-start {} --batch-size {} --xp {}".format(name, env, lr, seed, t_max, eps, batch_size, xp_replay_size_)
                                                                            python_command += " --epsilon-finish {}".format(eps_finish)
                                                                            python_command += " --target {}".format(target_network)
                                                                            python_command += " --logdir ../Logs"
                                                                            python_command += " --gamma {}".format(gamma)
                                                                            python_command += " --eps-steps {}".format(eps_steps)
                                                                            python_command += " --n-step {}".format(n_step)
                                                                            python_command += " --iters {}".format(iters)
                                                                            if variable_n_step:
                                                                                python_command += " --variable-n-step"
                                                                            if tabular:
                                                                                python_command += " --tabular"
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
                                                                                elif state_action_mode == "Force":
                                                                                    python_command += " --force-low-count-action --min-action-count {}".format(f_scaler)
                                                                                if ucb_bandit:
                                                                                    python_command += " --ucb"
                                                                            # parser.add_argument("--goal-state-interval", type=int, default=100)
                                                                            # parser.add_argument("--goal-state-threshold", type=float, default=0.5)
                                                                            # parser.add_argument("--goal-iters", type=int, default=100)
                                                                            # parser.add_argument("--max-option-steps", type=int, default=1000)
                                                                            python_command += " --goal-dqn"
                                                                            python_command += " --goal-state-interval {}".format(g_interval)
                                                                            python_command += " --goal-state-threshold {}".format(g_threshold)
                                                                            python_command += " --goal-iters {}".format(g_iters)
                                                                            python_command += " --max-option-steps {}".format(g_max_step)
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
            with open("exps_goal{}.sh".format(i + 1), "w") as f:
                f.write("")
                print("Cleared exps_goal{}.sh".format(i + 1))

    for i in range(uid):
        i += start_at
        i = i % files
        with open("exps_goal{}.sh".format(i + 1), "a") as f:
            # for cc in commands[i::files]:
            if len(commands) > 0:
                print("Writing to exps_goal{}.sh".format(i + 1))
                cc = commands[0]
                f.write("{}\n".format(cc))
                commands = commands[1:]

    # Write to the file running the experiments
    if not append:
        exp_num = 1
        with open("run_goal_experiments.sh", "w") as f:
            for _ in range(exps_per_gpu):
                for g in range(gpus):
                    g += gpu_start
                    if exp_num == exps_per_gpu * gpus:
                        f.write("CUDA_VISIBLE_DEVICES='{}' bash exps_goal{}.sh\n".format(g, exp_num))
                    else:
                        f.write("screen -mdS {}_Exps bash -c \"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='{}' bash exps_goal{}.sh\"\n".format(exp_num, g, exp_num))
                    exp_num += 1
            f.write("# {} Experiments total\n".format(uid))

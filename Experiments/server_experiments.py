from math import ceil
import datetime
import os


import sys
from math import ceil

exps_batch_name = "Mario_Test"
exps_batch_name += "__{}".format(datetime.datetime.now().strftime("%Y_%m_%d"))

# envs = ["DoomMazeHard-v0"] 
envs = ["Mario-1-1-v0"]
# envs = ["Thin-Maze-{}-Neg-v0".format(size) for size in [12]] 
# envs = ["Empty-Room-{}-v0".format(20)]
DOOM = False
if "Doom" in envs[0]:
    DOOM = True
target_network = 1000

eval_interval = 100
vis_interval = 100
exploration_steps = 500

lrs = [0.0001] # 0.0001
counts = [True]
# cts_convs = [False]
betas = [0.001] # 0.001
t_maxs = [x * 1000 for x in [1200]]
cts_sizes = [12] #[12]
num_seeds = 2
# num_seeds = 2
epsilon_starts = [1]
epsilon_finishs = [0.05]
epsilon_steps = [1]
batch_sizes = [(32, 1)]
xp_replay_sizes = [x * 1000 for x in [300]]
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
optimism_scalers = [0, 0.001, 0.01, 0.1]
bandit_ps = [1/2] #[(1/4), (1/2), (1), (2)]
state_action_modes = [None] + ["Optimistic" for _ in optimism_scalers]
force_scalers = [0 for _ in state_action_modes]
bandit_no_epsilon_scaling = True #HACK
ucb_bandits = [False for _ in state_action_modes] #[True, True, True, False, False, False]

bonus_replay = False
bonus_replay_thresholds = [0.0005, 0.001, 0.01]
bonus_replay_sizes = [x * 1000 for x in [100]]
if not bonus_replay:
    bonus_replay_thresholds = [1]
    bonus_replay_sizes = [1]

distrib_agent = False
atoms = [5, 11, 21, 51]
v_min = -1
v_max = +1
if not distrib_agent:
    atoms = [1]

model_agent = False
model_save = 10000
model_losses = [0.25, 0.5]
model_depths = [0, 1, 2]
leaf_only = True
if not model_agent:
    model_losses = [0]
    model_depths = [False]


SARSA = False
sarsa_trains = [100, 1000]
if not SARSA:
    sarsa_trains = [1]

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

alphas = [0.3, 0.5, 0.7]
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

# big_model = True
fc_model = False
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
if DOOM:
    exps_per_gpu = 1
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
                                for state_action_mode, o_scaler, f_scaler, ucb_bandit in zip(state_action_modes, optimism_scalers, force_scalers, ucb_bandits):
                                    for stale_val in stale_limits:
                                        for beta in betas:
                                            for cts_size in cts_sizes:
                                                for neg_reward, neg_reward_scaler in negative_rewards:
                                                    for prioritized, is_weight, neg_scaler, sub_pseudo_reward, alpha in prioritizeds:
                                                        for bandit_p in bandit_ps:
                                                            for set_replay, set_replay_num in set_replays:
                                                                for double in doubles:
                                                                    for bonus_replay_size, bonus_replay_threshold in [(a, b)for a in bonus_replay_sizes for b in bonus_replay_thresholds]:
                                                                        for model_loss, model_depth in [(ml, mla) for ml in model_losses for mla in model_depths]:
                                                                            for seed in seeds:

                                                                                if state_action_mode != None and count is False:
                                                                                    continue
                                                                                if bandit_no_epsilon_scaling and state_action_mode != None:
                                                                                    eps_scaling = False
                                                                                    eps = eps_finish
                                                                                    if eps_steps != 1:
                                                                                        continue
                                                                                if set_replay:
                                                                                    xp_replay_size_ = t_max
                                                                                else:
                                                                                    xp_replay_size_ = xp_replay_size

                                                                                name = env.replace("-", "_")[:-3]
                                                                                if variable_n_step:
                                                                                    name += "_Variable"
                                                                                name += "_{}_stp".format(n_step)
                                                                                name += "_LR_{}".format(lr)
                                                                                name += "_Gamma_{}".format(gamma)
                                                                                name += "_Batch_{}_itrs_{}_Xp_{}k".format(batch_size, iters, str(xp_replay_size_)[:-3])
                                                                                # if big_model:
                                                                                    # name += "_Big"
                                                                                if fc_model:
                                                                                    name += "_FC"
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
                                                                                # name += "_BonusClip_{}".format(reward_clip)
                                                                                # if option:
                                                                                #     if random_macros:
                                                                                #         name += "_Rnd_Macros_{}_Length_{}_Mseed_{}_Primitives_{}".format(num_macro, max_macro_length, macro_seed, with_primitives)
                                                                                #     else:
                                                                                #         name += "_Options"
                                                                                if double:
                                                                                    name += "_Double"
                                                                                if tabular:
                                                                                    name += "_TABULAR"
                                                                                if model_agent:
                                                                                    name += "_Model_{}_Loss_{}_Look".format(model_loss, model_depth)
                                                                                # if distrib_agent:
                                                                                #     name += "_DISTRIB_{}_Atoms".format(atom)
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
                                                                                        name += "_{}_Bandit_{}_Scaler".format(bandit_p, o_scaler)
                                                                                    elif state_action_mode == "Force":
                                                                                        name += "_ForceAction_{}_FCount".format(f_scaler)

                                                                                if bonus_replay:
                                                                                    name += "_{}k_2Replay_{}_Thr".format(str(bonus_replay_size)[:-3], bonus_replay_threshold)

                                                                                if SARSA:
                                                                                    name += "_SARSA_{}_".format(sarsa_train)
                                                                                if count:
                                                                                    name += "_Count_{}_Stle_{}k_Beta_{}_Eps_{}_{}_{}k_uid_{}".format(cts_size, str(stale)[:-3], beta, eps, eps_finish, str(eps_steps)[:-3], uid)
                                                                                else:
                                                                                    name += "_Eps_{}_{}_uid_{}".format(eps, eps_finish, uid)
                                                                                python_command = "python3 ../Main.py --name {} --env {} --lr {} --seed {} --t-max {} --eps-start {} --batch-size {} --xp {}".format(name, env, lr, seed, t_max, eps, batch_size, xp_replay_size_)
                                                                                python_command += " --epsilon-finish {}".format(eps_finish)
                                                                                python_command += " --target {}".format(target_network)
                                                                                python_command += " --logdir ../Logs/{}/".format(exps_batch_name)
                                                                                python_command += " --gamma {}".format(gamma)
                                                                                python_command += " --eps-steps {}".format(eps_steps)
                                                                                python_command += " --n-step {}".format(n_step)
                                                                                python_command += " --iters {}".format(iters)
                                                                                if SARSA:
                                                                                    python_command += " --sarsa --sarsa-train {}".format(sarsa_train)
                                                                                # python_command += " --bonus-clip {}".format(reward_clip)
                                                                                if bonus_replay:
                                                                                    python_command += " --bonus-replay-threshold {}".format(bonus_replay_threshold)
                                                                                    python_command += " --bonus-replay"
                                                                                    python_command += " --bonus-replay-size {}".format(bonus_replay_size)
                                                                                if variable_n_step:
                                                                                    python_command += " --variable-n-step"
                                                                                if tabular:
                                                                                    python_command += " --tabular"
                                                                                # if big_model:
                                                                                    # python_command += " --model {}-Big".format(env)
                                                                                if fc_model:
                                                                                    python_command += " --model {}-FC".format(env)
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
                                                                                        python_command += " --bandit-p {}".format(bandit_p)
                                                                                    elif state_action_mode == "Force":
                                                                                        python_command += " --force-low-count-action --min-action-count {}".format(f_scaler)
                                                                                    if ucb_bandit:
                                                                                        python_command += " --ucb"
                                                                                # if distrib_agent:
                                                                                #     python_command += " --distrib"
                                                                                #     python_command += " --atoms {} --v-min {} --v-max {}".format(atom, v_min, v_max)
                                                                                if model_agent:
                                                                                    python_command += " --model-dqn"
                                                                                    python_command += " --model-loss {}".format(model_loss)
                                                                                    python_command += " --model-save-image {}".format(model_save)
                                                                                    python_command += " --lookahead-depth {}".format(model_depth)
                                                                                    if leaf_only:
                                                                                        python_command += " --only-leaf"
                                                                                    # if model_lookahead:
                                                                                        # python_command += " --lookahead-plan"
                                                                                # if option:
                                                                                #     if random_macros:
                                                                                #         python_command += " --options Random_Macros --num-macros {} --max-macro-length {} --macro-seed {}".format(num_macro, max_macro_length, macro_seed)
                                                                                #         if with_primitives:
                                                                                #             python_command += " --train-primitives"
                                                                                #     else:
                                                                                #         python_command += " --options Maze_Good"

                                                                                python_command += " --eval-interval {}".format(eval_interval)
                                                                                python_command += " --interval-size {} --frontier-interval {}".format(vis_interval, vis_interval)
                                                                                python_command += " --exploration-steps {}".format(exploration_steps)
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



# Experiments = ["touch ../Logs/{}.test".format(i) for i in range(3)]
Experiments = commands

# (Server, [Gpus to use], experiments per gpu)
# Servers = [("brown", [0, 2, 3, 4, 6], 2), ("dgx1", [0, 1, 2, 3, 4, 5, 6, 7], 1), ("savitar", [0, 1, 7], 2)]
Servers = [("dgx1", [i for i in range(8)], 1)]

Central_Logs = "/data/savitar/tabhid/Runs/Servers"

num_experiments = len(Experiments)

server_ratios = [(len(gpus) * exps_per) for _, gpus, exps_per in Servers]
sum_server_ratios = sum(server_ratios)

print("--- {} Experiments Total ---".format(num_experiments))

print("\n--- {} Different Hyperparameters ---\n".format(round(uid / num_seeds)))
# print("\n--- {} Runs ---\n--- {} Files => Upto {} Runs per file ---\n".format(uid, files, ceil(uid / files)))
# print("--- {} GPUs, {} Concurrent runs per GPU ---\n".format(gpus, exps_per_gpu))

if not write_to_files:
    print("Not writing")
    exit()

path = "{}__Experiments".format(exps_batch_name)
if not os.path.exists(path):
    os.makedirs(path)

with open("{}/run_exps_on_servers.sh".format(path), "w") as f:
    f.write("\n")

with open("{}/screen_wipe.sh".format(path), "w") as f:
    f.write("\n")

with open("{}/copy_logs.sh".format(path), "w") as f:
    f.write("\n")

with open("{}/kill_screens.sh".format(path), "w") as f:
    f.write("\n")

uid = 0

for server, gpus, exps_per in Servers:
    ratio_of_exps_for_this_server = (sum_server_ratios / (len(gpus) * exps_per))
    num_exps_for_this_server = ceil(num_experiments / ratio_of_exps_for_this_server)
    exps_for_this_server = Experiments[uid: uid + num_exps_for_this_server]
    print("{} Experiments on {}".format(len(exps_for_this_server), server))

    cd_to_docker = "cd /data/{}/tabhid".format(server)
    run_docker = "bash run_docker_server.sh"
    cd_to_rl = "cd RL"
    mk_server_exps = "mkdir -p Server_Exps"
    cd_server_exps = "cd Server_Exps"

    git_pull = "git pull"

    experiment_files = ["" for _ in range(len(gpus) * exps_per)]
    for index, exp in enumerate(exps_for_this_server):
        file_to_append = index % len(experiment_files)
        experiment_files[file_to_append] += "{}\n".format(exp)
        uid += 1

    write_to_exp_files = ""
    for index, exps in enumerate(experiment_files):
        # write_to_exp_files += "\ntouch server_exps_{}.sh\n".format(index + 1)
        write_to_exp_files += "echo '{}' > server_exps_{}.sh;".format(exps, index + 1)
    write_to_exp_files = write_to_exp_files[:-1]

    make_exps_file = "touch run_server_experiments.sh"
    exps_file = ""
    exp_num = 1
    for _ in range(exps_per):
        for g in gpus:
            exps_file += "sleep {}; screen -mdS {}_Exps_{} bash -c \\\"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='{}' bash server_exps_{}.sh\\\"\n".format(exp_num, exp_num, exps_batch_name, g, exp_num)
            exp_num += 1
    # exps_file += "# {} Experiments total\n".format(num_exps_for_this_server)

    write_exps_file = "echo '{}' > run_server_experiments.sh".format(exps_file)

    run_server_exps = "bash run_server_experiments.sh"
    started_running = "echo 'Started Running Experiments on {}'".format(server)

    # ssh_commands_to_run = [cd_to_docker, run_docker, cd_to_rl, mk_server_exps, cd_server_exps, write_to_exp_files, make_exps_file, write_exps_file, run_server_exps, started_running]
    # ssh_commands_joined = " ;".join(ssh_commands_to_run)
    # screen_command = "screen -mdS server_screen bash -c \"{}\"".format(ssh_commands_joined)

    server_exps_commands = [cd_to_docker, cd_server_exps, write_to_exp_files, write_exps_file]
    server_command = ";".join(server_exps_commands)

    docker_commands = [cd_to_rl, git_pull, cd_server_exps, run_server_exps, started_running]
    docker_commands = ";".join(docker_commands)
    docker_command = "docker exec -it tabhid_exps /bin/bash -c \\\"{}\\\"".format(docker_commands)
    # print(docker_command)

    ssh_exps_command = "ssh -t {} \"{}\"\n".format(server, server_command)
    ssh_run_command = "ssh -t {} \"{}\"\n".format(server, docker_command)

    with open("{}/run_exps_on_servers.sh".format(path), "a") as f:
        f.write(ssh_exps_command)
        f.write("\necho 'Written experiments to {}'\n".format(server))
        f.write(ssh_run_command)
        f.write("\necho 'Started running {} exps on {}'\n\n\n".format(num_exps_for_this_server, server))

    # Check for dead screens
    screen_wipe = "screen -wipe"
    screen_commands = [screen_wipe]
    screen_commands = ";".join(screen_commands)
    screen_command = "docker exec -it tabhid_exps /bin/bash -c \\\"{}\\\"".format(screen_commands)

    ssh_screen_command = "server_output=$(ssh -t {} \"{}\")\n".format(server, screen_command)

    with open("{}/screen_wipe.sh".format(path), "a") as f:
        f.write("\necho {}\n".format(server))
        f.write(ssh_screen_command)
        f.write("echo \"$server_output\"\n\n")

    # Copy all experiments over to a central place (savitar at the moment)
    copy_command = "cp -r /data/{}/tabhid/Server_Logs/{} {}/{}".format(server, exps_batch_name, Central_Logs, exps_batch_name)
    ssh_copy_command = "ssh -t {} \"{}\"\n".format(server, copy_command)
    with open("{}/copy_logs.sh".format(path), "a") as f:
        f.write("\necho \"Copying {} Logs\"\n".format(server))
        f.write(ssh_copy_command)
        f.write("\necho \"Finished copying {} Logs\"\n".format(server))

    # Kill screens
    kill_screens_command = ""
    for i in range(exp_num - 1):
        kill_screens_command += "screen -X -S {}_Exps_{} kill;".format(i + 1, exps_batch_name)
    docker_kill_screens = "docker exec -it tabhid_exps /bin/bash -c \\\"{}\\\"".format(kill_screens_command)
    ssh_kill_screens = "ssh -t {} \"{}\"\n".format(server, docker_kill_screens)

    with open("{}/kill_screens.sh".format(path), "a") as f:
        f.write("\necho \"Killing screens on {}\"\n".format(server))
        f.write(ssh_kill_screens)
        f.write("\necho \"Killed screens on {}\"\n".format(server))

print("\nWritten to {}".format(exps_batch_name))

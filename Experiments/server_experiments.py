from math import ceil

Experiments = ["touch ../Logs/{}.test".format(i) for i in range(3)]

# (Server, [Gpus to use], experiments per gpu)
Servers = [("brown", [0,1,2], 1)]#, ("brown", [0, 1, 7], 2), ("dgx1", [0, 4, 7], 1)]

num_experiments = len(Experiments)

server_ratios = [(len(gpus) * exps_per) for _, gpus, exps_per in Servers]
sum_server_ratios = sum(server_ratios)

print("{} Experiments Total\n".format(num_experiments))

with open("run_exps_on_servers.sh", "w") as f:
    f.write("\n")

uid = 0

for server, gpus, exps_per in Servers:
    ratio_of_exps_for_this_server = (sum_server_ratios / (len(gpus) * exps_per))
    num_exps_for_this_server = ceil(num_experiments / ratio_of_exps_for_this_server)
    exps_for_this_server = Experiments[uid: uid + num_exps_for_this_server]
    print("{} Experiments on {}".format(len(exps_for_this_server), server))

    cd_to_docker = "cd /data/{}/tabhid".format(server)
    run_docker = "bash run_docker_server.sh"
    cd_to_rl = "cd RL/Server_Exps"
    mk_server_exps = "mkdir -p Server_Exps"
    cd_server_exps = "cd Server_Exps"

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
            exps_file += "screen -mdS {}_Exps bash -c \\\"export LD_LIBRARY_PATH='/usr/local/nvidia/lib:/usr/local/nvidia/lib64'; CUDA_VISIBLE_DEVICES='{}' bash server_exps_{}.sh\\\"\n".format(exp_num, g, exp_num)
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

    docker_commands = [cd_to_rl, run_server_exps, started_running]
    docker_commands = ";".join(docker_commands)
    docker_command = "docker exec -it tabhid_exps /bin/bash -c \\\"{}\\\"".format(docker_commands)
    # print(docker_command)

    ssh_exps_command = "ssh -t {} \"{}\"\n".format(server, server_command)
    ssh_run_command = "ssh -t {} \"{}\"\n".format(server, docker_command)

    with open("run_exps_on_servers.sh", "a") as f:
        f.write(ssh_exps_command)
        f.write("\necho 'Written experiments to {}'\n".format(server))
        f.write(ssh_run_command)
        f.write("\necho 'Started running {} exps on {}'\n\n\n".format(num_exps_for_this_server, server))




echo savitar
server_output=$(ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -wipe\"")
echo "$server_output"


echo brown
server_output=$(ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"screen -wipe\"")
echo "$server_output"


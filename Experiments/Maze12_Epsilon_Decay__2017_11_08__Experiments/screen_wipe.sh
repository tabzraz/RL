

echo savitar
server_output=$(ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -wipe\"")
echo "$server_output"


echo dgx1
server_output=$(ssh -t dgx1 "docker exec -it tabhid_exps /bin/bash -c \"screen -wipe\"")
echo "$server_output"


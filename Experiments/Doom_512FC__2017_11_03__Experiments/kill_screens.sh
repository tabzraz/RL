

echo "Killing screens on savitar"
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Doom_512FC__2017_11_03 kill;screen -X -S 2_Exps_Doom_512FC__2017_11_03 kill;screen -X -S 3_Exps_Doom_512FC__2017_11_03 kill;\""

echo "Killed screens on savitar"

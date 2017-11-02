

echo "Killing screens on savitar"
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_DoomMazeHard_25_Stp__2017_11_02 kill;screen -X -S 2_Exps_DoomMazeHard_25_Stp__2017_11_02 kill;screen -X -S 3_Exps_DoomMazeHard_25_Stp__2017_11_02 kill;\""

echo "Killed screens on savitar"

echo "Killing screens on brown"
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_DoomMazeHard_25_Stp__2017_11_02 kill;screen -X -S 2_Exps_DoomMazeHard_25_Stp__2017_11_02 kill;screen -X -S 3_Exps_DoomMazeHard_25_Stp__2017_11_02 kill;\""

echo "Killed screens on brown"

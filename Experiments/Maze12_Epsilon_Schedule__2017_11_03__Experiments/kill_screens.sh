

echo "Killing screens on brown"
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Maze12_Epsilon_Schedule__2017_11_03 kill;screen -X -S 2_Exps_Maze12_Epsilon_Schedule__2017_11_03 kill;screen -X -S 3_Exps_Maze12_Epsilon_Schedule__2017_11_03 kill;screen -X -S 4_Exps_Maze12_Epsilon_Schedule__2017_11_03 kill;screen -X -S 5_Exps_Maze12_Epsilon_Schedule__2017_11_03 kill;\""

echo "Killed screens on brown"

echo "Killing screens on savitar"
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Maze12_Epsilon_Schedule__2017_11_03 kill;screen -X -S 2_Exps_Maze12_Epsilon_Schedule__2017_11_03 kill;screen -X -S 3_Exps_Maze12_Epsilon_Schedule__2017_11_03 kill;\""

echo "Killed screens on savitar"

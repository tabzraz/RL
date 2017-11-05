

echo "Killing screens on savitar"
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Maze_12_XpSizes_v2__2017_11_05 kill;screen -X -S 2_Exps_Maze_12_XpSizes_v2__2017_11_05 kill;screen -X -S 3_Exps_Maze_12_XpSizes_v2__2017_11_05 kill;screen -X -S 4_Exps_Maze_12_XpSizes_v2__2017_11_05 kill;screen -X -S 5_Exps_Maze_12_XpSizes_v2__2017_11_05 kill;screen -X -S 6_Exps_Maze_12_XpSizes_v2__2017_11_05 kill;\""

echo "Killed screens on savitar"

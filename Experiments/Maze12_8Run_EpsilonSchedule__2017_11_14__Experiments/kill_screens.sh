

echo "Killing screens on brown"
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;screen -X -S 2_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;\""

echo "Killed screens on brown"

echo "Killing screens on dgx1"
ssh -t dgx1 "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;screen -X -S 2_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;screen -X -S 3_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;screen -X -S 4_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;\""

echo "Killed screens on dgx1"

echo "Killing screens on savitar"
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;screen -X -S 2_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;screen -X -S 3_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;screen -X -S 4_Exps_Maze12_8Run_EpsilonSchedule__2017_11_14 kill;\""

echo "Killed screens on savitar"

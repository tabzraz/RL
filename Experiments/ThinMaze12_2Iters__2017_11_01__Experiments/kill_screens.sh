

echo "Killing screens on brown"
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_ThinMaze12_2Iters__2017_11_01 kill;screen -X -S 2_Exps_ThinMaze12_2Iters__2017_11_01 kill;screen -X -S 3_Exps_ThinMaze12_2Iters__2017_11_01 kill;screen -X -S 4_Exps_ThinMaze12_2Iters__2017_11_01 kill;screen -X -S 5_Exps_ThinMaze12_2Iters__2017_11_01 kill;\""

echo "Killed screens on brown"

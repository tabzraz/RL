

echo "Killing screens on savitar"
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 2_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 3_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 4_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 5_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 6_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;\""

echo "Killed screens on savitar"

echo "Killing screens on brown"
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 2_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 3_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 4_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;\""

echo "Killed screens on brown"

echo "Killing screens on dgx1"
ssh -t dgx1 "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 2_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 3_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 4_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 5_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 6_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 7_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;screen -X -S 8_Exps_ThinMaze12_BonusReplayBandit_Reruns__2017_11_01 kill;\""

echo "Killed screens on dgx1"

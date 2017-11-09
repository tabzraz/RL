

echo "Killing screens on brown"
ssh -t brown "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 2_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 3_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;\""

echo "Killed screens on brown"

echo "Killing screens on savitar"
ssh -t savitar "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 2_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 3_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 4_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 5_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 6_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 7_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;screen -X -S 8_Exps_Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09 kill;\""

echo "Killed screens on savitar"

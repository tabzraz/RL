

echo "Killing screens on dgx1"
ssh -t dgx1 "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_DoomHard_NoNegReward__2017_11_06 kill;screen -X -S 2_Exps_DoomHard_NoNegReward__2017_11_06 kill;screen -X -S 3_Exps_DoomHard_NoNegReward__2017_11_06 kill;screen -X -S 4_Exps_DoomHard_NoNegReward__2017_11_06 kill;screen -X -S 5_Exps_DoomHard_NoNegReward__2017_11_06 kill;screen -X -S 6_Exps_DoomHard_NoNegReward__2017_11_06 kill;screen -X -S 7_Exps_DoomHard_NoNegReward__2017_11_06 kill;screen -X -S 8_Exps_DoomHard_NoNegReward__2017_11_06 kill;\""

echo "Killed screens on dgx1"

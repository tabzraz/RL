

echo "Killing screens on dgx1"
ssh -t dgx1 "docker exec -it tabhid_exps /bin/bash -c \"screen -X -S 1_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 2_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 3_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 4_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 5_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 6_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 7_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 8_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 9_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 10_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 11_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 12_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 13_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 14_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 15_Exps_EmptyRoom20__2017_11_03 kill;screen -X -S 16_Exps_EmptyRoom20__2017_11_03 kill;\""

echo "Killed screens on dgx1"

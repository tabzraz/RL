

echo "Copying brown Logs"
ssh -t savitar "cp -r /data/brown/tabhid/Server_Logs/Maze12_EpsilonSchedule_StateCounts__2017_11_05/. /data/savitar/tabhid/Runs/Servers/Maze12_EpsilonSchedule_StateCounts__2017_11_05"

echo "Finished copying brown Logs"

echo "Copying dgx1 Logs"
ssh -t savitar "cp -r /data/dgx1/tabhid/Server_Logs/Maze12_EpsilonSchedule_StateCounts__2017_11_05/. /data/savitar/tabhid/Runs/Servers/Maze12_EpsilonSchedule_StateCounts__2017_11_05"

echo "Finished copying dgx1 Logs"

echo "Copying savitar Logs"
ssh -t savitar "cp -r /data/savitar/tabhid/Server_Logs/Maze12_EpsilonSchedule_StateCounts__2017_11_05/. /data/savitar/tabhid/Runs/Servers/Maze12_EpsilonSchedule_StateCounts__2017_11_05"

echo "Finished copying savitar Logs"

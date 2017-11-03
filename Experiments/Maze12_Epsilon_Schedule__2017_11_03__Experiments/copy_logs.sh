

echo "Copying brown Logs"
ssh -t savitar "cp -r /data/brown/tabhid/Server_Logs/Maze12_Epsilon_Schedule__2017_11_03/. /data/savitar/tabhid/Runs/Servers/Maze12_Epsilon_Schedule__2017_11_03"

echo "Finished copying brown Logs"

echo "Copying savitar Logs"
ssh -t savitar "cp -r /data/savitar/tabhid/Server_Logs/Maze12_Epsilon_Schedule__2017_11_03/. /data/savitar/tabhid/Runs/Servers/Maze12_Epsilon_Schedule__2017_11_03"

echo "Finished copying savitar Logs"

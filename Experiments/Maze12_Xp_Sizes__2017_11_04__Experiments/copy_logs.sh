

echo "Copying brown Logs"
ssh -t savitar "cp -r /data/brown/tabhid/Server_Logs/Maze12_Xp_Sizes__2017_11_04/. /data/savitar/tabhid/Runs/Servers/Maze12_Xp_Sizes__2017_11_04"

echo "Finished copying brown Logs"

echo "Copying savitar Logs"
ssh -t savitar "cp -r /data/savitar/tabhid/Server_Logs/Maze12_Xp_Sizes__2017_11_04/. /data/savitar/tabhid/Runs/Servers/Maze12_Xp_Sizes__2017_11_04"

echo "Finished copying savitar Logs"

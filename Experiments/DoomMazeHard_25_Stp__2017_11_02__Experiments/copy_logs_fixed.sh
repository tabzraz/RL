

echo "Copying brown Logs"
ssh -t savitar "cp -r /data/brown/tabhid/Server_Logs/DoomMazeHard_25_Stp__2017_11_02/. /data/savitar/tabhid/Runs/Servers/DoomMazeHard_25_Stp__2017_11_02"

echo "Finished copying savitar logs"

echo "Copying savitar Logs"
ssh -t savitar "cp -r /data/savitar/tabhid/Server_Logs/DoomMazeHard_25_Stp__2017_11_02/. /data/savitar/tabhid/Runs/Servers/DoomMazeHard_25_Stp__2017_11_02"

echo "Finished copying brown Logs"



echo "Copying dgx1 Logs"
ssh -t savitar "cp -r /data/dgx1/tabhid/Server_Logs/ThinMaze12_BonusReplayBandit_Reruns__2017_11_01/. /data/savitar/tabhid/Runs/Servers/ThinMaze12_BonusReplayBandit_Reruns__2017_11_01"
echo "Copying savitar Logs"
ssh -t savitar "cp -r /data/savitar/tabhid/Server_Logs/ThinMaze12_BonusReplayBandit_Reruns__2017_11_01/. /data/savitar/tabhid/Runs/Servers/ThinMaze12_BonusReplayBandit_Reruns__2017_11_01"
echo "Copying brown Logs"
ssh -t savitar "cp -r /data/brown/tabhid/Server_Logs/ThinMaze12_BonusReplayBandit_Reruns__2017_11_01/. /data/savitar/tabhid/Runs/Servers/ThinMaze12_BonusReplayBandit_Reruns__2017_11_01"




echo "Copying brown Logs"
ssh -t savitar "cp -r /data/brown/tabhid/Server_Logs/Maze12_Bonus_Replay_Bandits_v3__2017_11_06/. /data/savitar/tabhid/Runs/Servers/Maze12_Bonus_Replay_Bandits_v3__2017_11_06"

echo "Finished copying brown Logs"

echo "Copying dgx1 Logs"
ssh -t savitar "cp -r /data/dgx1/tabhid/Server_Logs/Maze12_Bonus_Replay_Bandits_v3__2017_11_06/. /data/savitar/tabhid/Runs/Servers/Maze12_Bonus_Replay_Bandits_v3__2017_11_06"

echo "Finished copying dgx1 Logs"

echo "Copying savitar Logs"
ssh -t savitar "cp -r /data/savitar/tabhid/Server_Logs/Maze12_Bonus_Replay_Bandits_v3__2017_11_06/. /data/savitar/tabhid/Runs/Servers/Maze12_Bonus_Replay_Bandits_v3__2017_11_06"

echo "Finished copying savitar Logs"

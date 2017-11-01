

echo "Making directory"
ssh -t savitar "mkdir /data/savitar/tabhid/Runs/Servers/ThinMazeNeg12_Bonus_Replay_Bandit__2017_10_31/"

echo "Copying brown Logs"
ssh -t savitar "cp -r /data/brown/tabhid/Server_Logs/ThinMazeNeg12_Bonus_Replay_Bandit__2017_10_31/. /data/savitar/tabhid/Runs/Servers/ThinMazeNeg12_Bonus_Replay_Bandit__2017_10_31/"

echo "Finished copying brown Logs"

echo "Copying dgx1 Logs"
ssh -t savitar "cp -r /data/dgx1/tabhid/Server_Logs/ThinMazeNeg12_Bonus_Replay_Bandit__2017_10_31/. /data/savitar/tabhid/Runs/Servers/ThinMazeNeg12_Bonus_Replay_Bandit__2017_10_31/"

echo "Finished copying dgx1 Logs"

echo "Copying savitar Logs"
ssh -t savitar "cp -r /data/savitar/tabhid/Server_Logs/ThinMazeNeg12_Bonus_Replay_Bandit__2017_10_31/. /data/savitar/tabhid/Runs/Servers/ThinMazeNeg12_Bonus_Replay_Bandit__2017_10_31/"

echo "Finished copying savitar Logs"

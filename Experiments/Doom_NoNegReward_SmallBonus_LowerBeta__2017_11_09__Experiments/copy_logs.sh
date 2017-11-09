

echo "Copying brown Logs"
ssh -t savitar "cp -r /data/brown/tabhid/Server_Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/. /data/savitar/tabhid/Runs/Servers/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09"

echo "Finished copying brown Logs"

echo "Copying savitar Logs"
ssh -t savitar "cp -r /data/savitar/tabhid/Server_Logs/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09/. /data/savitar/tabhid/Runs/Servers/Doom_NoNegReward_SmallBonus_LowerBeta__2017_11_09"

echo "Finished copying savitar Logs"

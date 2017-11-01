

echo "Copying savitar Logs"
ssh -t savitar "savitar"

echo "Finished copying savitar Logs"

echo "Copying brown Logs"
ssh -t savitar "brown"

echo "Finished copying brown Logs"

echo "Copying dgx1 Logs"
ssh -t savitar "dgx1"

echo "Finished copying dgx1 Logs"

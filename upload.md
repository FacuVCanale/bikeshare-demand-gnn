1- upload: rsync -avz --progress --partial -e "ssh -p 12402"  data/ root@149.7.4.9:/workspace/EcoBici-AI/data/








2- download: rsync -avz -P -e "ssh -p 4193" --exclude='checkpoint*.pt' root@160.250.70.41:/workspace/EcoBici-AI/experiments/gnn/gnn_experiment_20250702_200859/ experiment_0_42_r2_transformer/



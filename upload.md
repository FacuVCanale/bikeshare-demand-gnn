1- upload: rsync -avz --progress --partial -e "ssh -p 4193" --exclude='data/meteo' data/ root@160.250.70.41:/workspace/EcoBici-AI/data/


2- download: rsync -avz -P -e "ssh -p 59000" root@175.28.230.22:/workspace/EcoBici-AI/data/clustered/ no_cluster_dataset/
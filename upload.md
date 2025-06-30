1- upload: 
rsync -avz --progress --partial -e "ssh -p 40319" data/raw/combined/trips.parquet root@78.82.34.72:/workspace/EcoBici-AI/data/raw/combined/


2- download: rsync -avz -P -e "ssh -p 59000" root@175.28.230.22:/workspace/EcoBici-AI/data/clustered/ no_cluster_dataset/
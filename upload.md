1- upload: rsync -avz --progress --partial -e "ssh -p 44386" data/raw/combined/trips.parquet root@24.124.32.70:/workspace/EcoBici-AI/data/raw/combined/


2- download: rsync -avz -P -e "ssh -p 59000" root@175.28.230.22:/workspace/EcoBici-AI/data/clustered/ no_cluster_dataset/
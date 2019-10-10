#! bin/sh

# Prepare Data
cd data
#sh download_data.sh
python norm_split_dataset.py

cd .. 
# Run SetRank Model
sh ./scripts/train_lambdamart_istella.sh
sh ./scripts/prepare_data_lambda_istella.sh
sh ./scripts/train_transformer_istella.sh
